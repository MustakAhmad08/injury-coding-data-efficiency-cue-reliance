"""
NEISS Injury Diagnosis SFT Training Pipeline
Fine-tunes Mistral-style instruct models on the NEISS narrative → diagnosis code task.
Uses QLoRA when CUDA + bitsandbytes are available and falls back to standard LoRA
on Apple Silicon / CPU environments.

Requirements:
    pip install -r requirements.txt
"""

import argparse
import importlib.util
import json
import os
import platform
import re
import sys
from importlib.metadata import version as package_version
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datasets import Dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


def bitsandbytes_available() -> bool:
    if BitsAndBytesConfig is None or importlib.util.find_spec("bitsandbytes") is None:
        return False
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


# ── 1. NEISS CODE DEFINITIONS ──────────────────────────────────────────────────

DIAGNOSIS_CODES = {
    41: "INGESTION: Object or substance swallowed into digestive tract",
    42: "ASPIRATION: Object or substance inhaled into respiratory tract",
    46: "BURN, ELECTRICAL: Injury from electrical current",
    47: "BURN, NOT SPECIFIED: Burns where type is not specified",
    48: "BURN, SCALD: Burn from hot liquid or steam",
    49: "BURN, CHEMICAL: Burn from chemical substance",
    50: "AMPUTATION: Loss or partial loss of body part",
    51: "BURN, THERMAL: Burn from heat, flame, or hot surface",
    52: "CONCUSSION: Head injury - standard concussion",
    53: "CONTUSION/ABRASION: Bruises and scrapes",
    54: "CRUSHING: Injury from compression between objects",
    55: "DISLOCATION: Joint displacement",
    56: "FOREIGN BODY: Object stuck in/on body part",
    57: "FRACTURE: Broken bone",
    58: "HEMATOMA: Collection of blood under skin",
    59: "LACERATION: Cut or tear in skin (standard cuts/slashes)",
    60: "DENTAL INJURY: Injury to teeth",
    61: "NERVE DAMAGE: Injury to nerves",
    62: "INTERNAL INJURY: Injury to internal organs; also closed head injury, subdural hematoma, cerebral contusion",
    63: "PUNCTURE: Hole made by sharp object",
    64: "STRAIN/SPRAIN: Muscle or ligament injury",
    65: "ANOXIA: Lack of oxygen",
    66: "HEMORRHAGE: Severe bleeding",
    67: "ELECTRIC SHOCK: Shock from electrical current",
    68: "POISONING: Toxic substance exposure",
    69: "SUBMERSION: Near drowning",
    70: "OTHER: Injuries not fitting other codes",
    71: "UNKNOWN/NOT STATED: Unknown or unspecified injury type",
    72: "AVULSION: Tissue torn or forcibly detached; degloving; skin tears",
    73: "RADIATION: Injury from radiation exposure",
    74: "DERMATITIS/CONJUNCTIVITIS: Skin or eye membrane injury",
}

BODY_PART_CODES = {
    0:  "INTERNAL",
    30: "SHOULDER",
    31: "UPPER TRUNK",
    32: "ELBOW",
    33: "LOWER ARM",
    34: "WRIST",
    35: "KNEE",
    36: "LOWER LEG",
    37: "ANKLE",
    38: "PUBIC REGION",
    75: "HEAD",
    76: "FACE",
    77: "EYEBALL",
    79: "LOWER TRUNK",
    80: "UPPER ARM",
    81: "UPPER LEG",
    82: "HAND",
    83: "FOOT",
    84: "25-50% OF BODY",
    85: "ALL OF BODY",
    87: "NOT STATED/UNKNOWN",
    88: "MOUTH",
    89: "NECK",
    92: "FINGER",
    93: "TOE",
    94: "EAR",
}

# ── 2. NEISS CODING RULES (from manual) ───────────────────────────────────────

# Diagnoses that MUST be paired with Body Part 85 (All of body)
BODY_PART_ALL_OF_BODY = {65, 67, 68, 69}

# Diagnoses that MUST be paired with Body Part 0 (Internal)
BODY_PART_INTERNAL = {41, 42}

# Head injury disambiguation
HEAD_INJURY_MAP = {
    "concussion": 52,
    "closed head injury": 62,
    "subdural hematoma": 62,
    "cerebral contusion": 62,
    "intracranial": 62,
}

# Burn disambiguation by mechanism
BURN_MAP = {
    "hot water": 48, "steam": 48, "scald": 48,
    "flame": 51, "fire": 51, "hot surface": 51, "thermal": 51,
    "chemical": 49, "acid": 49, "alkali": 49,
    "electrical": 46, "electric": 46,
}

SYSTEM_PROMPT = """You are a specialized medical coder for the U.S. Consumer Product Safety Commission (CPSC) National Electronic Injury Surveillance System (NEISS).

Your task is to analyze emergency department injury narratives and assign the correct NEISS diagnosis codes and body part codes.

CODING RULES YOU MUST FOLLOW:
1. PRIORITY: If multiple injuries, identify the MOST SEVERE as Diagnosis 1. Second most severe as Diagnosis 2.
2. EXCLUSIONS: Do NOT code unrelated illnesses (flu, diabetes), symptoms already captured by primary injury, or behavioral/mental disorders.
3. UNCONFIRMED DX: If only "probable", "possible", or "rule out" diagnoses exist, code the unconfirmed diagnosis rather than vague symptoms.
4. MISSING DATA: If no specific injury, use Diagnosis 71 (Unknown/Not Stated).

MANDATORY BODY PART RULES:
- Codes 65 (Anoxia), 67 (Electric Shock), 68 (Poisoning), 69 (Submersion) → ALWAYS use Body Part 85 (All of Body)
- Codes 41 (Ingestion), 42 (Aspiration) → ALWAYS use Body Part 0 (Internal)

DISAMBIGUATION RULES:
- "Concussion" → Code 52. "Closed head injury", "subdural hematoma", "cerebral contusion" → Code 62 (Internal Injury)
- Hot water/steam burns → Code 48 (Scald). Flame/hot surface → Code 51 (Thermal). Chemical → Code 49. Electrical → Code 46
- Standard cuts/slashes → Code 59 (Laceration). Degloving/skin tears → Code 72 (Avulsion)
- "Choking" with no other info → Code 42 (Aspiration)
- If Diagnosis 1 = 71, use the available diagnosis description in the rationale.

Respond ONLY with a valid JSON object containing Diagnosis_1_Code,
Diagnosis_1_Label, Body_Part_1_Code, Diagnosis_2_Code, Body_Part_2_Code, and
Rationale. The label must be the semantic code name and the rationale must be
one sentence grounded in the narrative. No markdown or extra text."""

DIAGNOSIS_ONLY_SYSTEM_PROMPT = """You are a specialized medical coder for NEISS-style emergency department injury narratives.

Your task is to read the injury narrative and assign the correct primary NEISS diagnosis code.

Respond ONLY with a valid JSON object using this exact key:
{"Diagnosis_1_Code": <integer>}

No explanation, no markdown, no extra text."""

SFT_INSTRUCTION = (
    "Read the emergency-department injury narrative. Assign the primary NEISS "
    "diagnosis code and body-part code, include the secondary codes when present, "
    "and give a one-sentence rationale for the primary diagnosis."
)

# ── 3. TRAINING EXAMPLE CONSTRUCTION ──────────────────────────────────────────

@dataclass
class NEISSRecord:
    narrative: str
    diag1: int
    body1: int
    diag2: Optional[int] = None
    body2: Optional[int] = None
    verbatim_dx: Optional[str] = None  # required if diag1 == 71


def validate_body_part(diag_code: int, body_code: int) -> int:
    """Enforce mandatory body part rules from NEISS coding manual."""
    if diag_code in BODY_PART_ALL_OF_BODY:
        return 85
    if diag_code in BODY_PART_INTERNAL:
        return 0
    return body_code


def build_training_example(record: NEISSRecord) -> dict:
    """
    Convert a NEISS record into an instruction-tuning example.
    Returns a dict with 'text' field in ChatML format for SFTTrainer.
    """
    # Enforce body part rules
    body1 = validate_body_part(record.diag1, record.body1)
    body2 = validate_body_part(record.diag2, record.body2) if record.diag2 else None

    # Build expected JSON output
    output = {
        "Verbatim_DX": record.verbatim_dx or "",
        "Diagnosis_1_Code": record.diag1,
        "Body_Part_1_Code": body1,
        "Diagnosis_2_Code": record.diag2,
        "Body_Part_2_Code": body2,
    }

    # Format as ChatML (Mistral instruction format)
    text = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"Injury Narrative: {record.narrative} [/INST] "
        f"{json.dumps(output, indent=None)} </s>"
    )
    return {"text": text}


# ── 4. SYNTHETIC TRAINING DATA GENERATOR ──────────────────────────────────────
# Replace this with your actual NEISS records. This generates rule-correct
# synthetic examples to demonstrate the pipeline structure.

def generate_synthetic_examples() -> list[dict]:
    """
    Generate synthetic NEISS training examples that demonstrate coding rules.
    In production: replace with real NEISS records loaded from CSV/DataFrame.
    """
    examples_raw = [
        # --- Concussion (52) vs Internal Head Injury (62) disambiguation ---
        NEISSRecord(
            narrative="17YOM fell off skateboard and struck head on pavement. LOC briefly. GCS 15 in ED. DX: Concussion.",
            diag1=52, body1=75
        ),
        NEISSRecord(
            narrative="45YOF fell from e-scooter. CT head showed subdural hematoma. Admitted to neurosurgery.",
            diag1=62, body1=75
        ),
        NEISSRecord(
            narrative="8YOM hit head on handlebar. Diagnosed with closed head injury. Observation admitted.",
            diag1=62, body1=75
        ),
        NEISSRecord(
            narrative="22YOM fell off electric skateboard. Cerebral contusion on CT. ICU admit.",
            diag1=62, body1=75
        ),
        # --- Burn disambiguation ---
        NEISSRecord(
            narrative="3YOF pulled hot coffee off counter, spilled on arm. Scald burns to left forearm.",
            diag1=48, body1=33
        ),
        NEISSRecord(
            narrative="30YOM touched exposed electrical wire. Burn to right hand from electrical current.",
            diag1=46, body1=82
        ),
        NEISSRecord(
            narrative="19YOM touched hot exhaust pipe of electric scooter. Thermal burn to right calf.",
            diag1=51, body1=36
        ),
        NEISSRecord(
            narrative="25YOF cleaning with bleach, chemical burn to left forearm.",
            diag1=49, body1=33
        ),
        # --- Laceration vs Avulsion disambiguation ---
        NEISSRecord(
            narrative="34YOM cut finger on kitchen knife. 2cm laceration to right index finger requiring sutures.",
            diag1=59, body1=92
        ),
        NEISSRecord(
            narrative="55YOF degloving injury to right hand caught in machinery. Skin completely torn away.",
            diag1=72, body1=82
        ),
        NEISSRecord(
            narrative="28YOM skin tear to left forearm when fell on electric scooter. Avulsion noted.",
            diag1=72, body1=33
        ),
        # --- Aspiration vs Ingestion (mandatory body part = 0 internal) ---
        NEISSRecord(
            narrative="2YOM choking on small toy part. X-ray confirms foreign object in trachea.",
            diag1=42, body1=0  # Aspiration → Internal
        ),
        NEISSRecord(
            narrative="18MO swallowed a coin. Abdominal X-ray shows coin in stomach.",
            diag1=41, body1=0  # Ingestion → Internal
        ),
        NEISSRecord(
            narrative="3YOF choking on grape. No further info available.",
            diag1=42, body1=0  # Choking with no other info → Aspiration
        ),
        # --- Poisoning/Anoxia/Submersion → Body Part 85 (All of body) ---
        NEISSRecord(
            narrative="4YOM ingested household cleaning product. Transferred for poison control management.",
            diag1=68, body1=85  # Poisoning → All of body
        ),
        NEISSRecord(
            narrative="16YOM near drowning in pool. Pulled from water unconscious, resuscitated in ED.",
            diag1=69, body1=85  # Submersion → All of body
        ),
        NEISSRecord(
            narrative="8YOF electric shock from power outlet. Cardiac monitoring initiated.",
            diag1=67, body1=85  # Electric shock → All of body
        ),
        # --- Fractures ---
        NEISSRecord(
            narrative="14YOM fell off manual scooter hitting curb. X-ray: displaced fracture of distal radius.",
            diag1=57, body1=34
        ),
        NEISSRecord(
            narrative="67YOF fell from electric scooter. Hip X-ray: intertrochanteric fracture left femur. Admitted.",
            diag1=57, body1=81
        ),
        # --- Dislocation ---
        NEISSRecord(
            narrative="22YOM fell off skateboard landing on outstretched hand. Anterior shoulder dislocation confirmed.",
            diag1=55, body1=30
        ),
        # --- Strain/Sprain ---
        NEISSRecord(
            narrative="33YOF tripped off electric scooter. X-ray negative. DX: ankle sprain. ACE wrap applied.",
            diag1=64, body1=37
        ),
        # --- Foreign body ---
        NEISSRecord(
            narrative="7YOF has small bead stuck in right ear canal. Removed by physician under visualization.",
            diag1=56, body1=94
        ),
        # --- Multi-diagnosis (Diag1 + Diag2) ---
        NEISSRecord(
            narrative="19YOM fell off electric scooter at speed. Laceration to forehead requiring 6 sutures. X-ray: fracture of right wrist.",
            diag1=57, body1=34, diag2=59, body2=76
        ),
        NEISSRecord(
            narrative="27YOM skateboard fall. Concussion. Also abrasions to both knees and elbows.",
            diag1=52, body1=75, diag2=53, body2=35
        ),
        NEISSRecord(
            narrative="31YOF fell from e-scooter. Right ankle fracture and dislocation of same ankle.",
            diag1=57, body1=37, diag2=55, body2=37
        ),
        # --- Unknown / Code 71 with verbatim ---
        NEISSRecord(
            narrative="Patient states they were injured using a hoverboard. No further clinical info available.",
            diag1=71, body1=87,
            verbatim_dx="hoverboard injury - no further info"
        ),
        # --- Contusion / abrasion ---
        NEISSRecord(
            narrative="11YOM fell off skateboard. Contusion to left knee and abrasion to left elbow. No fracture.",
            diag1=53, body1=35
        ),
        # --- Dental injury ---
        NEISSRecord(
            narrative="9YOM fell off scooter striking face on pavement. Two front teeth avulsed. Dental consult.",
            diag1=60, body1=88
        ),
        # --- Hematoma ---
        NEISSRecord(
            narrative="41YOM struck shin on footboard of electric scooter. Large hematoma right lower leg.",
            diag1=58, body1=36
        ),
        # --- Puncture ---
        NEISSRecord(
            narrative="35YOF stepped on nail from wooden ramp at skatepark. Puncture wound to left foot.",
            diag1=63, body1=83
        ),
        # --- Unconfirmed / probable diagnosis ---
        NEISSRecord(
            narrative="Patient presents after skateboard fall. Probable fracture right hand. X-ray pending at discharge.",
            diag1=57, body1=82  # Code unconfirmed diagnosis per NEISS rules
        ),
    ]
    return [build_training_example(r) for r in examples_raw]


# ── 5. LOAD YOUR REAL NEISS DATA ──────────────────────────────────────────────

def build_diagnosis_training_example(narrative: str, diag1: int) -> dict:
    """Build a diagnosis-only SFT example for files with text/label columns."""
    output = {"Diagnosis_1_Code": int(diag1)}
    text = (
        f"<s>[INST] <<SYS>>\n{DIAGNOSIS_ONLY_SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"Injury Narrative: {str(narrative).strip()} [/INST] "
        f"{json.dumps(output, indent=None)} </s>"
    )
    return {"text": text}


def read_tabular_file(path: str) -> pd.DataFrame:
    """Read CSV/XLS/XLSX files with pandas based on file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data file extension: {suffix}. Use .csv, .xls, or .xlsx.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to make Excel/CSV schemas easier to detect."""
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def parse_int_code(value, field_name: str) -> int:
    if pd.isna(value):
        raise ValueError(f"{field_name} is missing")
    return int(float(value))


def detect_data_mode(df: pd.DataFrame) -> str:
    columns = set(df.columns)
    if {"text", "label"}.issubset(columns):
        return "diagnosis"
    if {"narrative", "diag1", "body1"}.issubset(columns):
        return "neiss"
    raise ValueError(
        "Could not detect data schema. Expected either text/label columns "
        "or narrative/diag1/body1 columns."
    )


def load_training_file(data_path: str) -> tuple[list[dict], str]:
    """
    Load real NEISS records from CSV/XLS/XLSX and convert to training examples.

    Supported diagnosis-only columns:
        text            : free-text ED narrative
        label           : primary diagnosis code (int)

    Supported full NEISS columns:
        narrative       : free-text ED narrative
        diag1           : primary diagnosis code (int)
        body1           : primary body part code (int)
        diag2           : secondary diagnosis code (int or NaN)
        body2           : secondary body part code (int or NaN)
        verbatim_dx     : verbatim DX string for code 71 (str or NaN)

    Returns (training examples, data mode).
    """
    if Path(data_path).suffix.lower() == ".jsonl":
        examples = []
        with Path(data_path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                required = {"instruction", "narrative", "response"}
                if not required.issubset(record):
                    raise ValueError(f"Missing SFT fields at {data_path}:{line_number}")
                examples.append({key: record[key] for key in required})
        return examples, "neiss"

    df = normalize_columns(read_tabular_file(data_path))
    mode = detect_data_mode(df)
    examples = []

    for _, row in df.iterrows():
        try:
            if mode == "diagnosis":
                examples.append(
                    build_diagnosis_training_example(
                        narrative=row["text"],
                        diag1=parse_int_code(row["label"], "label"),
                    )
                )
            else:
                record = NEISSRecord(
                    narrative=str(row["narrative"]).strip(),
                    diag1=parse_int_code(row["diag1"], "diag1"),
                    body1=parse_int_code(row["body1"], "body1"),
                    diag2=parse_int_code(row["diag2"], "diag2") if pd.notna(row.get("diag2")) else None,
                    body2=parse_int_code(row["body2"], "body2") if pd.notna(row.get("body2")) else None,
                    verbatim_dx=str(row["verbatim_dx"]).strip()[:50]
                                if pd.notna(row.get("verbatim_dx")) else None,
                )
                examples.append(build_training_example(record))
        except Exception as e:
            print(f"Skipping row due to error: {e}")
    return examples, mode


# ── 6. DATASET PREPARATION ────────────────────────────────────────────────────

def prepare_dataset(
    data_path: str = None,
    validation_path: str = None,
    test_size: float = 0.1,
    include_synthetic: bool = False,
):
    """
    Build HuggingFace Dataset from real NEISS CSV/XLS/XLSX records.
    The held-out test set must NOT appear here.
    """
    all_examples = generate_synthetic_examples() if include_synthetic else []
    data_mode = "neiss"

    if data_path:
        real_examples, data_mode = load_training_file(data_path)
        all_examples += real_examples
        print(f"Loaded {len(real_examples)} records from {data_path}.")

    if not all_examples:
        raise ValueError("No training examples were loaded. Provide a train file or use --include-synthetic.")

    if validation_path:
        val_raw, validation_mode = load_training_file(validation_path)
        if validation_mode != data_mode:
            raise ValueError("Training and validation files use different data schemas")
        print(f"Loaded fixed validation split with {len(val_raw)} records from {validation_path}.")
        return Dataset.from_list(all_examples), Dataset.from_list(val_raw), data_mode

    print(f"Total training examples: {len(all_examples)}")
    train_raw, val_raw = train_test_split(
        all_examples, test_size=test_size, random_state=42
    )
    train_ds = Dataset.from_list(train_raw)
    val_ds   = Dataset.from_list(val_raw)
    return train_ds, val_ds, data_mode


# ── 7. MODEL LOADING (QLoRA / 4-bit) ─────────────────────────────────────────

def detect_runtime() -> tuple[str, bool]:
    """
    Return the best available device and whether 4-bit QLoRA is supported.
    """
    if torch.cuda.is_available():
        return "cuda", bitsandbytes_available()
    if torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


def configure_cuda_for_rtx() -> None:
    """
    Enable fast CUDA math paths for NVIDIA RTX cards.
    RTX 5090 supports bf16 and TF32, which are ideal for this LoRA/QLoRA job.
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def print_cuda_summary() -> None:
    if not torch.cuda.is_available():
        return
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    capability = f"{props.major}.{props.minor}"
    print(f"CUDA GPU: {props.name} | VRAM: {total_gb:.1f} GB | Capability: {capability}")
    if props.major >= 12:
        print("Detected Blackwell-class GPU. Use a PyTorch/CUDA build that supports sm_120.")


def load_model_and_tokenizer(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    use_4bit_override: Optional[bool] = None,
):
    """
    Load the model for the best available runtime.
    CUDA uses 4-bit QLoRA when bitsandbytes is installed.
    Apple Silicon / CPU fall back to standard model loading.
    """
    configure_cuda_for_rtx()
    print_cuda_summary()
    device, detected_4bit = detect_runtime()
    use_4bit = detected_4bit if use_4bit_override is None else use_4bit_override
    if use_4bit and not detected_4bit:
        raise RuntimeError(
            "4-bit QLoRA was requested, but bitsandbytes is not installed or is not usable. "
            "Install a CUDA-compatible bitsandbytes build, or rerun with --no-4bit."
        )
    print(f"Runtime device: {device} | 4-bit QLoRA: {use_4bit}")

    model_kwargs = {"trust_remote_code": True}

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
        model_kwargs["attn_implementation"] = "sdpa"
    else:
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
            model_kwargs["attn_implementation"] = "sdpa"
        if device == "mps":
            model_kwargs["torch_dtype"] = torch.float16
        elif device == "cpu":
            model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    # Some instruct checkpoints store sampling-only defaults even when greedy
    # decoding is requested. Clear the inactive values to avoid one warning per
    # example; this does not alter do_sample=False generation.
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    elif device in {"mps", "cpu"}:
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


# ── 8. LORA CONFIGURATION ─────────────────────────────────────────────────────

def get_lora_config() -> LoraConfig:
    """
    QLoRA configuration targeting attention and MLP layers.
    r=16 is a good balance between capacity and VRAM usage for a 7B model.
    """
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ── 9. TRAINING ARGUMENTS ─────────────────────────────────────────────────────

def get_training_args(
    output_dir: str = "./neiss_mistral_sft",
    batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    gradient_checkpointing: bool = True,
    use_4bit: Optional[bool] = None,
    seed: int = 20260718,
) -> TrainingArguments:
    device, detected_4bit = detect_runtime()
    effective_4bit = detected_4bit if use_4bit is None else use_4bit
    use_bf16 = device == "cuda"
    worker_count = 0 if platform.system() in {"Darwin", "Windows"} else min(4, os.cpu_count() or 1)
    per_device_train_batch_size = batch_size or (8 if effective_4bit else 2)
    per_device_eval_batch_size = eval_batch_size or per_device_train_batch_size
    grad_accum = gradient_accumulation_steps or (2 if effective_4bit else 8)

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=gradient_checkpointing,
        optim="paged_adamw_8bit" if effective_4bit else "adamw_torch",
        learning_rate=2e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=False,
        bf16=use_bf16,
        tf32=device == "cuda",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",                    # swap to "wandb" if you use W&B
        dataloader_num_workers=worker_count,
        seed=seed,
        data_seed=seed,
    )


# ── 10. INFERENCE & EVALUATION ────────────────────────────────────────────────

def get_prompt_for_mode(data_mode: str) -> str:
    return DIAGNOSIS_ONLY_SYSTEM_PROMPT if data_mode == "diagnosis" else SYSTEM_PROMPT


def render_chat_template(tokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    """Render shared conversation content with the model's native chat wrapper."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    system = messages[0]["content"]
    user = messages[1]["content"]
    if add_generation_prompt:
        return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
    assistant = messages[2]["content"]
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant} </s>"


def format_sft_dataset(dataset: Dataset, tokenizer, data_mode: str) -> Dataset:
    """Apply the Mistral or Qwen tokenizer's native template to SFT records."""
    required = {"instruction", "narrative", "response"}
    if not required.issubset(dataset.column_names):
        return dataset
    system_prompt = get_prompt_for_mode(data_mode)

    def format_row(row: dict) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{row['instruction']}\n\nInjury Narrative: {row['narrative']}",
            },
            {"role": "assistant", "content": row["response"]},
        ]
        return {"text": render_chat_template(tokenizer, messages, add_generation_prompt=False)}

    return dataset.map(format_row, desc="Applying model-native chat template")


def load_fewshot_examples(path: str | Path) -> list[dict]:
    """Load fixed demonstrations used identically for every baseline model."""
    examples = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not {"narrative", "response"}.issubset(record):
                raise ValueError(f"Invalid demonstration at {path}:{line_number}")
            examples.append(record)
    return examples


def parse_model_response(response: str) -> dict:
    """Parse JSON from model output, with a conservative diagnosis-code fallback."""
    response = response.strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*?\}", response, flags=re.DOTALL)
    if json_match:
        json_text = json_match.group(0)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            fixed = re.sub(r"#.*", "", json_text).strip()
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    code_match = re.search(
        r"(?:Diagnosis_1_Code|diagnosis code|primary diagnosis code)[^0-9]{0,40}([0-9]{2})",
        response,
        flags=re.IGNORECASE,
    )
    if code_match:
        return {
            "Diagnosis_1_Code": int(code_match.group(1)),
            "raw_output": response,
            "parse_fallback": True,
        }

    return {"raw_output": response, "parse_error": True}


def predict(
    narrative: str,
    model,
    tokenizer,
    data_mode: str = "neiss",
    max_new_tokens: int = 256,
    demonstrations: Optional[list[dict]] = None,
) -> dict:
    """
    Run inference on a single narrative. Returns parsed JSON output.
    Use this to evaluate your held-out test set.
    """
    system_prompt = get_prompt_for_mode(data_mode)
    messages = [{"role": "system", "content": system_prompt}]
    for example in demonstrations or []:
        instruction = example.get("instruction", SFT_INSTRUCTION)
        messages.extend([
            {
                "role": "user",
                "content": f"{instruction}\n\nInjury Narrative: {example['narrative']}",
            },
            {"role": "assistant", "content": example["response"]},
        ])
    messages.append({
        "role": "user",
        "content": f"{SFT_INSTRUCTION}\n\nInjury Narrative: {narrative}",
    })
    prompt = render_chat_template(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return parse_model_response(response)


def evaluate_test_set(
    test_path: str,
    model,
    tokenizer,
    data_mode: str,
    max_new_tokens: int = 256,
    demonstrations: Optional[list[dict]] = None,
    inference_batch_size: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate model on held-out test set.
    Returns predictions and summary metrics.
    """
    df = normalize_columns(read_tabular_file(test_path))
    test_mode = detect_data_mode(df)
    if test_mode != data_mode:
        raise ValueError(f"Train data mode is {data_mode}, but test data mode is {test_mode}.")

    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be at least 1")

    system_prompt = get_prompt_for_mode(data_mode)
    prompts = []
    for _, row in df.iterrows():
        messages = [{"role": "system", "content": system_prompt}]
        for example in demonstrations or []:
            instruction = example.get("instruction", SFT_INSTRUCTION)
            messages.extend([
                {
                    "role": "user",
                    "content": f"{instruction}\n\nInjury Narrative: {example['narrative']}",
                },
                {"role": "assistant", "content": example["response"]},
            ])
        narrative_col = "text" if data_mode == "diagnosis" else "narrative"
        messages.append({
            "role": "user",
            "content": f"{SFT_INSTRUCTION}\n\nInjury Narrative: {row[narrative_col]}",
        })
        prompts.append(render_chat_template(tokenizer, messages, add_generation_prompt=True))

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    predictions = []
    for start in range(0, len(prompts), inference_batch_size):
        batch_prompts = prompts[start:start + inference_batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_width = inputs.input_ids.shape[-1]
        for output in outputs:
            response = tokenizer.decode(output[prompt_width:], skip_special_tokens=True)
            predictions.append(parse_model_response(response))
        completed = min(start + len(batch_prompts), len(prompts))
        if completed % 25 == 0 or completed == len(prompts):
            print(f"  Evaluated {completed}/{len(prompts)} examples", flush=True)
    tokenizer.padding_side = original_padding_side

    results = []
    for (idx, row), pred in zip(df.iterrows(), predictions):
        narrative_col = "text" if data_mode == "diagnosis" else "narrative"
        label_col = "label" if data_mode == "diagnosis" else "diag1"
        true_diag1 = parse_int_code(row[label_col], label_col)
        pred_diag1 = pred.get("Diagnosis_1_Code")
        try:
            pred_diag1 = int(float(pred_diag1))
        except (TypeError, ValueError):
            pred_diag1 = None

        result = {
            "case_id": row.get("case_id", idx),
            "year": row.get("year"),
            "narrative": row[narrative_col],
            "true_diag1": true_diag1,
            "true_diagnosis_category": row.get("diagnosis_category"),
            "pred_diag1": pred_diag1,
            "diag1_correct": true_diag1 == pred_diag1,
            "parse_error": bool(pred.get("parse_error", False)),
            "raw_output": pred.get("raw_output", ""),
        }
        if data_mode == "neiss":
            result.update({
                "true_body1": parse_int_code(row["body1"], "body1"),
                "pred_body1": pred.get("Body_Part_1_Code"),
                "true_diag2": row.get("diag2"),
                "true_body2": row.get("body2"),
                "pred_diag2": pred.get("Diagnosis_2_Code"),
            })
            result["body1_correct"] = result["true_body1"] == result["pred_body1"]
        results.append(result)

    results_df = pd.DataFrame(results)
    y_true = results_df["true_diag1"].astype(int).tolist()
    y_pred = results_df["pred_diag1"].fillna(-1).astype(int).tolist()
    metrics = {
        "data_mode": data_mode,
        "test_file": test_path,
        "n_examples": int(len(results_df)),
        "diagnosis_accuracy": float(accuracy_score(y_true, y_pred)),
        "parse_error_rate": float(results_df["parse_error"].mean()),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=sorted(set(y_true)),
            zero_division=0,
            output_dict=True,
        ),
    }
    if data_mode == "neiss" and "body1_correct" in results_df:
        metrics["body_part_accuracy"] = float(results_df["body1_correct"].mean())
    return results_df, metrics


# ── 11. MAIN TRAINING SCRIPT ──────────────────────────────────────────────────

def write_run_manifest(
    output_dir: str,
    args,
    run_type: str,
    model=None,
    training_args: Optional[TrainingArguments] = None,
) -> None:
    """Write the exact software, hardware, data, and hyperparameter configuration."""
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "vram_bytes": int(props.total_memory),
            "compute_capability": f"{props.major}.{props.minor}",
        }
    packages = {}
    for name in ("torch", "transformers", "peft", "trl", "bitsandbytes", "datasets"):
        try:
            packages[name] = package_version(name)
        except Exception:
            packages[name] = None
    payload = {
        "run_type": run_type,
        "command": sys.argv,
        "model_name": args.model_name,
        "model_commit": getattr(getattr(model, "config", None), "_commit_hash", None),
        "seed": args.seed,
        "train_file": getattr(args, "train_file", None),
        "validation_file": getattr(args, "validation_file", None),
        "test_file": getattr(args, "test_file", None),
        "shots": getattr(args, "shots", None),
        "fewshot_file": getattr(args, "fewshot_file", None),
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        },
        "quantization": {
            "bits": 4 if not args.no_4bit else None,
            "type": "nf4" if not args.no_4bit else None,
            "double_quantization": not args.no_4bit,
            "compute_dtype": "bfloat16" if not args.no_4bit else None,
        },
        "training_arguments": training_args.to_dict() if training_args is not None else None,
        "early_stopping": {"patience": 3, "threshold": 0.0} if training_args is not None else None,
        "hardware": {"platform": platform.platform(), "gpu": gpu, "cuda": torch.version.cuda},
        "package_versions": packages,
    }
    os.makedirs(output_dir, exist_ok=True)
    manifest_name = f"run_manifest_{run_type}.json"
    with open(Path(output_dir) / manifest_name, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)

def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    split_dir = project_root / "data" / "neiss_temporal_2021_2025"
    parser = argparse.ArgumentParser(description="Train a NEISS SFT model.")
    parser.add_argument(
        "--model-name",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Hugging Face model ID to fine-tune.",
    )
    parser.add_argument(
        "--train-file",
        "--csv-path",
        dest="train_file",
        default=str(split_dir / "train.jsonl"),
        help="Training file (.jsonl, .csv, .xls, .xlsx). Defaults to the fixed 2021-2023 split.",
    )
    parser.add_argument(
        "--validation-file",
        default=str(split_dir / "validation.jsonl"),
        help="Fixed validation JSONL. Defaults to the held-out 2024 split.",
    )
    parser.add_argument(
        "--test-file",
        "--test-csv",
        dest="test_file",
        default=None,
        help="Locked test file for final evaluation only; use data/neiss_temporal_2021_2025/locked_test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="./neiss_mistral_sft",
        help="Directory where adapters and outputs will be written.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable bitsandbytes 4-bit loading. Useful on Windows if bitsandbytes is unavailable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device training batch size. Defaults to 8 for 4-bit CUDA, 2 for bf16 CUDA.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Per-device eval batch size. Defaults to the training batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps. Defaults to keep effective batch near 16.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length for SFT examples.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing if you prefer speed over VRAM savings.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Append built-in synthetic NEISS examples to the training file.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate the adapter already saved in --output-dir.",
    )
    parser.add_argument(
        "--baseline-eval",
        action="store_true",
        help="Evaluate the base instruct model without loading or training an adapter.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        choices=(0, 10),
        default=0,
        help="Number of fixed demonstrations for --baseline-eval.",
    )
    parser.add_argument(
        "--fewshot-file",
        default=str(split_dir / "fewshot_10.jsonl"),
        help="The fixed ten-demonstration JSONL shared by both baseline models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260718,
        help="Random seed used for training and data ordering.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per test example during evaluation.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help="Number of test prompts generated together. This affects throughput only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    model_name = args.model_name
    train_file = args.train_file
    validation_file = args.validation_file
    test_file = args.test_file
    output_dir = args.output_dir

    print("=" * 60)
    print("NEISS SFT Training Pipeline")
    print("=" * 60)

    # ── Build dataset ──
    if args.baseline_eval:
        if not test_file:
            raise ValueError("--baseline-eval requires --test-file.")
        data_mode = detect_data_mode(normalize_columns(read_tabular_file(test_file)))
        demonstrations = load_fewshot_examples(args.fewshot_file) if args.shots == 10 else []
        if len(demonstrations) != args.shots:
            raise ValueError(f"Requested {args.shots} shots but loaded {len(demonstrations)}")
        print(f"\nBaseline evaluation: {args.shots}-shot | {model_name}")
        use_4bit_override = False if args.no_4bit else None
        model, tokenizer = load_model_and_tokenizer(model_name, use_4bit_override=use_4bit_override)
        model.eval()
        os.makedirs(output_dir, exist_ok=True)
        write_run_manifest(output_dir, args, f"baseline_{args.shots}shot", model=model)
        results_df, metrics = evaluate_test_set(
            test_file,
            model,
            tokenizer,
            data_mode=data_mode,
            max_new_tokens=args.max_new_tokens,
            demonstrations=demonstrations,
            inference_batch_size=args.inference_batch_size,
        )
        results_df.to_csv(f"{output_dir}/test_results.csv", index=False)
        with open(f"{output_dir}/test_metrics.json", "w", encoding="utf-8") as stream:
            json.dump(metrics, stream, indent=2)
        print(f"  Diagnosis-1 Accuracy : {metrics['diagnosis_accuracy']:.3f}")
        print(f"  Parse Error Rate     : {metrics['parse_error_rate']:.3f}")
        return

    if args.eval_only:
        if not test_file:
            raise ValueError("--eval-only requires --test-file.")
        data_mode = detect_data_mode(normalize_columns(read_tabular_file(test_file)))
        print(f"\nEval-only mode. Data mode: {data_mode}")
        print(f"\n[1/2] Loading base model: {model_name}")
        use_4bit_override = False if args.no_4bit else None
        model, tokenizer = load_model_and_tokenizer(model_name, use_4bit_override=use_4bit_override)
        print(f"\n[2/2] Loading adapter and evaluating: {output_dir}")
        model = PeftModel.from_pretrained(model, output_dir)
        model.eval()
        os.makedirs(output_dir, exist_ok=True)
        write_run_manifest(output_dir, args, "sft_final_evaluation", model=model)
        results_df, metrics = evaluate_test_set(
            test_file,
            model,
            tokenizer,
            data_mode=data_mode,
            max_new_tokens=args.max_new_tokens,
            inference_batch_size=args.inference_batch_size,
        )
        results_df.to_csv(f"{output_dir}/test_results.csv", index=False)
        with open(f"{output_dir}/test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Diagnosis-1 Accuracy : {metrics['diagnosis_accuracy']:.3f}")
        if "body_part_accuracy" in metrics:
            print(f"  Body-Part-1 Accuracy : {metrics['body_part_accuracy']:.3f}")
        print(f"  Parse Error Rate     : {metrics['parse_error_rate']:.3f}")
        print(f"\nFull results saved to: {output_dir}/test_results.csv")
        print(f"Metrics saved to: {output_dir}/test_metrics.json")
        print("\nDone.")
        return

    print("\n[1/4] Preparing dataset...")
    train_ds, val_ds, data_mode = prepare_dataset(
        data_path=train_file,
        validation_path=validation_file,
        include_synthetic=args.include_synthetic,
    )
    print(f"  Data mode: {data_mode}")
    print(f"  Train: {len(train_ds)} examples | Val: {len(val_ds)} examples")

    # ── Load model ──
    print(f"\n[2/4] Loading model: {model_name}")
    use_4bit_override = False if args.no_4bit else None
    model, tokenizer = load_model_and_tokenizer(model_name, use_4bit_override=use_4bit_override)
    train_ds = format_sft_dataset(train_ds, tokenizer, data_mode)
    val_ds = format_sft_dataset(val_ds, tokenizer, data_mode)

    # ── Apply LoRA ──
    print("\n[3/4] Applying QLoRA adapters...")
    lora_cfg = get_lora_config()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Train ──
    print("\n[4/4] Starting SFT training...")
    os.makedirs(output_dir, exist_ok=True)
    training_args = get_training_args(
        output_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        use_4bit=False if args.no_4bit else None,
        seed=args.seed,
    )
    write_run_manifest(output_dir, args, "sft_training", model=model, training_args=training_args)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)],
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to: {output_dir}")

    # ── Evaluate on held-out test set ──
    if test_file:
        print("\nEvaluating on held-out test set...")
        results_df, metrics = evaluate_test_set(
            test_file,
            model,
            tokenizer,
            data_mode=data_mode,
            max_new_tokens=args.max_new_tokens,
            inference_batch_size=args.inference_batch_size,
        )
        results_df.to_csv(f"{output_dir}/test_results.csv", index=False)
        with open(f"{output_dir}/test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        diag1_acc = metrics["diagnosis_accuracy"]
        print(f"  Diagnosis-1 Accuracy : {diag1_acc:.3f}")
        if "body_part_accuracy" in metrics:
            print(f"  Body-Part-1 Accuracy : {metrics['body_part_accuracy']:.3f}")
        print(f"  Parse Error Rate     : {metrics['parse_error_rate']:.3f}")
        print(f"\nFull results saved to: {output_dir}/test_results.csv")
        print(f"Metrics saved to: {output_dir}/test_metrics.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
