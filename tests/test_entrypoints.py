"""Check relocated entry points without scientific packages, data, or model/API calls."""
import ast
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def test_repository_roots(self):
        checked = 0
        for folder in ('methods', 'analysis'):
            for source in (ROOT / folder).rglob('*.py'):
                tree = ast.parse(source.read_text(encoding='utf-8'))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign):
                        continue
                    if not any(isinstance(t, ast.Name) and t.id in {'ROOT', 'PROJECT_ROOT', 'project_root', 'root'} for t in node.targets):
                        continue
                    expression = ast.unparse(node.value)
                    if '__file__' not in expression and 'HERE.parents' not in expression:
                        continue
                    # prepare_model.py intentionally uses its own method directory.
                    if source.name == 'prepare_model.py':
                        continue
                    namespace = {'Path': Path, '__file__': str(source), 'HERE': source.parent}
                    actual = eval(compile(ast.Expression(node.value), str(source), 'eval'), namespace)
                    self.assertEqual(actual, ROOT, str(source.relative_to(ROOT)))
                    checked += 1
        self.assertGreaterEqual(checked, 20)

    def test_repository_imports_resolve(self):
        for source in ROOT.rglob('*.py'):
            if '.git' in source.parts:
                continue
            for node in ast.walk(ast.parse(source.read_text(encoding='utf-8'))):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                self.assertNotEqual(node.module.split('.')[0], 'scripts', str(source))
                if node.module.split('.')[0] in {'methods', 'analysis'}:
                    target = ROOT.joinpath(*node.module.split('.')).with_suffix('.py')
                    if not target.is_file():
                        package = target.with_suffix('')
                        self.assertTrue(package.is_dir(), f'{source}: {node.module}')
                        for imported in node.names:
                            self.assertTrue((package / (imported.name + '.py')).is_file(), f'{source}: {node.module}.{imported.name}')

    def test_configured_runner_commands_and_test_gate(self):
        runner = ROOT / 'methods/experiments/run_methods_condition.py'
        matrix = json.loads((ROOT / 'configs/experiment_matrix.json').read_text(encoding='utf-8'))
        for condition in matrix['conditions']:
            stages = ['final-eval'] if condition['method'] in {'0-shot', '10-shot'} else ['train', 'final-eval']
            for stage in stages:
                with self.subTest(condition=condition['condition_id'], stage=stage):
                    command = [sys.executable, '-B', str(runner), condition['condition_id'], '--stage', stage, '--dry-run']
                    if stage == 'final-eval':
                        command.append('--confirm-locked-test')
                    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
                    self.assertIn(str(ROOT / 'methods/sft/train.py'), result.stdout)
                    self.assertIn(condition['model_id'], result.stdout)
        result = subprocess.run([sys.executable, '-B', str(runner), 'qwen_sft_6000', '--stage', 'final-eval', '--dry-run'], cwd=ROOT, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('confirm-locked-test', result.stderr)

    def test_cue_adapter_default_and_override(self):
        source = ROOT / 'analysis/robustness/cue_masking/run_cue_masking_analysis.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'ROOT', 'ADAPTER_ROOT', 'MODELS'} for t in n.targets)]
        code = compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec')
        for override in (None, str(ROOT / 'alternate_adapters')):
            with self.subTest(override=override), patch.dict(os.environ):
                os.environ.pop('NEISS_ADAPTER_ROOT', None)
                if override is not None:
                    os.environ['NEISS_ADAPTER_ROOT'] = override
                namespace = {'Path': Path, 'os': os, '__file__': str(source)}
                exec(code, namespace)
                expected = Path(override) if override else ROOT / 'methods_runs'
                for condition in ('qwen_sft_6000', 'mistral_sft_3000'):
                    self.assertEqual(namespace['MODELS'][condition]['adapter'], expected / condition)

    def test_generated_notebooks_and_root_discovery(self):
        builder = runpy.run_path(str(ROOT / 'analysis/publication_figures/build_figure_notebooks.py'))
        for filename, title, function, output, description in builder['FIGURES']:
            notebook = json.loads((ROOT / 'notebooks/publication_figures' / filename).read_text(encoding='utf-8'))
            self.assertEqual(notebook, builder['notebook'](title, function, output, description))
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                self.assertFalse(cell['outputs'])
                source = ''.join(cell['source'])
                tree = ast.parse(source)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == 'find_project_root':
                        namespace = {'Path': Path}
                        exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
                        for cwd in (ROOT, ROOT / 'notebooks/publication_figures'):
                            with patch.object(Path, 'cwd', return_value=cwd):
                                self.assertEqual(namespace['find_project_root'](), ROOT)
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'OUTPUT' for t in node.targets):
                        actual = eval(compile(ast.Expression(node.value), filename, 'eval'), {'ROOT': ROOT})
                        self.assertEqual(actual, ROOT / 'figures' / output)

    def test_gpt_orchestrator_sibling_commands(self):
        source = ROOT / 'methods/gpt54/orchestrate_gpt54_batches.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        targets = []
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.left, ast.Name) and node.left.id == 'SCRIPT_DIR' and isinstance(node.right, ast.Constant):
                target = source.parent / node.right.value
                self.assertTrue(target.is_file(), str(target))
                targets.append(target.name)
        self.assertEqual(set(targets), {'run_gpt54_evaluation.py', 'run_gpt54_10shot_chunks.py'})


if __name__ == '__main__':
    unittest.main()
