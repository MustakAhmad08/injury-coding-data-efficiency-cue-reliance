"""Regenerate the four-model cue figure from included aggregates only."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.publication_figures.generate_publication_figures import configure_style, masking_effects

if __name__ == '__main__':
    configure_style()
    masking_effects()
