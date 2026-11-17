"""Regenerate the submission from licensed tracking and frozen checkpoints."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=ROOT / 'outputs/submission.csv')
args = parser.parse_args()
(ROOT / 'outputs').mkdir(exist_ok=True)
for script in ['sync_recordings.py', 'finish_sync.py', 'features.py']:
    subprocess.run([sys.executable, str(ROOT / 'src' / script)], cwd=ROOT, check=True)
subprocess.run([sys.executable, str(ROOT / 'src/predict.py'), '--output', str(args.output.resolve())], cwd=ROOT, check=True)
subprocess.run([sys.executable, str(ROOT / 'evaluation/validate_submission.py'), str(args.output.resolve())], cwd=ROOT, check=True)
