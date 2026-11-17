"""Full-match frozen-model inference. Never reads a possession-label file."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model import PossessionModel
from smoothing import smooth_short_runs

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = [(11185, 90652), (110808, 190198)]


def infer(path, arrays, batch_size=32):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = PossessionModel(tokens=checkpoint['tokens'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    p, m, t, s = [torch.from_numpy(x.copy()) for x in arrays]
    s[:, 6] = 0
    for first, end in BLOCKS:
        age = 0.
        for i in range(int(np.ceil(first / 5)), int(np.ceil(end / 5))):
            age = 0. if s[i, 4] > 0 else min(1., age + .02)
            s[i, 7] = age
    radius = checkpoint['tokens'] // 2
    offsets = np.arange(-radius, radius + 1) * 5
    prediction = ['X'] * 199319
    with torch.inference_mode():
        for first, end in BLOCKS:
            centres = np.rint(np.arange(first / 25, end / 25, .25) * 25).astype(np.int64)
            centres = centres[centres < end]
            values = []
            for start in range(0, len(centres), batch_size):
                ix = np.rint((centres[start:start + batch_size, None] + offsets) / 5).astype(np.int64)
                ix = np.clip(ix, int(np.ceil(first / 5)), (end - 1) // 5)
                pp, mm, tt, ss = p[ix].clone(), m[ix], t[ix], s[ix].clone()
                active = mm.any(dim=(0, 1))
                if not active.any():
                    active[0] = True
                pp, mm, tt = pp[:, :, active], mm[:, :, active], tt[:, :, active]
                edge = torch.from_numpy(ix == int(np.ceil(first / 5)))
                pp[:, :, :, 2:4].masked_fill_(edge[:, :, None, None], 0)
                pp[:, :, :, 9].masked_fill_(edge[:, :, None], 0)
                ss[:, :, 2:4].masked_fill_(edge[:, :, None], 0)
                ss[:, :, 5].masked_fill_(edge, 0)
                values.extend(model(pp, mm, tt, ss).argmax(-1).tolist())
            edges = (centres[:-1] + centres[1:]) // 2
            expanded = np.asarray(values)[np.searchsorted(edges, np.arange(first, end), side='left')]
            prediction[first:end] = ['HAD'[i] for i in expanded]
            print(path.name, 'half complete', first, end, flush=True)
    return prediction


def write(path, labels):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['frame', 'label'])
        writer.writerows(enumerate(labels))


def main(output):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with np.load(ROOT / 'outputs/features.npz') as f:
        arrays = [f[k] for k in ('player', 'mask', 'team', 'scene')]
    results = []
    for name in ['5s', '10s']:
        raw = infer(ROOT / 'models' / f'pooled_{name}.pt', arrays)
        smooth, _ = smooth_short_runs(raw, BLOCKS)
        write(ROOT / 'outputs' / f'{name}_raw.csv', raw)
        write(ROOT / 'outputs' / f'{name}_smooth2s.csv', smooth)
        results.append(smooth)
    labels = ['D' if a == 'D' or b == 'D' else a for a, b in zip(*results)]
    assert all(labels[f] in 'HAD' for a, b in BLOCKS for f in range(a, b))
    write(output, labels)
    report = dict(frames=len(labels), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  rule='Smooth each model independently at 2s, then D if either D; otherwise keep 5s. No smoothing after merge.',
                  models=json.loads((ROOT / 'models/manifest.json').read_text()),
                  public_possession_labels_read=False, pseudo_labels_read=False,
                  counts={x: labels.count(x) for x in 'HADX'},
                  torch_version=torch.__version__, numpy_version=np.__version__)
    (ROOT / 'outputs/submission_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/submission.csv')
    main(parser.parse_args().output.resolve())
