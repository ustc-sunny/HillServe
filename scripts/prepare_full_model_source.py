#!/usr/bin/env python3
"""Apply only a verified prefix of our patch series to the existing b6000 tree.

No fetch, checkout, reset, stash, source replacement or upstream upgrade.
Expected files are reconstructed from git objects in a temporary directory;
unknown tracked edits are rejected before any mutation.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PIN = '4762ad7316dcdec20016ab5985fb46a27902204d'
PATCHES = [ROOT / 'patches' / p for p in (
    'llama-b6000-round2.patch',
    'llama-b6000-early-rehearsal.patch',
    'llama-b6000-external-attention-device.patch',
    'llama-b6000-native-prefill-transition.patch',
    'llama-b6000-memory-baselines.patch',
    'llama-b6000-h2o-external-decode.patch')]


def run(argv, cwd, binary=False):
    return subprocess.check_output(argv, cwd=cwd, text=not binary)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT / 'third_party/llama-round2')
    ap.add_argument('--check-only', action='store_true')
    x = ap.parse_args()
    source = x.source.resolve()
    assert run(['git', 'rev-parse', 'HEAD'], source).strip() == PIN, 'fixed b6000 required'
    files = sorted({line[6:] for p in PATCHES for line in p.read_text().splitlines()
                    if line.startswith('+++ b/')})
    changed = set(run(['git', 'diff', 'HEAD', '--name-only'], source).splitlines())
    assert changed <= set(files), 'unknown tracked source changes outside patch series'
    actual = {f: (source / f).read_bytes() for f in files}
    states = []
    with tempfile.TemporaryDirectory(prefix='hillserve-patch-check-') as temporary:
        temp = Path(temporary)
        for f in files:
            dest = temp / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(run(['git', 'show', PIN + ':' + f], source, binary=True))
        states.append({f: (temp / f).read_bytes() for f in files})
        for patch in PATCHES:
            subprocess.run(['git', 'apply', '--check', str(patch)], cwd=temp, check=True)
            subprocess.run(['git', 'apply', str(patch)], cwd=temp, check=True)
            states.append({f: (temp / f).read_bytes() for f in files})
    matches = [i for i, state in enumerate(states) if state == actual]
    assert len(matches) == 1, 'existing source is not an exact known patch prefix; preserve and inspect'
    prefix = matches[0]
    if x.check_only:
        assert prefix == len(PATCHES), 'full model patch series not yet applied'
    else:
        for patch in PATCHES[prefix:]:
            subprocess.run(['git', 'apply', '--check', str(patch)], cwd=source, check=True)
            subprocess.run(['git', 'apply', str(patch)], cwd=source, check=True)
        assert {f: (source / f).read_bytes() for f in files} == states[-1]
    print(f'passed: pinned {PIN}; full model patch series; {len(files)} verified files')


if __name__ == '__main__':
    main()
