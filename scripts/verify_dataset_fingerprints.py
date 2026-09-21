"""Compare local source workbooks or reconstructed splits to recorded SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path)
    parser.add_argument('--data-dir',type=Path)
    args=parser.parse_args()
    if args.source_dir is None and args.data_dir is None:parser.error('Supply --source-dir and/or --data-dir')
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'data/reference_dataset_manifest.json').read_text())
    failures=[];checked=0
    for directory,key in [(args.source_dir,'source_sha256'),(args.data_dir,'output_sha256')]:
        if directory is None:continue
        for name,expected in manifest[key].items():
            path=directory/name
            if not path.is_file():failures.append(f'Missing: {name}')
            elif digest(path)!=expected:failures.append(f'Hash mismatch: {name}')
            checked+=1
    if failures:raise SystemExit('\n'.join(failures)+'\nDo not claim exact reconstruction; investigate source revisions and serialization differences.')
    print(f'Passed: {checked} file fingerprints match the recorded benchmark.')

if __name__=='__main__':main()
