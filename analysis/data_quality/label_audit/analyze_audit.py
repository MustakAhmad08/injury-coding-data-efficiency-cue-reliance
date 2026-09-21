"""Compute descriptive agreement from a local audit CSV without exporting cases."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    args=parser.parse_args()
    raw=args.csv.read_bytes();rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    counts={(a,b):0 for a in ['yes','no'] for b in ['yes','no']};bad=0
    for row in rows:
        a=row['review1_code_matches_narrative'].strip().lower()
        b=row['review2_code_matches_narrative'].strip().lower()
        if (a,b) not in counts:raise ValueError('Incomplete/invalid paired judgments; do not silently drop records.')
        counts[a,b]+=1
        bad+=row['intercoder_Agreement'].strip().lower()!=('yes' if a==b else 'no')
    n=len(rows)
    if not n:raise ValueError('Empty audit')
    observed=(counts['yes','yes']+counts['no','no'])/n
    expected=sum(sum(counts[a,b] for b in ['yes','no'])*sum(counts[b,a] for b in ['yes','no'])/n**2 for a in ['yes','no'])
    print(json.dumps({'source_sha256':hashlib.sha256(raw).hexdigest(),'n':n,'counts':{f'{a}_{b}':v for (a,b),v in counts.items()},'observed_agreement':observed,'expected_agreement':expected,'cohens_kappa':(observed-expected)/(1-expected) if expected<1 else None,'stored_agreement_column_mismatches':bad},indent=2))

if __name__=='__main__':main()
