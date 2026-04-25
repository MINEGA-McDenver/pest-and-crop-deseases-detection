import csv

in_path = 'analysis_outputs/field_audit_rows.tsv'
out_path = 'analysis_outputs/banana_maize_failures.csv'

with open(in_path, encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    rows = [r for r in reader if r.get('expected_crop') in ('banana', 'maize') and r.get('resultType') in ('unsupported', 'other_leaf', 'uncertain')]

fields = ['expected_crop', 'file', 'resultType', 'gate', 'bestCrop', 'bestClass', 'bestClassProb', 'bestCropTotal', 'otherLeaf', 'top1', 'top1_p', 'top2', 'top2_p', 'top3', 'top3_p', 'entropy']

with open(out_path, 'w', encoding='utf-8', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, '') for k in fields})

print('WROTE', out_path, 'ROWS', len(rows))
