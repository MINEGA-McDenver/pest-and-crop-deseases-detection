import csv

inpath = 'analysis_outputs/field_audit_rows.tsv'
outpath = 'analysis_outputs/banana_maize_failures.csv'

with open(inpath, 'r', encoding='utf-8') as fr:
    header_line = fr.readline().rstrip('\n')
    header = header_line.split('\t')
    with open(outpath, 'w', newline='', encoding='utf-8') as fo:
        writer = csv.writer(fo)
        writer.writerow(header)
        count = 0
        for line in fr:
            if line.startswith('banana\t') or line.startswith('maize\t'):
                writer.writerow(line.rstrip('\n').split('\t'))
                count += 1

print('WROTE', outpath, count)
