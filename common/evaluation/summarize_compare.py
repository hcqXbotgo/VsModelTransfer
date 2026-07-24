#!/usr/bin/env python3
"""Sort StatlasQuant layer comparison results and write a compact report."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit('compare result not found: {}'.format(input_path))

    with input_path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or 'cos_sim' not in rows[0]:
        raise SystemExit('invalid compare result: {}'.format(input_path))

    normalized = []
    for row in rows:
        try:
            cosine = float(row['cos_sim'])
        except (TypeError, ValueError):
            continue
        normalized.append({
            'Layername': row.get('Layername', ''),
            'node': row.get('node', ''),
            'cos_sim': cosine,
            'cosine_error': max(0.0, 1.0 - cosine),
        })
    normalized.sort(key=lambda item: item['cosine_error'], reverse=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(
            stream, fieldnames=('Layername', 'node', 'cos_sim', 'cosine_error'))
        writer.writeheader()
        writer.writerows(normalized)

    report_path = Path(args.report)
    shown = normalized[:args.top]
    lines = [
        '# Layer comparison report',
        '',
        '- Layers compared: {}'.format(len(normalized)),
        '- Metric: `cosine_error = 1 - cosine_similarity`',
        '- Larger error means a larger float/quant activation difference.',
        '',
        '| Rank | Layer | Node | Cosine similarity | Cosine error |',
        '|---:|---|---|---:|---:|',
    ]
    for index, row in enumerate(shown, 1):
        layer = str(row['Layername']).replace('|', '\\|')
        node = str(row['node']).replace('|', '\\|')
        lines.append('| {} | `{}` | `{}` | {:.8f} | {:.8f} |'.format(
            index, layer, node, row['cos_sim'], row['cosine_error']))
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    if normalized:
        worst = normalized[0]
        print('compared {} layers'.format(len(normalized)))
        print('worst layer: {} cosine={:.8f} error={:.8f}'.format(
            worst['Layername'], worst['cos_sim'], worst['cosine_error']))
    print('sorted CSV:', output_path)
    print('report:', report_path)


if __name__ == '__main__':
    main()
