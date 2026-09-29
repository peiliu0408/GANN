"""Rescore an existing predictions JSON without loading the model.

  python scripts/rescore.py ckpt/preds_test_xe.json
  python scripts/rescore.py preds.json --out score.json # also save metrics json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

from gann.language_eval import evaluate_predictions          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pred_json', help='predictions json written by '
                    'save_predictions() (contains gts/preds keys)')
    ap.add_argument('--out', default='',
                    help='optional path to write metrics json')
    args = ap.parse_args()

    with open(args.pred_json, encoding='utf-8') as f:
        data = json.load(f)
    gts = data['gts']
    preds = data['preds']
    missing = set(gts) - set(preds)
    if missing:
        raise SystemExit(f'ERROR: {len(missing)} ids in gts missing from '
                         f'preds, e.g. {sorted(missing)[:5]}')
    if set(gts) != set(preds):
        preds = {k: preds[k] for k in gts}

    metrics = evaluate_predictions(gts, preds)

    old = data.get('info', {}).get('metrics', {})
    print(json.dumps(metrics, indent=2))
    if old:
        print('\n--- previous info.metrics ---')
        print(json.dumps(old, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'source': args.pred_json, 'metrics': metrics}, f,
                      indent=2)
        print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
