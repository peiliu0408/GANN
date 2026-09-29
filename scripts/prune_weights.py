"""Storage discipline for ablation sweeps.

Deletes heavy weight files from log_<id> dirs while keeping the run's
configuration, training history and predictions (the actual ablation record).

  python scripts/prune_weights.py --all           # keep model-best only
  python scripts/prune_weights.py --all --none    # delete ALL weights
  python scripts/prune_weights.py --keep E2_gann_paper E7b_batch128 --all
  python scripts/prune_weights.py --dry-run --all
"""
import argparse
import glob
import os

WEIGHT_FILES = ('model.pth', 'model-best.pth')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', help='all log_* dirs')
    ap.add_argument('--only', nargs='*', default=[], help='specific log ids')
    ap.add_argument('--keep', nargs='*', default=[],
                    help='ids whose weights are kept untouched')
    ap.add_argument('--none', action='store_true',
                    help='delete model-best too (after final reporting)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    ids = args.only
    if args.all:
        ids += [os.path.basename(p) for p in glob.glob('log_*')
                if os.path.isdir(p)]
    ids = [i for i in dict.fromkeys(ids) if i not in args.keep]
    if not ids:
        print('nothing to prune (use --all or --only <id>...)')
        return

    freed = 0
    for i in ids:
        d = i if os.path.isdir(i) else f'log_{i}'
        if not os.path.isdir(d):
            print(f'[skip] {d} not found')
            continue
        for w in WEIGHT_FILES:
            p = os.path.join(d, w)
            if not os.path.exists(p):
                continue
            if w == 'model-best.pth' and not args.none:
                continue                     # best stays by default
            size = os.path.getsize(p)
            freed += size
            if args.dry_run:
                print(f'[dry] delete {p} ({size/1e6:.0f} MB)')
            else:
                os.remove(p)
                print(f'deleted {p} ({size/1e6:.0f} MB)')
    print(f'total freed: {freed/1e9:.2f} GB'
          + (' (dry run)' if args.dry_run else ''))


if __name__ == '__main__':
    main()
