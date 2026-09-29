"""All options for GANN reproduction.

Defaults follow the paper (docs/01_模型与Loss详解.md Sec.7); values marked
"objRel/ruotianluo" come from the base implementation conventions.
"""
import argparse
import json
import sys
import time


def parse_opt():
    parser = argparse.ArgumentParser()

    # ---------------- data ----------------
    parser.add_argument('--karpathy_json', default='data/captions/dataset_coco.json',
                        help='Karpathy dataset_coco.json (train/val/test splits)')
    parser.add_argument('--input_json', default='data/processed/vocab.json',
                        help='generated: vocabulary + split index (itow/wtoi/ix_to_id)')
    parser.add_argument('--feature_files', default='',
                        help='comma list of .npy feature banks (banks mode)')
    parser.add_argument('--feature_mode', default='files',
                        choices=['files', 'banks', 'packed'],
                        help="files = cocobu_att per-image npz (default); "
                             "packed = single memmap binary via "
                             "pack_features.py (fastest, A800 recommended); "
                             "banks = mmap'd npy matrix")
    parser.add_argument('--feature_index', default='data/processed/feat_index.json',
                        help='cocoid -> (bank_id, row) alignment json')
    parser.add_argument('--labels_json', default='data/processed/labels.json',
                        help='generated: 220 caption-object labels')
    parser.add_argument('--labels_vec', default='data/processed/labels_{split}.npz',
                        help='per-image 0/1 caption-object vectors')
    parser.add_argument('--input_label_h5', default='none',
                        help='unused (kept for objRel command compatibility)')
    parser.add_argument('--input_fc_dir', default='none')
    parser.add_argument('--input_att_dir', default='none')
    parser.add_argument('--input_box_dir', default='none')
    parser.add_argument('--input_rel_box_dir', default='none')

    # ---------------- model ----------------
    parser.add_argument('--arch', default='paper',
                        choices=['paper', 'tiny', 'small', 'medium', 'wide',
                                 'deep', 'large', 'custom'],
                        help='model scale preset (see ARCH_PRESETS); '
                             'explicit CLI flags override the preset')
    parser.add_argument('--feat_dim', type=int, default=2048)
    parser.add_argument('--input_encoding_size', type=int, default=512,
                        help='d_model')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='default depth for BOTH encoder & decoder '
                             '(paper: 6)')
    parser.add_argument('--num_encoder_layers', type=int, default=-1,
                        help='-1 = use --num_layers')
    parser.add_argument('--num_decoder_layers', type=int, default=-1,
                        help='-1 = use --num_layers')
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--init_scheme', default='xavier',
                        choices=['xavier', 'bert', 'trunc_normal'],
                        help='xavier = objRel/original Transformer '
                             '(xavier_uniform, all p.dim()>1); '
                             'bert = N(0,0.02) + LN(1,0); '
                             'trunc_normal = truncnormal(0,0.02)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_seq_len', type=int, default=17,
                        help='caption length cap (incl. <eos>)')
    parser.add_argument('--no_gann', type=int, default=0,
                        help='1 = disable GlobalAttention (base Transformer)')
    parser.add_argument('--weight_tie', type=int, default=0,
                        help='1 = tie output projection with word embedding')
    parser.add_argument('--gann_direct', type=int, default=0,
                        help='1 = literal Eq.12; 0 = log-space equivalent')
    parser.add_argument('--gann_share_heads', type=int, default=0,
                        help='1 = share global projections across heads')

    # caption object loss
    parser.add_argument('--label_loss_weight', type=float, default=0.2,
                        help='lambda in Eq.18/19 (paper: 0.2)')

    # ---------------- optimization ----------------
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--learning_rate_decay_start', type=int, default=0)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.95)
    parser.add_argument('--learning_rate_decay_every', type=int, default=3,
                        help='anneal x0.95 every 3 epochs (paper IV-B)')
    parser.add_argument('--noamopt', type=int, default=1,
                        help='1 = objRel-style noam warmup (paper: "same warm up as [23]")')
    parser.add_argument('--noamopt_warmup', type=int, default=10000)
    parser.add_argument('--noamopt_base_lr', type=float, default=2e-4)
    parser.add_argument('--optim_alpha', type=float, default=0.9)
    parser.add_argument('--optim_beta', type=float, default=0.999)
    parser.add_argument('--optim_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='0 = off; 0.999 = EMA weights used for val/save')
    parser.add_argument('--scst_decay', type=int, default=0,
                        help='1 = apply step decay in SCST phase (default: '
                             'constant scst_learning_rate)')

    # scheduled sampling (paper IV-B; default off for Transformer, see docs)
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1)
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5)
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05)
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25)

    # ---------------- phases / schedule ----------------
    parser.add_argument('--phase', default='all',
                        choices=['enc_pre', 'xe', 'scst', 'all'],
                        help="enc_pre = encoder+labelhead only (L_label); "
                             "paper: 10 epochs enc_pre then xe, then scst")
    parser.add_argument('--enc_pre_epochs', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--scst_max_epochs', type=int, default=30)
    parser.add_argument('--scst_learning_rate', type=float, default=5e-5,
                        help='ruotianluo convention; paper silent')
    parser.add_argument('--self_critical_after', type=int, default=-1,
                        help='legacy knob (unused with phase-based flow)')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='epochs without val improvement before stopping')
    parser.add_argument('--val_every_steps', type=int, default=0,
                        help='>0: run a mid-epoch val every N steps (fast '
                             'early-stop signal; 0 = epoch-level only)')
    parser.add_argument('--val_steps_subset', type=int, default=1000,
                        help='images for mid-epoch val checks')
    parser.add_argument('--collapse_drop', type=float, default=12.0,
                        help='abort run if val CIDEr falls this far below '
                             'best on two consecutive checks')
    parser.add_argument('--train_sample_n', type=int, default=1,
                        help='SCST samples per image (1 = paper Eq.17)')

    # ---------------- eval ----------------
    parser.add_argument('--val_images_use', type=int, default=5000)
    parser.add_argument('--max_train_images', type=int, default=-1,
                        help='cap training set size (E0 overfit sanity)')
    parser.add_argument('--language_eval', type=int, default=1)
    parser.add_argument('--sample_method', default='greedy',
                        choices=['greedy', 'beam'])
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--length_penalty', type=float, default=0.0,
                        help='beam score = sum_logp / len^alpha; '
                             '0 = off; scan {0.5,1.0,1.5} at final eval')
    parser.add_argument('--eval_splits', default='test')

    # ---------------- bookkeeping ----------------
    parser.add_argument('--id', default='gann_base')
    parser.add_argument('--checkpoint_path', default='',
                        help='default: log_<id>')
    parser.add_argument('--start_from', default='', help='resume dir')
    parser.add_argument('--start_from_model', default='',
                        help='load weights only (phase hand-off)')
    parser.add_argument('--save_checkpoint_every', type=int, default=3000,
                        help='iterations')
    parser.add_argument('--save_history_ckpt', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--amp', type=int, default=0,
                        help='1 = bf16 autocast (~2x on A800/H200, no scaler '
                             'needed); keep 0 on CPU')
    parser.add_argument('--cider_cache', default='data/processed/coco-train-idxs',
                        help='precomputed train n-gram pkl for SCST reward')
    parser.add_argument('--tensorboard', type=int, default=0)

    args = parser.parse_args()
    if not args.checkpoint_path:
        args.checkpoint_path = 'log_' + args.id
    args.itow = None
    apply_arch_preset(args)
    return args


# d_k is kept at 64 throughout (paper convention); heads = d_model / 64.
# Preset names mirror the BERT family (tiny/small/medium/...). Depth is given
# as (encoder, decoder) so asymmetric variants are one flag away.
ARCH_PRESETS = {
    #                d_model  n_head  d_ff   enc  dec  notes
    'tiny':   dict(input_encoding_size=256,  n_head=4,  d_ff=1024,
                   num_encoder_layers=3,  num_decoder_layers=3),
    'small':  dict(input_encoding_size=512,  n_head=8,  d_ff=1536,
                   num_encoder_layers=3,  num_decoder_layers=3),
    'paper':  dict(input_encoding_size=512,  n_head=8,  d_ff=2048,
                   num_encoder_layers=6,  num_decoder_layers=6),
    'medium': dict(input_encoding_size=512,  n_head=8,  d_ff=2048,
                   num_encoder_layers=8,  num_decoder_layers=8),   # BERT-Medium depth
    'wide':   dict(input_encoding_size=768,  n_head=12, d_ff=3072,
                   num_encoder_layers=6,  num_decoder_layers=6),   # BERT-Base width
    'deep':   dict(input_encoding_size=512,  n_head=8,  d_ff=2048,
                   num_encoder_layers=12, num_decoder_layers=12),
    'large':  dict(input_encoding_size=768,  n_head=12, d_ff=3072,
                   num_encoder_layers=12, num_decoder_layers=12),  # BERT-Base width x BERT-Large depth
}


def apply_arch_preset(args):
    """Apply --arch preset unless the user explicitly passed the flag.
    Keeps num_layers (shared default) consistent with per-side values."""
    if args.arch == 'custom':
        if args.num_encoder_layers < 0:
            args.num_encoder_layers = args.num_layers
        if args.num_decoder_layers < 0:
            args.num_decoder_layers = args.num_layers
        return
    preset = ARCH_PRESETS[args.arch]
    explicit = set()
    for a in sys.argv:
        if a.startswith('--'):
            explicit.add(a.split('=')[0][2:])
    for k, v in preset.items():
        if k not in explicit:
            setattr(args, k, v)
    if 'num_encoder_layers' not in explicit and \
            'num_layers' not in explicit and args.num_layers != 6:
        pass  # preset already set both sides
    # keep shared num_layers coherent for opts.json readability
    if args.num_encoder_layers == args.num_decoder_layers:
        args.num_layers = args.num_encoder_layers


def save_opt(opt, path):
    d = {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None)))
             else str(v)) for k, v in vars(opt).items()}
    d['_saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
