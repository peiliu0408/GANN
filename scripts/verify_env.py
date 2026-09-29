"""Step-0 gate for the intranet machine: verify the training environment
BEFORE launching any experiment.

  python scripts/verify_env.py

Checks: torch/CUDA/GPU count, bf16 support (H200), tiny fwd/bwd on GPU,
throughput estimate, data files present, then prints PASS/FAIL + advice.
"""
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import types

import torch                                                    # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ok = True


def check(name, fn):
    global ok
    try:
        msg = fn()
        print(f'[PASS] {name}: {msg}')
    except Exception as e:                                    # noqa
        ok = False
        print(f'[FAIL] {name}: {e}')


def c_torch():
    import torch as t
    return f'version {t.__version__}'


def c_cuda():
    import torch as t
    assert t.cuda.is_available(), 'CUDA not available'
    n = t.cuda.device_count()
    names = [t.cuda.get_device_name(i) for i in range(n)]
    return f'{n} GPU(s): {names[0]}{"..." if n > 1 else ""}'


def c_bf16():
    import torch as t
    assert t.cuda.is_available()
    x = torch.randn(8, 8, device='cuda', dtype=torch.bfloat16)
    y = x @ x
    assert torch.isfinite(y).all()
    return 'bf16 matmul ok'


def c_trainstep():
    import torch as t
    from gann.models.gann import GANNCaptioner
    opt = types.SimpleNamespace(
        input_encoding_size=512, n_head=8, d_ff=2048, dropout=0.1,
        num_layers=6, feat_dim=2048, vocab_size=9160, n_labels=207,
        max_seq_len=17, label_loss_weight=0.2, label_smoothing=0.0,
        no_gann=0, gann_direct=0, gann_share_heads=0, itow={},
        init_scheme='xavier', num_encoder_layers=6, num_decoder_layers=6)
    m = GANNCaptioner(opt).cuda()
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    feats = torch.randn(16, 36, 2048, device='cuda')
    labels = torch.randint(4, 9160, (16, 15), device='cuda')
    labels[:, 0] = 1
    labels[torch.arange(16), 14] = 2
    lv = (torch.rand(16, 207, device='cuda') > 0.7).float()
    opti = torch.optim.Adam(m.parameters(), lr=1e-4)
    # warmup
    for _ in range(3):
        out = m(feats, labels, lv, keys_valid=torch.full((16,), 36,
                                                         dtype=torch.long,
                                                         device='cuda'),
                mode='xe')
        opti.zero_grad()
        out['loss'].backward()
        opti.step()
    torch.cuda.synchronize()
    t0 = time.time()
    iters = 10
    for _ in range(iters):
        out = m(feats, labels, lv, keys_valid=torch.full((16,), 36,
                                                         dtype=torch.long,
                                                         device='cuda'),
                mode='xe')
        opti.zero_grad()
        out['loss'].backward()
        opti.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1e9
    return (f'fwd+bwd OK, {n_params/1e6:.1f}M params, '
            f'{iters/dt:.1f} it/s @bs16, peak mem {mem:.1f} GB')


def c_data():
    need = ['data/captions/dataset_coco.json',
            'data/processed/vocab.json',
            'data/processed/feat_index.json',
            'data/processed/labels.json',
            'data/processed/coco-train-idxs.pkl']
    miss = [p for p in need if not os.path.exists(os.path.join(BASE, p))]
    assert not miss, f'missing {miss}'
    idx = os.path.join(BASE, 'data', 'processed', 'feat_index.json')
    import json
    n = len(json.load(open(idx, encoding='utf-8')))
    assert n == 123287, f'feat_index has {n} entries (expect 123287)'
    return f'all {len(need)} files present, feat_index {n} entries'


def c_cocoeval():
    base = os.path.join(BASE, 'third_party')
    assert os.path.isdir(os.path.join(base, 'pycocoevalcap')), 'no vendored eval'
    java = shutil.which('java') is not None
    return 'pycocoevalcap present' + (' (+Java: METEOR/SPICE enabled)'
                                      if java else
                                      ' (no Java: METEOR/SPICE will be skipped)')


def c_disk():
    free = shutil.disk_usage(BASE).free / 1e9
    assert free > 60, f'only {free:.0f} GB free (need ~60+ for ckpts)'
    return f'{free:.0f} GB free'


if __name__ == '__main__':
    print('=== GANN intranet environment check ===')
    check('torch', c_torch)
    check('cuda', c_cuda)
    check('bf16', c_bf16)
    check('train step (GPU)', c_trainstep)
    check('data', c_data)
    check('eval libs', c_cocoeval)
    check('disk', c_disk)
    print('=== ' + ('ALL PASS - 可以开训' if ok else '有 FAIL 项，先解决再开训') +
          ' ===')
    sys.exit(0 if ok else 1)
