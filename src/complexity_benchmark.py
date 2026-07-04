import argparse
import os
import sys
import time
from pathlib import Path

import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lightning_model import ConvNeXtV2Classifier


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, device):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        model(x)
    return fc.get_total_flops()


@torch.no_grad()
def measure_latency(model, device, batch_size, iters, warmup=10):
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    is_cuda = device.type == 'cuda'
    for _ in range(warmup):
        model(x)
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    per_batch_ms = elapsed / iters * 1e3
    per_image_ms = per_batch_ms / batch_size
    return per_batch_ms, per_image_ms


def bench(name, model, device, iters):
    model = model.to(device)
    total, trainable = count_params(model)
    try:
        flops = count_flops(model, device)
        gflops = flops / 1e9
    except Exception as e:
        gflops = float('nan')
        print(f'  [warn] FLOP count gagal untuk {name}: {e}')
    b1_batch, b1_img = measure_latency(model, device, 1, iters)
    b32_batch, b32_img = measure_latency(model, device, 32, max(5, iters // 4))
    return {
        'name': name,
        'params_m': total / 1e6,
        'trainable_m': trainable / 1e6,
        'gflops': gflops,
        'gmacs': gflops / 2,
        'lat_b1_img_ms': b1_img,
        'lat_b32_img_ms': b32_img,
        'throughput_b32': 1e3 / b32_img,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--out', default='results/complexity.md')
    args = ap.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    dev_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    print(f'Device: {device} ({dev_name})\n')

    models = [
        ('ConvNeXtV2-Tiny (no MHA)',
         ConvNeXtV2Classifier(num_classes=2, pretrained=False, use_mha=False)),
    ]

    rows = []
    for name, m in models:
        print(f'Benchmark: {name} ...')
        rows.append(bench(name, m, device, args.iters))
        del m
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    L = ['# Kompleksitas Model & Latensi', '',
         f'Input 224x224, FP32, device **{dev_name}**. Latensi = warmup 10 + rata-rata '
         f'{args.iters} iterasi (batch=1) / {max(5, args.iters // 4)} iterasi (batch=32), CUDA-synced. '
         'Bobot acak (tidak memengaruhi params/FLOPs/latensi).', '',
         '| Model | Params (M) | FLOPs (G) | GMACs | Latensi b1 (ms/img) | Latensi b32 (ms/img) | Throughput b32 (img/s) |',
         '| ----- | :--------: | :-------: | :---: | :-----------------: | :------------------: | :--------------------: |']
    for r in rows:
        L.append(f"| {r['name']} | {r['params_m']:.1f} | {r['gflops']:.2f} | {r['gmacs']:.2f} "
                 f"| {r['lat_b1_img_ms']:.2f} | {r['lat_b32_img_ms']:.2f} | {r['throughput_b32']:.0f} |")
    L += ['', '> FLOPs via `torch.utils.flop_counter.FlopCounterMode` (total_flops = MACs). '
          'GMACs = FLOPs/2. Latensi spesifik-hardware; pakai untuk perbandingan relatif antar-model '
          'pada perangkat sama, bukan angka absolut universal.']

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L), encoding='utf-8')

    print('\n' + '\n'.join(L[4:8]))
    print(f'\nSaved -> {out}')


if __name__ == '__main__':
    main()
