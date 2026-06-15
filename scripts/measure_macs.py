"""
measure_macs.py — измерение MACs и параметров для ConvTasNet с разным
количеством DSC-блоков с использованием thop и fvcore.

Запуск:
    python measure_macs.py
"""

import csv
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Импорт модели из проекта
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.models import ConvTasNet  # noqa: E402

# ---------------------------------------------------------------------------
# Конфигурации
# ---------------------------------------------------------------------------
CONFIGS = {
    "L=0": dict(dsc_layers=0),
    "L=5": dict(dsc_layers=5),
    "L=10": dict(dsc_layers=10),
    "L=20": dict(dsc_layers=20),
    "L=24": dict(dsc_layers=24),
}

# Общие гиперпараметры (из configs/baseline.yaml)
BASE_KWARGS = dict(
    num_sources=4,
    audio_channels=1,
    encoder_channels=512,
    bottleneck_channels=128,
    hidden_channels=512,
    skip_channels=128,
    num_blocks=8,
    num_repeats=3,
    kernel_size=3,
    encoder_kernel_size=16,
    encoder_stride=8,
    causal=False,
)

# ---------------------------------------------------------------------------
# Входные данные (строго одинаковые для всех конфигураций)
# ---------------------------------------------------------------------------
torch.manual_seed(0)
DUMMY_INPUT = torch.randn(1, 1, 44100)

# ---------------------------------------------------------------------------
# Измерения
# ---------------------------------------------------------------------------
results = []

for name, extra_kwargs in CONFIGS.items():
    print(f"\n--- Измерение {name} ---")

    # 1. Инициализация модели (без загрузки весов)
    model = ConvTasNet(**BASE_KWARGS, **extra_kwargs)
    model.eval()

    # 2. thop
    from thop import profile  # noqa: E402

    macs_thop, params = profile(
        model,
        inputs=(DUMMY_INPUT,),
        verbose=False,
    )

    # 3. fvcore
    from fvcore.nn import FlopCountAnalysis  # noqa: E402

    flops_fvcore = FlopCountAnalysis(model, DUMMY_INPUT).total()
    macs_fvcore = flops_fvcore / 2.0  # FLOPs → MACs

    print(f"  params:        {params / 1e6:.4f} M")
    print(f"  MACs (thop):   {macs_thop / 1e9:.4f} G")
    print(f"  MACs (fvcore): {macs_fvcore / 1e9:.4f} G")

    results.append(
        {
            "config": name,
            "params_M": params / 1e6,
            "macs_thop_GMACs": macs_thop / 1e9,
            "macs_fvcore_GMACs": macs_fvcore / 1e9,
        }
    )

# ---------------------------------------------------------------------------
# Сохранение CSV
# ---------------------------------------------------------------------------
csv_path = Path(__file__).resolve().parent / "macs_results.csv"
fieldnames = ["config", "params_M", "macs_thop_GMACs", "macs_fvcore_GMACs"]

with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nРезультаты сохранены в {csv_path}")

# ---------------------------------------------------------------------------
# Вывод таблицы в консоль
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"{'config':<10} {'params_M':<12} {'macs_thop_GMACs':<20} {'macs_fvcore_GMACs':<20}")
print("-" * 80)
for r in results:
    print(
        f"{r['config']:<10} {r['params_M']:<12.4f} {r['macs_thop_GMACs']:<20.4f} {r['macs_fvcore_GMACs']:<20.4f}"
    )
print("=" * 80)