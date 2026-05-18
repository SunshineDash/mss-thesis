"""Parameter count ablation — for Table 3.1 of the thesis."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.conv_tasnet import ConvTasNet
from src.models.dsc_conv_tasnet import DSCConvTasNet
from src.utils import count_parameters

base = ConvTasNet(num_sources=4, audio_channels=2)
configs = [
    ('baseline (L=0)',  DSCConvTasNet(num_sources=4, audio_channels=2, dsc_layers=0)),
    ('DSC-5  (L=5)',    DSCConvTasNet(num_sources=4, audio_channels=2, dsc_layers=5)),
    ('DSC-10 (L=10)',   DSCConvTasNet(num_sources=4, audio_channels=2, dsc_layers=10)),
    ('DSC-20 (L=20)',   DSCConvTasNet(num_sources=4, audio_channels=2, dsc_layers=20)),
    ('DSC-full (L=24)', DSCConvTasNet(num_sources=4, audio_channels=2, dsc_layers=24)),
]

print(f'BaseConvTasNet params: {count_parameters(base):,}')
print()
for name, m in configs:
    p = count_parameters(m)
    reduction = (1 - p / count_parameters(base)) * 100
    print(f'{name}: {p:,} params  ({reduction:+.1f}% vs baseline)')

# Проверка: L=0 должен давать то же число параметров что и baseline
assert count_parameters(configs[0][1]) == count_parameters(base), \
    'FAIL: DSCConvTasNet(L=0) != ConvTasNet по числу параметров!'
print()
print('Ablation param counts: OK')
