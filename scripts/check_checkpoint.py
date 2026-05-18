"""Test checkpoint save/load round-trip for DSCConvTasNet."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
import pathlib

from src.models.dsc_conv_tasnet import DSCConvTasNet
from src.utils import save_checkpoint, load_checkpoint

model = DSCConvTasNet(num_sources=4, audio_channels=2, dsc_layers=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
config = {'test': True, 'dsc_layers': 10}

with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp) / 'test.pt'

    # Сохранить
    save_checkpoint(path, model, optimizer, epoch=5,
                    best_val_loss=0.123, config=config, seed=42)

    # Загрузить в новую модель
    model2 = DSCConvTasNet(num_sources=4, audio_channels=2, dsc_layers=10)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    ckpt = load_checkpoint(path, model2, optimizer2, torch.device('cpu'))

    # Проверить что веса совпадают
    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), model2.state_dict().items()
    ):
        assert torch.allclose(v1, v2), f'Mismatch in {k1}'

    assert ckpt['epoch'] == 5
    assert abs(ckpt['best_val_loss'] - 0.123) < 1e-6
    assert ckpt['config'] == config
    assert ckpt['seed'] == 42

print('Checkpoint save/load: OK')
