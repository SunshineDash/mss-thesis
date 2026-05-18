import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import Dataset, DataLoader
from src.models.conv_tasnet import ConvTasNet
from src.models.dsc_conv_tasnet import DSCConvTasNet
from src.losses import si_sdr_loss
from src.utils import count_parameters


class FakeDataset(Dataset):
    def __len__(self): return 8
    def __getitem__(self, i):
        mixture = torch.randn(2, 44100 * 4)
        sources = torch.randn(4, 2, 44100 * 4)
        return mixture, sources


loader = DataLoader(FakeDataset(), batch_size=2)

for name, model in [
    ('ConvTasNet baseline', ConvTasNet(num_sources=4, audio_channels=2)),
    ('DSCConvTasNet L=10',  DSCConvTasNet(num_sources=4, audio_channels=2, dsc_layers=10)),
]:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
    losses = []
    for epoch in range(2):
        for mix, src in loader:
            est = model(mix)
            B, S, C, T = est.shape
            loss = si_sdr_loss(est.reshape(B, S*C, T), src.reshape(B, S*C, T))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
    scheduler.step(losses[-1])
    print(f'{name}: final_loss={losses[-1]:.3f}, params={count_parameters(model):,}')
    print(f'  backward: OK, scheduler.step: OK')

print()
print('MINI RUN PASSED')
