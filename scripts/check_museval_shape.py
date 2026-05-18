"""Check that museval.metrics.bss_eval accepts the correct array shape."""
import numpy as np
import museval

# Симулируем один трек: 3 секунды, стерео, 4 источника
sr = 44100
duration = 3
samples = sr * duration
channels = 2
n_sources = 4

ref = np.random.randn(n_sources, samples, channels).astype(np.float32)
est = ref + 0.01 * np.random.randn(*ref.shape).astype(np.float32)

# Тест: SDR должен быть высоким (~40 dB) когда est ≈ ref
for src_idx in range(n_sources):
    sdr, isr, sir, sar, perm = museval.metrics.bss_eval(
        ref[src_idx:src_idx+1],  # [1, samples, channels]
        est[src_idx:src_idx+1],  # [1, samples, channels]
    )
    sdr_val = float(np.nanmedian(sdr))
    sir_val = float(np.nanmedian(sir))
    print(f'Source {src_idx}: SDR={sdr_val:.1f} dB, SIR={sir_val:.1f} dB')

print()
print('Expected: SDR >> 20 dB for near-identical signals')
print('If SDR is near 0 or negative — shape is wrong!')
