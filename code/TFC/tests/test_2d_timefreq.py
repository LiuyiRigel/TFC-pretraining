"""
批量生成 5方法 × 5信号 的二维时频图
========================================
对每种频域变换方法提取其二维时频表示，可视化对比。
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from freq_transforms import (MultiScaleFFT, CWT_Approx, STFT_Pool, EnvelopeFFT)

N, FS = 2048, 1000
T_ARRAY = np.arange(N) / FS
SNR_DB = 5
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_outputs')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---- 信号生成 (同前) ----
def add_noise(signal, snr_db):
    sp = np.mean(signal**2)
    npow = sp / (10**(snr_db/10))
    return signal + np.random.randn(len(signal)) * np.sqrt(npow)

t = T_ARRAY
s1 = np.sin(2*np.pi*10*t)
f0, f1 = 10, 200; k = (f1-f0)/t[-1]
s2 = np.sin(2*np.pi*(f0*t + 0.5*k*t**2))
s3 = add_noise(s1.copy(), SNR_DB)
s4 = add_noise(s2.copy(), SNR_DB)
bpfo, fc, fm = 60, 400, 25
s5 = np.zeros_like(t)
for kk in range(1,4): s5 += np.sin(2*np.pi*kk*bpfo*t)
res = np.exp(-50*(t%(1/bpfo)))*np.sin(2*np.pi*fc*t)
sb = 0.3*np.sin(2*np.pi*fm*t)*np.sin(2*np.pi*fc*t)
s5c = (0.4*s5 + 0.5*res + 0.1*sb); s5c /= np.max(np.abs(s5c))
s5n = add_noise(s5c.copy(), SNR_DB)

signals = {
    '01_单频': s1, '02_调频': s2,
    '03_单频+噪声': s3, '04_调频+噪声': s4,
    '05_轴承故障+噪声': s5n,
}

sig_labels = list(signals.keys())

# ---- 初始化所有变换 (预计算2D图) ----
def get_2d_maps():
    """返回 dict: method_name -> list of 5 2D arrays"""
    x_tensors = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                 for k, v in signals.items()}

    # FFT: 无2D表示，用STFT代替
    fft_maps, multi_maps, cwt_maps, stft_maps, env_maps = [], [], [], [], []
    multi = MultiScaleFFT(n_samples=N)
    cwt_approx = CWT_Approx(n_samples=N, n_scales=48)
    stft_pool = STFT_Pool(n_samples=N)
    envelope = EnvelopeFFT(n_samples=N)

    for sig_name in sig_labels:
        x = x_tensors[sig_name]
        with torch.no_grad():
            # FFT: 用标准STFT作为二维表示
            xf = x.squeeze(1)
            stft_fft = torch.stft(xf, n_fft=128, hop_length=32, return_complex=True,
                                  window=torch.hann_window(128), pad_mode='reflect')
            fft_maps.append(stft_fft.abs().squeeze().numpy())

            multi_maps.append(multi.get_2d_map(x).numpy())
            cwt_2d = cwt_approx.get_2d_map(x).numpy()
            cwt_maps.append(cwt_2d)
            stft_maps.append(stft_pool.get_2d_map(x).numpy())
            env_maps.append(envelope.get_2d_map(x).numpy())

    return {
        'FFT (STFT)': fft_maps,
        'MultiScaleFFT': multi_maps,
        'CWT_Approx': cwt_maps,
        'STFT_Pool': stft_maps,
        'EnvelopeFFT': env_maps,
    }

print("Computing 2D time-frequency maps...")
all_maps = get_2d_maps()
print("Done.")

# ---- 可视化: 每种方法一张大图，5个子图 ----
for method_name, maps in all_maps.items():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (ax, sig_name, spec) in enumerate(zip(axes, sig_labels, maps)):
        # 取前 256 个频率 bin 以便查看
        display_spec = spec[:min(256, spec.shape[0]), :]
        im = ax.imshow(display_spec, aspect='auto', origin='lower',
                       cmap='magma', extent=[0, N/FS, 0, FS/2])
        ax.set_title(sig_name.replace('_', ' '), fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Freq (Hz)', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.7)

    # 隐藏多余的子图
    for idx in range(len(sig_labels), len(axes)):
        axes[idx].set_visible(False)

    safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
    fig.suptitle(f'{method_name} — 2D Time-Frequency Representations', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fname = f'05_2d_{safe_name}.png'
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()
    print(f"[OK] {fname}")

# ---- 额外: 5×5 总览矩阵 (缩略图) ----
methods = list(all_maps.keys())
fig, axes = plt.subplots(5, 5, figsize=(20, 18))

for i, method_name in enumerate(methods):
    for j, sig_name in enumerate(sig_labels):
        ax = axes[j, i]
        spec = all_maps[method_name][j]
        display_spec = spec[:min(128, spec.shape[0]), :]
        ax.imshow(display_spec, aspect='auto', origin='lower', cmap='magma')
        if j == 0:
            ax.set_title(method_name, fontsize=8, fontweight='bold')
        if i == 0:
            ax.set_ylabel(sig_name.replace('_', '\n'), fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

fig.suptitle('All Methods × All Signals — 2D Time-Frequency Overview', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '05_2d_all_methods_grid.png'), dpi=150)
plt.close()
print("[OK] 05_2d_all_methods_grid.png")
print(f"\n✅ All 2D plots saved to: {OUTPUT_DIR}")
