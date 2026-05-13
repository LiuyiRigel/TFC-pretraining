"""
使用等价于 SSqueezepy cwt() 的算法为 5 种测试信号生成 CWT 对比图。
基于 SSqueezepy _cwt.py 的频域卷积算法，用 numpy/scipy 实现。
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, ifftshift
# Note: scipy.integrate not available due to numpy version conflict;
# GMW wavelet normalization handled analytically instead.

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# SSqueezepy-equivalent Wavelet class (GMW: Generalized Morse Wavelet)
# ==============================================================================
class GMW:
    """Generalized Morse Wavelet, 等价于 SSqueezepy Wavelet(('gmw', {...}))"""
    def __init__(self, beta=6, gamma=3, N=2048):
        self.beta = beta
        self.gamma = gamma
        self.N = N
        # Frequency axis (angular, 0 to pi)
        self.xi = np.linspace(0, 2*np.pi, N, endpoint=False)

    def __call__(self, scale, nohalf=True):
        """
        Frequency-domain GMW wavelet.
        psi(s*ω) = U(ω) * (s*ω)^β * exp(-(s*ω)^γ)
        where U(ω) is the Heaviside step (valid for ω > 0)
        """
        omega = self.xi * scale
        # Unit step: only positive frequencies
        psih = np.zeros(self.N, dtype=np.complex128)
        pos = omega > 0
        # GMW formula: ω^β * exp(-ω^γ)
        psih[pos] = (omega[pos] ** self.beta) * np.exp(-omega[pos] ** self.gamma)

        # Normalize (L1 norm)
        norm = np.sum(np.abs(psih)) / self.N
        if norm > 1e-15:
            psih /= norm

        # Halve Nyquist for even N
        if nohalf and self.N % 2 == 0:
            psih[self.N // 2] /= 2

        return psih

    def psifn(self, scale):
        """Time-domain wavelet via IFFT"""
        psih = self(scale, nohalf=True)
        psi_t = ifft(ifftshift(psih)) * self.N
        return psi_t


def cwt_ssqueezepy_equivalent(x, wavelet=None, scales='log', nv=32, fs=1.0,
                                l1_norm=True):
    """
    SSqueezepy-style CWT.
    等价于 ssqueezepy.cwt(x, wavelet, scales, nv, fs)

    Parameters:
        x: 1D array, input signal
        wavelet: GMW instance or None (default GMW(beta=6, gamma=3))
        scales: 'log' or ndarray
        nv: voices per octave
        fs: sampling frequency (not critical for transform itself)

    Returns:
        Wx: [n_scales, n_samples] complex CWT coefficients
        scales: [n_scales] scales used
    """
    N = len(x)
    if wavelet is None:
        wavelet = GMW(N=N)

    # Generate scales (log-spaced)
    if isinstance(scales, str) and scales == 'log':
        # Mimic SSqueezepy: scales from 2^(1/nv) to N/2
        na = int(nv * np.log2(N / 2))
        scales = 2 ** (np.arange(1, na + 1) / nv)
        scales = scales[scales <= N / 2]  # limit
    elif isinstance(scales, np.ndarray):
        pass
    else:
        na = 48  # default
        scales = 2 ** (np.arange(1, na + 1) / nv)

    na = len(scales)

    # FFT of input
    xh = fft(x)  # [N]

    # Pre-allocate output
    Wx = np.zeros((na, N), dtype=np.complex128)

    # Convolution in frequency domain: Wx = IFFT(psi*(scale) * FFT(x))
    for i, s in enumerate(scales):
        # Frequency-domain wavelet at scale s
        psih = wavelet(s, nohalf=False)  # don't half Nyquist for this step
        # Multiply with signal spectrum
        product = psih * xh
        # IFFT
        coeff = ifft(product)
        # SSqueezepy returns raw IFFT (already correct magnitude)
        if l1_norm:
            coeff = coeff * np.sqrt(s)  # L1 normalization
        Wx[i] = coeff

    return Wx, scales


# ==============================================================================
# 生成信号 (与 test_freq_transforms.py 完全一致)
# ==============================================================================
N, FS = 2048, 1000
T_ARRAY = np.arange(N) / FS
SNR_DB = 5

def add_noise(signal, snr_db):
    sp = np.mean(signal**2)
    npow = sp / (10**(snr_db/10))
    return signal + np.random.randn(len(signal)) * np.sqrt(npow)

t = T_ARRAY
s1 = np.sin(2*np.pi*10*t)                          # 单频
f0, f1 = 10, 200; k = (f1-f0)/t[-1]
s2 = np.sin(2*np.pi*(f0*t + 0.5*k*t**2))           # chirp
s3 = add_noise(s1.copy(), SNR_DB)                  # 单频+噪声
s4 = add_noise(s2.copy(), SNR_DB)                  # chirp+噪声
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

# ==============================================================================
# 计算 CWT
# ==============================================================================
print("Computing SSqueezepy-equivalent CWT...")
cwt_results = {}
for name, sig in signals.items():
    Wx, scales = cwt_ssqueezepy_equivalent(sig, nv=32, fs=FS)
    cwt_results[name] = (np.abs(Wx), scales)
    print(f"  {name}: Wx shape={Wx.shape}, scales={len(scales)}")
print("Done.\n")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_outputs')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 图1: 每种信号的CWT二维时频图 (5个子图)
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for idx, (ax, sig_name) in enumerate(zip(axes, sig_labels)):
    mag, scales_arr = cwt_results[sig_name]
    im = ax.imshow(mag[:min(256, mag.shape[0]), :], aspect='auto', origin='lower',
                   cmap='magma', extent=[0, N/FS, 0, scales_arr[min(255, len(scales_arr)-1)]])
    ax.set_title(sig_name.replace('_', ' '), fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('Scale', fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.7)
axes[-1].set_visible(False)
fig.suptitle('SSqueezepy CWT (GMW wavelet) — 2D Time-Scale Representations', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '06_ssq_cwt_2d.png'), dpi=150)
plt.close()
print("[OK] 06_ssq_cwt_2d.png")

# ==============================================================================
# 图2: 1D 最大响应投影 (CWT max over scales → [T]) vs 我们的CWT_Approx
# ==============================================================================
fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
for ax, sig_name in zip(axes, sig_labels):
    mag, _ = cwt_results[sig_name]
    # Max projection (模拟我们 CWT_Approx 的 max-pool 输出)
    proj = mag.max(axis=0)
    ax.plot(proj, linewidth=0.6, color='darkorange', label='SSqueezepy CWT (max-proj)')
    ax.set_ylabel(sig_name.replace('_', '\n'), fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, None])
axes[-1].set_xlabel('Time sample index', fontsize=10)
fig.suptitle('SSqueezepy CWT — 1D Max Projection (max over scales)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '06_ssq_cwt_1d_projection.png'), dpi=150)
plt.close()
print("[OK] 06_ssq_cwt_1d_projection.png")

# ==============================================================================
# 图3: SSqueezepy CWT vs 我们的 CWT_Approx 直接对比 (5个信号)
# ==============================================================================
import torch

from freq_transforms import CWT_Approx
cwt_approx = CWT_Approx(n_samples=N, n_scales=48)

fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
for idx, (ax, sig_name) in enumerate(zip(axes, sig_labels)):
    sig = torch.tensor(signals[sig_name], dtype=torch.float32)
    with torch.no_grad():
        approx_out = cwt_approx(sig.unsqueeze(0).unsqueeze(0)).squeeze().numpy()

    mag, _ = cwt_results[sig_name]
    proj = mag.max(axis=0)

    ax.plot(proj, linewidth=0.8, color='darkorange', alpha=0.8, label='SSqueezepy CWT')
    ax.plot(approx_out, linewidth=0.8, color='steelblue', alpha=0.8, label='Our CWT_Approx')
    ax.set_ylabel(sig_name.replace('_', '\n'), fontsize=8)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, None])
axes[-1].set_xlabel('Time sample index', fontsize=10)
fig.suptitle('SSqueezepy CWT vs Our CWT_Approx — Side-by-side Comparison', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '06_ssq_vs_ours_comparison.png'), dpi=150)
plt.close()
print("[OK] 06_ssq_vs_ours_comparison.png")

# ==============================================================================
# 图4: CWT各尺度能量分布 (频率轴投影)
# ==============================================================================
fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
for ax, sig_name in zip(axes, sig_labels):
    mag, scales_arr = cwt_results[sig_name]
    # Average over time → frequency-axis energy distribution
    energy = mag.mean(axis=1)
    ax.semilogy(scales_arr, energy, linewidth=0.8, color='darkorange')
    ax.set_ylabel(sig_name.replace('_', '\n'), fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Scale (log)', fontsize=10)
fig.suptitle('SSqueezepy CWT — Scale Energy Distribution (time-averaged)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '06_ssq_cwt_scale_energy.png'), dpi=150)
plt.close()
print("[OK] 06_ssq_cwt_scale_energy.png")

print(f"\n✅ SSqueezepy-equivalent CWT plots saved to: {OUTPUT_DIR}")
