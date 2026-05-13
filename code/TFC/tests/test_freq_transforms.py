"""
TFC freq_transforms 信号测试与可视化
========================================
生成 5 种测试信号，验证 4 种频域变换的效果。

信号类型 (点数: 2048):
  1. 单频率正弦波 (10 Hz, fs=1000)
  2. 线性调频 chirp (10 Hz → 200 Hz)
  3. 单频 + 高斯白噪声 (SNR=5dB)
  4. chirp + 高斯白噪声 (SNR=5dB)
  5. 轴承故障仿真 (BPFO ≈ 60Hz + 谐波 + SNR=5dB)

所有噪声信号的 SNR 统一为 5dB。

运行:
    python test_freq_transforms.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
import torch

# 确保可以 import freq_transforms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from freq_transforms import (MultiScaleFFT, CWT_Approx, STFT_Pool, EnvelopeFFT,
                              get_freq_transform)

# ==============================================================================
# 配置
# ==============================================================================
N = 2048          # 样本点数
FS = 1000         # 采样率 Hz
T_ARRAY = np.arange(N) / FS  # 时间轴
SNR_DB = 5        # 噪声信号的信噪比 (dB)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 变换名称映射
TRANSFORM_NAMES = {'fft': 'FFT', 'multiscale': 'MultiScaleFFT',
                   'cwt_approx': 'CWT_Approx', 'stft_pool': 'STFT_Pool',
                   'envelope': 'EnvelopeFFT'}

# matplotlib 中英文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 信号生成
# ==============================================================================
def add_noise(signal, snr_db):
    """向信号添加高斯白噪声, 使输出 SNR = snr_db dB"""
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
    return signal + noise


def compute_snr(clean, noisy):
    """计算 SNR (dB)"""
    noise = noisy - clean
    Ps = np.mean(clean ** 2)
    Pn = np.mean(noise ** 2)
    return 10 * np.log10(Ps / Pn) if Pn > 0 else np.inf


def signal_single_freq(t):
    """1. 单频率正弦波: 10 Hz"""
    return np.sin(2 * np.pi * 10 * t)


def signal_chirp(t):
    """2. 线性调频: 10 Hz → 200 Hz"""
    f0, f1 = 10, 200
    k = (f1 - f0) / t[-1]
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t ** 2))


def signal_bearing_fault(t):
    """
    3. 轴承外圈故障仿真信号
    BPFO ≈ 60 Hz, 带 2 次和 3 次谐波, 加调制边带
    """
    bpfo = 60                    # 外圈故障特征频率
    fc = 400                     # 载波/共振频率
    fm = 25                      # 调制频率

    # 主冲击系列 (3 个谐波)
    s = np.zeros_like(t)
    for k in range(1, 4):
        s += np.sin(2 * np.pi * k * bpfo * t)

    # 共振调制包络
    resonance = np.exp(-50 * (t % (1 / bpfo))) * np.sin(2 * np.pi * fc * t)
    # 调制边带
    sideband = 0.3 * np.sin(2 * np.pi * fm * t) * np.sin(2 * np.pi * fc * t)

    y = 0.4 * s + 0.5 * resonance + 0.1 * sideband
    return y / np.max(np.abs(y))  # 归一化


def generate_all_signals():
    """生成全部 5 种信号, 返回 dict"""
    t = T_ARRAY

    s1_clean = signal_single_freq(t)
    s2_clean = signal_chirp(t)
    s5_clean = signal_bearing_fault(t)

    # 含噪声版本: 统一 SNR=5dB
    s3_clean = signal_single_freq(t)
    s4_clean = signal_chirp(t)

    signals = {
        '01_单频_10Hz':           s1_clean,
        '02_线性调频_10to200Hz':  s2_clean,
        '03_单频_噪声_5dB':       add_noise(s3_clean, SNR_DB),
        '04_调频_噪声_5dB':       add_noise(s4_clean, SNR_DB),
        '05_轴承故障_噪声_5dB':    add_noise(s5_clean, SNR_DB),
    }

    # 验证 SNR
    print("=" * 60)
    print(f"信号 SNR 验证 (目标: {SNR_DB} dB)")
    print("-" * 60)
    print(f"{'单频_噪声':>18s}:  SNR = {compute_snr(s3_clean, signals['03_单频_噪声_5dB']):.2f} dB")
    print(f"{'调频_噪声':>18s}:  SNR = {compute_snr(s4_clean, signals['04_调频_噪声_5dB']):.2f} dB")
    print(f"{'轴承故障_噪声':>18s}:  SNR = {compute_snr(s5_clean, signals['05_轴承故障_噪声_5dB']):.2f} dB")
    print("=" * 60)

    return signals


# ==============================================================================
# 运行变换
# ==============================================================================
def run_transforms(signals):
    """对每种信号, 运行全部 4 种变换 (含 FFT baseline)"""
    transform_types = ['fft', 'multiscale', 'cwt_approx', 'stft_pool', 'envelope']
    results = {}

    for sig_name, sig_array in signals.items():
        # 转换为 [B=1, C=1, T] 张量
        x = torch.tensor(sig_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = x.to(DEVICE)

        results[sig_name] = {}
        for ttype in transform_types:
            transform = get_freq_transform(ttype, n_samples=N)
            with torch.no_grad():
                if transform is None:
                    out = torch.fft.fft(x, dim=-1).abs()
                else:
                    transform = transform.to(DEVICE)
                    out = transform(x)
            results[sig_name][ttype] = out.squeeze().cpu().numpy()

    return results


# ==============================================================================
# 可视化
# ==============================================================================
def plot_signals(signals):
    """图1: 5种时域信号 (原始)"""
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 5))

    for ax, (name, sig), c in zip(axes, signals.items(), colors):
        ax.plot(T_ARRAY, sig, color=c, linewidth=0.6)
        ax.set_ylabel(name.replace('_', '\n'), fontsize=9)
        ax.grid(True, alpha=0.3)
        if '噪声' not in name:
            ax.set_ylim([-1.2, 1.2])

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle('Test Signals (N=2048, fs=1000 Hz)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '01_signals_time_domain.png'), dpi=150)
    plt.close()
    print("[OK] 01_signals_time_domain.png")


def plot_transforms_by_signal(signals, results):
    """
    图2: 每种信号 × 5种变换 = 5行5列子图
    每行: 一种信号; 每列: 一种频域变换
    """
    sig_names = list(signals.keys())
    transform_types = ['fft', 'multiscale', 'cwt_approx', 'stft_pool', 'envelope']
    n_sigs, n_trans = len(sig_names), len(transform_types)

    fig, axes = plt.subplots(n_sigs, n_trans, figsize=(18, 14))

    for i, sig_name in enumerate(sig_names):
        for j, ttype in enumerate(transform_types):
            ax = axes[i, j]
            spec = results[sig_name][ttype]
            ax.plot(spec, linewidth=0.5, color='steelblue')
            ax.set_xlim([0, len(spec)])
            # 标注 Y 轴范围
            ax.set_ylim([0, None])

            if i == 0:
                ax.set_title(TRANSFORM_NAMES[ttype], fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel(sig_name.replace('_', '\n'), fontsize=8)
            ax.grid(True, alpha=0.2)

    fig.suptitle('Frequency-domain Representations (per signal × method)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '02_all_transforms_grid.png'), dpi=150)
    plt.close()
    print("[OK] 02_all_transforms_grid.png")


def plot_transforms_by_method(signals, results):
    """
    图3: 同一种变换 × 5种信号 对比
    每种变换一张图, 5个子图堆叠
    """
    sig_names = list(signals.keys())
    transform_types = ['fft', 'multiscale', 'cwt_approx', 'stft_pool', 'envelope']

    for ttype in transform_types:
        fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, 5))

        for ax, sig_name, c in zip(axes, sig_names, colors):
            spec = results[sig_name][ttype]
            ax.plot(spec, linewidth=0.6, color=c)
            ax.set_ylabel(sig_name.replace('_', '\n'), fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, None])

        axes[-1].set_xlabel('Frequency bin / Time index', fontsize=10)
        fig.suptitle(f'{TRANSFORM_NAMES[ttype]} — All Signals Comparison', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fname = f'03_method_{ttype}_comparison.png'
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
        plt.close()
        print(f"[OK] {fname}")


def plot_stft_detail(signals):
    """
    图4: 对 chirp 和 bearing 信号做标准 STFT 二维时频图,
    展示非平稳信号的时变频率特征 (用于对比 1D 表示)
    """
    sig_names = ['02_线性调频_10to200Hz', '05_轴承故障_噪声_5dB']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sname in zip(axes, sig_names):
        x = torch.tensor(signals[sname], dtype=torch.float32)
        mag = torch.stft(x, n_fft=128, hop_length=32, return_complex=True,
                         window=torch.hann_window(128), pad_mode='reflect').abs()
        im = ax.imshow(mag.numpy(), aspect='auto', origin='lower', cmap='magma',
                       extent=[0, N / FS, 0, FS / 2])
        ax.set_title(sname.replace('_', ' '), fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('STFT Spectrograms (2D time-frequency)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '04_stft_spectrograms.png'), dpi=150)
    plt.close()
    print("[OK] 04_stft_spectrograms.png")


# ==============================================================================
# 主函数
# ==============================================================================
def main():
    print(f"Device: {DEVICE}")
    print(f"Signal length: {N}, Sampling rate: {FS} Hz, SNR: {SNR_DB} dB")
    print("=" * 60)

    # 1. 生成信号
    signals = generate_all_signals()

    # 2. 运行变换
    print("\nRunning frequency transforms...")
    results = run_transforms(signals)
    print("Done.\n")

    # 3. 可视化
    print("Generating plots...")
    plot_signals(signals)
    plot_transforms_by_signal(signals, results)
    plot_transforms_by_method(signals, results)
    plot_stft_detail(signals)

    print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
