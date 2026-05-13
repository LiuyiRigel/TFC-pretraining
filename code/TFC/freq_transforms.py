# Non-stationary frequency transforms for TFC training.
# Provides differentiable alternatives to torch.fft for time-frequency analysis
# of non-stationary signals (e.g., bearing vibration, EEG, EMG).

# All transforms:
# - Are PyTorch-native (fully differentiable, GPU-compatible)
# - Accept input shape [B, C, T] and output [B, C, T] (same as torch.fft.fft)
# - Are designed for the TFC contrastive learning framework

# Transform options:
# 1. MultiScaleFFT       - FFT with learnable multi-scale windowing
# 2. CWT_Approx          - Approximate CWT via learnable filter bank (Morlet-like)
# 3. STFT_Pool           - STFT magnitude pooled back to 1D (frequency-aware)
# 4. HilbertFFT          - Hilbert-Huang inspired: envelope + FFT fusion

# Author: Adapted for TFC non-stationary signal analysis


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==============================================================================
# 1. MultiScaleFFT: 多尺度窗 FFT（类似多分辨率 STFT）
# ==============================================================================
class MultiScaleFFT(nn.Module):
    """
    Multi-scale FFT: applies several window sizes, computes FFT for each,
    then pools back to original length. Captures both transient (short window)
    and tonal (long window) components.

    Shape: [B, C, T] -> [B, C, T]
    """
    def __init__(self, n_samples=178, window_sizes=[8, 16, 32, 64], stride=4):
        super().__init__()
        self.n_samples = n_samples
        self.window_sizes = window_sizes
        self.stride = stride

        # Learnable combination weights for each scale
        self.scale_weights = nn.Parameter(torch.ones(len(window_sizes)) / len(window_sizes))

        # Learnable projection to map pooled STFT back to n_samples
        total_pooled = sum((n_samples // stride) for _ in window_sizes)
        self.proj = nn.Linear(total_pooled, n_samples)

    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        Returns:
            out: [B, C, T]  magnitude spectrum (real-valued, non-negative)
        """
        B, C, T = x.shape
        x_flat = x.reshape(B * C, T)  # [B*C, T]

        outputs = []
        weights = torch.softmax(self.scale_weights, dim=0)

        for i, win_size in enumerate(self.window_sizes):
            # Compute STFT with specific window
            # torch.stft returns [freq_bins, time_frames] complex
            stft_out = torch.stft(
                x_flat,
                n_fft=win_size,
                hop_length=self.stride,
                win_length=win_size,
                window=torch.hann_window(win_size, device=x.device),
                return_complex=True,
                pad_mode='reflect'
            )  # [B*C, freq_bins, time_frames]

            mag = stft_out.abs()  # [B*C, freq_bins, time_frames]

            # Pool over frequency dimension (mean) -> [B*C, time_frames]
            mag_pooled = mag.mean(dim=1)  # [B*C, time_frames]

            # Interpolate to T
            mag_pooled = F.interpolate(
                mag_pooled.unsqueeze(1),
                size=T,
                mode='linear',
                align_corners=False
            ).squeeze(1)  # [B*C, T]

            outputs.append(mag_pooled * weights[i])

        # Sum weighted multi-scale outputs
        out = torch.stack(outputs, dim=-1).sum(dim=-1)  # [B*C, T]

        # Optional: project through learnable linear layer
        # out = self.proj(out)  # uncomment if you want learnable combination

        out = out.reshape(B, C, T)
        return out

    def get_2d_map(self, x):
        """返回二维时频图 [n_freq, n_time] 用于可视化"""
        B, C, T = x.shape
        x_flat = x.reshape(B * C, T)
        maps = []
        weights = torch.softmax(self.scale_weights, dim=0)
        max_freq_bins = max(w // 2 + 1 for w in self.window_sizes)
        for i, win_size in enumerate(self.window_sizes):
            stft_out = torch.stft(x_flat, n_fft=win_size, hop_length=self.stride,
                                  win_length=win_size,
                                  window=torch.hann_window(win_size, device=x.device),
                                  return_complex=True, pad_mode='reflect')
            mag = stft_out.abs().squeeze(0)  # [n_freq, n_time]
            # Interpolate freq dimension to max_freq_bins via unsqueeze
            if mag.shape[0] != max_freq_bins:
                mag = mag.unsqueeze(0).unsqueeze(0)  # [1, 1, n_freq, n_time]
                mag = torch.nn.functional.interpolate(
                    mag, size=(max_freq_bins, mag.shape[-1]),
                    mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0)  # back to [n_freq, n_time]
            maps.append(mag * weights[i])
        return torch.stack(maps).mean(dim=0)


# ==============================================================================
# 2. CWT_Approx v2: GMW wavelet filterbank (SSqueezepy-equivalent, differentiable)
# ==============================================================================
class CWT_Approx(nn.Module):
    """
    Differentiable CWT using Generalized Morse Wavelet (GMW) filterbank.

    Matches SSqueezepy's cwt(wavelet='gmw') algorithm:
      psi(ω) = U(ω) · ω^β · exp(-ω^γ)
    where U(ω) is the Heaviside step function.

    Improvements over v1:
      - GMW kernel instead of simple Morlet (matches SSqueezepy)
      - Vectorized computation (all scales processed in parallel)
      - Higher default n_scales (96, equivalent to nv=16 for T=2048)
      - L1-normalized wavelet responses
      - Mean energy pooling (better than max for noisy signals)

    Shape: [B, C, T] -> [B, C, T]
    """

    def __init__(self, n_samples=178, n_scales=96,
                 beta_init=3.0, gamma_init=6.0,  # GMW parameters
                 pooling='energy'):  # 'energy', 'max', 'weighted'
        super().__init__()
        self.n_samples = n_samples
        self.n_scales = n_scales
        self.pooling = pooling
        self.eps = 1e-8

        # Log-spaced scales (matching SSqueezepy 'log-piecewise')
        # Scales from ~2 to n_samples/2
        log_s_min = np.log(2.0)
        log_s_max = np.log(n_samples / 2.0)
        log_scales = torch.linspace(log_s_min, log_s_max, n_scales)
        self.log_scales = nn.Parameter(log_scales)

        # GMW wavelet parameters (learnable)
        # beta controls narrowness in time; gamma controls narrowness in frequency
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

        # Per-scale amplitude factors (learnable, initialized to 1)
        self.scale_factors = nn.Parameter(torch.ones(n_scales))

        # Pooling weights (if pooling='weighted')
        self.pool_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

    def _gmw_fd(self, omega, beta, gamma):
        """
        Frequency-domain GMW wavelet (non-zero for ω > 0 only).

        psi(ω) = ω^β · exp(-ω^γ)  for ω > 0, else 0.

        Args:
            omega: [n_scales, n_freq]  angular frequency (scaled by s)
            beta: scalar
            gamma: scalar
        Returns:
            psi: [n_scales, n_freq]  complex wavelet in frequency domain
        """
        # Heaviside step: only positive frequencies
        pos = omega > 0
        psi = torch.zeros_like(omega)

        safe_omega = omega[pos]
        psi[pos] = (safe_omega ** beta) * torch.exp(-safe_omega ** gamma)

        # L1 normalize per scale
        norm = psi.abs().sum(dim=-1, keepdim=True) / omega.shape[-1]
        psi = psi / (norm + self.eps)

        return psi

    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        Returns:
            out: [B, C, T]  magnitude representation pooled over scales
        """
        B, C, T = x.shape
        device = x.device

        # Full FFT (complex) along time dimension
        x_fft = torch.fft.fft(x, dim=-1)  # [B, C, T] complex

        # Frequency axis (angular, 0 to 2π)
        omega_1d = torch.linspace(0, 2 * np.pi, T, device=device)  # [T]

        # Scales (positive, from log-space)
        scales = torch.exp(self.log_scales)  # [n_scales]

        # Ensure positivity
        beta = torch.clamp(self.beta, 1.0, 20.0)
        gamma = torch.clamp(self.gamma, 1.0, 20.0)

        # ---- Vectorized wavelet computation ----
        # omega = outer(scales, omega_1d): [n_scales, T]
        omega = scales.unsqueeze(1) * omega_1d.unsqueeze(0)  # [n_scales, T]

        # GMW wavelet in frequency domain: [n_scales, T]
        psih = self._gmw_fd(omega, beta, gamma)

        # Halve Nyquist bin for even T (SSqueezepy convention)
        if T % 2 == 0:
            psih[:, T // 2] = psih[:, T // 2] / 2.0

        # Apply per-scale factors: [n_scales, 1, 1]
        sf = self.scale_factors.view(self.n_scales, 1, 1)

        # Multiply in frequency domain: x_fft: [B, C, T] × psih: [n_scales, T]
        # -> [B, C, n_scales, T] via broadcasting
        filtered_fft = x_fft.unsqueeze(2) * psih.unsqueeze(0).unsqueeze(0) * sf
        # [B, C, n_scales, T]

        # IFFT back to time domain
        filtered_time = torch.fft.ifft(filtered_fft, dim=-1)  # [B, C, n_scales, T]

        # Magnitude
        mag = filtered_time.abs()  # [B, C, n_scales, T]

        # ---- Pool over scales to get [B, C, T] ----
        if self.pooling == 'max':
            out = mag.max(dim=2)[0]  # [B, C, T]
        elif self.pooling == 'weighted':
            w = torch.softmax(self.pool_weights, dim=0).view(1, 1, -1, 1)
            out = (mag * w).sum(dim=2)  # [B, C, T]
        else:  # 'energy' (default, SSqueezepy-like)
            out = mag.mean(dim=2)  # [B, C, T]

        return out

    def get_2d_map(self, x):
        """返回二维时频图 [n_scales, n_time] 用于可视化"""
        B, C, T = x.shape
        device = x.device

        x_fft = torch.fft.fft(x, dim=-1)
        omega_1d = torch.linspace(0, 2 * np.pi, T, device=device)
        scales = torch.exp(self.log_scales)
        beta = torch.clamp(self.beta, 1.0, 20.0)
        gamma = torch.clamp(self.gamma, 1.0, 20.0)

        omega = scales.unsqueeze(1) * omega_1d.unsqueeze(0)
        psih = self._gmw_fd(omega, beta, gamma)
        if T % 2 == 0:
            psih[:, T // 2] = psih[:, T // 2] / 2.0

        sf = self.scale_factors.view(self.n_scales, 1, 1)
        filtered_fft = x_fft.unsqueeze(2) * psih.unsqueeze(0).unsqueeze(0) * sf
        filtered_time = torch.fft.ifft(filtered_fft, dim=-1)
        # [B, C, n_scales, T] → take first batch, first channel
        mag = filtered_time[0, 0].abs()  # [n_scales, T]

        return mag.detach()


# ==============================================================================
# 3. STFT_Pool: STFT 幅度谱池化回1D（最接近 FFT 但有时频局部性）
# ==============================================================================
class STFT_Pool(nn.Module):
    """
    Computes STFT magnitude, then pools frequency bins back to original length.
    Retains some time-frequency locality unlike global FFT.

    Shape: [B, C, T] -> [B, C, T]
    """
    def __init__(self, n_samples=178, n_fft=32, hop_length=8, pool_mode='max'):
        super().__init__()
        self.n_samples = n_samples
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pool_mode = pool_mode

        n_freq_bins = n_fft // 2 + 1
        n_time_frames = (n_samples - n_fft) // hop_length + 1

        # Learnable projection: [n_freq_bins, n_time_frames] -> [1, T]
        self.proj = nn.Linear(n_freq_bins * n_time_frames, n_samples)

        # Or: learnable freq weights for pooling
        self.freq_weights = nn.Parameter(torch.ones(n_freq_bins) / n_freq_bins)

    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        Returns:
            out: [B, C, T]
        """
        B, C, T = x.shape
        x_flat = x.reshape(B * C, T)

        # STFT
        stft_out = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft, device=x.device),
            return_complex=True,
            pad_mode='reflect'
        )  # [B*C, n_freq_bins, n_time_frames]

        mag = stft_out.abs()  # [B*C, n_freq_bins, n_time_frames]

        # Weighted pool over frequency
        w = torch.softmax(self.freq_weights, dim=0)
        mag_weighted = mag * w[:, None]  # [B*C, n_freq, n_time]
        mag_pooled = mag_weighted.sum(dim=1)  # [B*C, n_time]

        # Interpolate to T
        mag_pooled = F.interpolate(
            mag_pooled.unsqueeze(1),
            size=T,
            mode='linear',
            align_corners=False
        ).squeeze(1)  # [B*C, T]

        out = mag_pooled.reshape(B, C, T)
        return out

    def get_2d_map(self, x):
        """返回二维时频图 [n_freq, n_time]"""
        B, C, T = x.shape
        x_flat = x.reshape(B * C, T)
        stft_out = torch.stft(x_flat, n_fft=self.n_fft, hop_length=self.hop_length,
                              win_length=self.n_fft,
                              window=torch.hann_window(self.n_fft, device=x.device),
                              return_complex=True, pad_mode='reflect')
        return stft_out.abs().squeeze(0)


# ==============================================================================
# 4. EnvelopeFFT: 包络谱 + FFT 融合（滚动轴承故障检测常用）
# ==============================================================================
class EnvelopeFFT(nn.Module):
    """
    Envelope spectrum + FFT fusion.
    Inspired by bearing fault diagnosis: Hilbert envelope → FFT of envelope
    captures modulation frequencies (fault frequencies).

    Output = α * FFT_magnitude + (1-α) * Envelope_Spectrum

    Shape: [B, C, T] -> [B, C, T]
    """
    def __init__(self, n_samples=178, alpha_init=0.5):
        super().__init__()
        self.n_samples = n_samples

        # Learnable fusion weight
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        Returns:
            out: [B, C, T]
        """
        B, C, T = x.shape

        # FFT magnitude (standard)
        x_fft = torch.fft.fft(x, dim=-1).abs()  # [B, C, T]

        # Hilbert envelope via FFT
        x_fft_complex = torch.fft.fft(x, dim=-1)  # [B, C, T]

        # Zero negative frequencies (analytic signal in freq domain)
        n = T
        # Create mask: keep DC, keep positive freqs, zero negative freqs
        mask = torch.zeros(T, device=x.device)
        mask[0] = 1.0  # DC
        mask[1:(n + 1) // 2] = 2.0  # positive freqs (double to preserve energy)
        if n % 2 == 0:
            mask[n // 2] = 1.0  # Nyquist
        mask = mask.view(1, 1, T)

        x_analytic_fft = x_fft_complex * mask
        x_analytic = torch.fft.ifft(x_analytic_fft, dim=-1)  # [B, C, T]

        # Envelope = |analytic signal|
        envelope = x_analytic.abs()  # [B, C, T]

        # FFT of envelope (envelope spectrum)
        envelope_fft = torch.fft.fft(envelope, dim=-1).abs()  # [B, C, T]

        # Fuse
        alpha_clamped = torch.sigmoid(self.alpha)
        out = alpha_clamped * x_fft + (1 - alpha_clamped) * envelope_fft

        return out

    def get_2d_map(self, x):
        """返回二维时频图 [n_freq, n_time] — 包络的STFT"""
        B, C, T = x.shape
        x_fft_complex = torch.fft.fft(x, dim=-1)
        n = T
        mask = torch.zeros(n, device=x.device)
        mask[0] = 1.0
        mask[1:(n + 1) // 2] = 2.0
        if n % 2 == 0:
            mask[n // 2] = 1.0
        mask = mask.view(1, 1, n)
        x_analytic = torch.fft.ifft(x_fft_complex * mask, dim=-1)
        envelope = x_analytic.abs()
        x_flat = envelope.reshape(B * C, T)
        stft_out = torch.stft(x_flat, n_fft=64, hop_length=16,
                              win_length=64,
                              window=torch.hann_window(64, device=x.device),
                              return_complex=True, pad_mode='reflect')
        return stft_out.abs().squeeze(0)


# ==============================================================================
# Factory function: select frequency transform
# ==============================================================================
def get_freq_transform(transform_type='fft', **kwargs):
    """
    Get a frequency transform module.

    Args:
        transform_type: one of
            'fft'         - standard FFT (torch.fft.fft)
            'multiscale'  - MultiScaleFFT
            'cwt_approx'  - CWT_Approx (learnable wavelet filter bank)
            'stft_pool'   - STFT_Pool
            'envelope'    - EnvelopeFFT (Hilbert envelope + FFT fusion)

    Returns:
        transform: callable or nn.Module
    """
    if transform_type == 'fft' or transform_type is None:
        return None  # use raw torch.fft.fft().abs()
    elif transform_type == 'multiscale':
        return MultiScaleFFT(**kwargs)
    elif transform_type == 'cwt_approx':
        return CWT_Approx(**kwargs)
    elif transform_type == 'stft_pool':
        return STFT_Pool(**kwargs)
    elif transform_type == 'envelope':
        return EnvelopeFFT(**kwargs)
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")


# ==============================================================================
# Short test
# ==============================================================================
if __name__ == '__main__':
    print("=== Testing freq_transforms ===")
    device = 'cpu'
    B, C, T = 4, 1, 178
    x = torch.randn(B, C, T)

    # Test each transform
    transforms = {
        'MultiScaleFFT': MultiScaleFFT(n_samples=T),
        'CWT_Approx': CWT_Approx(n_samples=T, n_scales=16),
        'STFT_Pool': STFT_Pool(n_samples=T),
        'EnvelopeFFT': EnvelopeFFT(n_samples=T),
    }

    for name, transform in transforms.items():
        out = transform(x)
        grad_test = out.sum().backward()
        print(f"  {name:15s}: input {list(x.shape)} -> output {list(out.shape)} | grad OK")

    print("\n=== All transforms produce [B, C, T] output with valid gradients ===")
