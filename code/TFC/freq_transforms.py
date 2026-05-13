
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


# ==============================================================================
# 2. CWT_Approx: 可学习的近似连续小波变换
# ==============================================================================
class CWT_Approx(nn.Module):
    """
    Approximates Continuous Wavelet Transform using a learnable filter bank
    of Morlet-like wavelets in the frequency domain.

    This is FULLY DIFFERENTIABLE and produces a 1D frequency representation
    by pooling over scales.

    Shape: [B, C, T] -> [B, C, T]

    Reference: SSqueezepy's cwt() core logic, translated to PyTorch.
    """
    def __init__(self, n_samples=178, n_scales=32, wavelet='morlet',
                 mu_init=6.0, sigma_init=1.0):
        super().__init__()
        self.n_samples = n_samples
        self.n_scales = n_scales
        self.wavelet_type = wavelet

        # Learnable scales (log-spaced)
        log_scales = torch.linspace(np.log(1), np.log(n_samples // 2), n_scales)
        self.scales = nn.Parameter(log_scales)  # log-scales, learnable

        # Learnable frequency-domain wavelet parameters (Morlet-like)
        # mu controls center frequency; sigma controls bandwidth
        self.mu = nn.Parameter(torch.tensor(mu_init))
        self.sigma = nn.Parameter(torch.tensor(sigma_init))
        self.scale_factors = nn.Parameter(torch.ones(n_scales))
        self.eps = 1e-8

    def _morlet_fd(self, freq, scale, mu, sigma):
        """
        Frequency-domain Morlet-like wavelet.

        psi(ω) = exp(-0.5 * σ² * (ω - μ/scale)²)

        Args:
            freq: [T] or [1, T]  frequency bins (normalized, 0 to pi)
            scale: scalar
            mu: center frequency parameter
            sigma: bandwidth parameter
        Returns:
            psi: same shape as freq
        """
        center = mu / (scale + self.eps)
        # Gaussian centered at mu/scale
        psi = torch.exp(-0.5 * (sigma ** 2) * ((freq - center) ** 2))
        return psi

    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        Returns:
            out: [B, C, T]  magnitude representation pooled over scales
        """
        B, C, T = x.shape

        # Real FFT along time dimension
        x_fft = torch.fft.rfft(x, dim=-1)  # [B, C, T//2+1]
        n_freq = x_fft.shape[-1]

        # Frequency axis (normalized)
        freq = torch.linspace(0, np.pi, n_freq, device=x.device)  # [n_freq]

        scales = torch.exp(self.scales)  # [n_scales], positive

        # Collect wavelet responses at each scale
        scale_responses = []
        for i in range(self.n_scales):
            scale = scales[i]
            sf = self.scale_factors[i]

            # Frequency-domain wavelet
            psi = self._morlet_fd(freq, scale, self.mu, self.sigma)  # [n_freq]

            # Multiply with signal spectrum
            filtered_fft = x_fft * psi * sf  # [B, C, n_freq]

            # Inverse FFT back to time domain -> get complex CWT row
            filtered_time = torch.fft.irfft(filtered_fft, n=T, dim=-1)  # [B, C, T]

            # Take magnitude
            mag = filtered_time.abs()  # [B, C, T]
            scale_responses.append(mag)

        # Stack: [B, C, n_scales, T] -> pool over scales
        scale_stack = torch.stack(scale_responses, dim=2)  # [B, C, n_scales, T]

        # Max-pool over scales (captures most responsive scale at each time point)
        out = scale_stack.max(dim=2)[0]  # [B, C, T]

        return out


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
