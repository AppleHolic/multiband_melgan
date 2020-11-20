import librosa
import torch
import torch.nn as nn
import numpy as np
from torchaudio.transforms import MelSpectrogram as TorchaudioMelSpectrogram
from pytorch_sound.models.transforms import STFT as CustomSTFT
from multiband_melgan import settings


class STFT(nn.Module):
    """
    Match interface between original one and pytorch official implementation
    """

    def __init__(self, filter_length: int = 1024, hop_length: int = 512, win_length: int = None, n_fft: int = None,
                 window: str = 'hann'):
        super().__init__()
        # original arguments
        self.filter_length = filter_length
        self.hop_length = hop_length
        if win_length:
            self.win_length = win_length
        else:
            self.win_length = self.filter_length
        if window == 'hann':
            self.register_buffer('window', torch.hann_window(self.win_length))
        else:
            raise NotImplemented(f'{window} is not implemented ! Use hann')

        # pytorch official arguments
        if n_fft:
            self.n_fft = n_fft
        else:
            self.n_fft = self.win_length

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            wav, self.n_fft, self.hop_length, self.win_length, self.window, True,
            'reflect', False, True
        )  # (N, C, T, 2)
        real_part, img_part = [x.squeeze(3) for x in stft.chunk(2, 3)]
        return real_part, img_part

    def transform(self, wav: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        :param wav: wave tensor
        :return: (N, Spec Dimension * 2, T) 3 dimensional stft tensor
        """
        real_part, img_part = self.forward(wav)
        return torch.sqrt(real_part ** 2 + img_part ** 2 + eps), torch.atan2(img_part, real_part)

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        # match dimension
        magnitude, phase = magnitude.unsqueeze(3), phase.unsqueeze(3)
        stft = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=3)
        return torch.istft(
            stft, self.n_fft, self.hop_length, self.win_length, self.window
        )


class LogMelSpectrogram(nn.Module):
    """
    Mel spectrogram module with above STFT class
    """

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, win_length: int,
                 hop_length: int, mel_min: float = 0., mel_max: float = None):
        super().__init__()
        self.mel_size = mel_size
        self.melfunc = TorchaudioMelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
                                      hop_length=hop_length, f_min=mel_min, f_max=mel_max, n_mels=mel_size,
                                      window_fn=torch.hann_window)

    def forward(self, wav: torch.tensor, eps: float = 1e-7) -> torch.tensor:
        # apply mel spectrogram
        mel = self.melfunc(wav)

        # to log-space
        return torch.log10(mel + eps)


#
# Mel Spec Module
#
class LogMelSpectrogram2(nn.Module):

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, win_length: int,
                 hop_length: int, mel_min: float = 0., mel_max: float = None,
                 normalize: bool = True):
        super().__init__()
        self.mel_size = mel_size
        self.normalize = normalize

        self.stft = CustomSTFT(filter_length=n_fft, hop_length=hop_length, win_length=win_length)

        # mel filter banks
        mel_filter = librosa.filters.mel(sample_rate, n_fft, mel_size, fmin=mel_min, fmax=mel_max)
        self.register_buffer('mel_filter',
                             torch.tensor(mel_filter, dtype=torch.float))
        if normalize:
            self.register_buffer('mean', torch.FloatTensor(np.array(settings.VCTK_MEL_MEAN)).unsqueeze(0).unsqueeze(2))
            self.register_buffer('std', torch.FloatTensor(np.array(settings.VCTK_MEL_STD)).unsqueeze(0).unsqueeze(2))

    def forward(self, wav: torch.tensor, eps: float = 1e-10) -> torch.tensor:
        mag, phase = self.stft.transform(wav)

        # apply mel filter
        mel = torch.matmul(self.mel_filter, mag)

        # to log-space
        mel = torch.log10(mel.clamp_min(eps))
        if self.normalize:
            mel = (mel - self.mean) / self.std
        return mel
