import os
import torch
from pytorch_sound.models.transforms import STFT, PQMF
from multiband_melgan import settings
from multiband_melgan.modules import LogMelSpectrogram2 as LogMelSpectrogram
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model


DEFAULT_VOC_TAG = 'ljspeech_multi_band_melgan.v2'
DEFAULT_MEL_MEAN = -2.34460057
DEFAULT_MEL_STD = 0.8997583


class Inferencer:
    """
    This class is made to build "mel function" and "vocoder" simply

    Methods
    -------
    encode(wav_tensor: torch.Tensor)
        :arg wav_tensor dimension 1 or 2
        :return log mel spectrum
    decode(mel_tensor: torch.Tensor)
        :arg mel_tensor (N, C, Tm)
        :return predicted wav_tensor (N, 1, Tw)

    Examples::
        inferencer = Inferencer()

        # load audio and make tensor
        wav, sr = librosa.load(audio_path, sr=22050)
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0).cuda()

        # convert to mel
        mel = inferencer.encode(wav_tensor)

        # convert back to wav
        pred_wav = inferencer.decode(mel)
    """

    def __init__(self):
        # make mel converter
        self.mel_func = LogMelSpectrogram(
            settings.SAMPLE_RATE, settings.MEL_SIZE, settings.N_FFT, settings.WIN_LENGTH, settings.HOP_LENGTH,
            float(settings.MEL_MIN), float(settings.MEL_MAX), normalize=False
        ).cuda()

        # PQMF module
        self.pqmf = PQMF().cuda()

        # load model
        self.vocoder = load_model(download_pretrained_model(DEFAULT_VOC_TAG)).cuda().eval()
        self.vocoder.remove_weight_norm()

        self.stft = STFT(settings.WIN_LENGTH, settings.HOP_LENGTH).cuda()

        # denoise - reference https://github.com/NVIDIA/waveglow/blob/master/denoiser.py
        mel_input = torch.zeros((1, 80, 88)).float().cuda()
        with torch.no_grad():
            bias_audio = self.decode(mel_input, is_denoise=False).squeeze(1)
            bias_spec, _ = self.stft.transform(bias_audio)

        self.bias_spec = bias_spec[:, :, 0][:, :, None]

    def encode(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert wav tensor to mel tensor
        :param wav_tensor: wav tensor (N, T)
        :return: mel tensor (N, C, T)
        """
        if len(wav_tensor.size()) == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        assert len(wav_tensor.size()) <= 2, 'The expected dimension of wav is 1 or 2'
        return (self.mel_func(wav_tensor.cuda()) - DEFAULT_MEL_MEAN) / DEFAULT_MEL_STD

    def decode(self, mel_tensor: torch.Tensor, is_denoise: bool = False) -> torch.Tensor:
        """
        Convert mel tensor to wav tensor by using multi-band melgan
        :param mel_tensor: mel tensor (N, C, T)
        :param is_denoise: using denoise function
        :return: wav tensor (N, T)
        """
        # inference generator and pqmf
        with torch.no_grad():
            pred = self.vocoder(mel_tensor)
            pred = self.pqmf.synthesis(pred)

        # denoising
        if is_denoise:
            pred = self.denoise(pred)

        return pred

    def denoise(self, audio: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
        audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised


if __name__ == '__main__':
    import sys
    import librosa
    audio_path = sys.argv[1]

    # make inferencer
    inferencer = Inferencer()

    # load audio and make tensor
    wav, sr = librosa.load(audio_path, sr=22050)
    wav_tensor = torch.FloatTensor(wav).unsqueeze(0).cuda()
    print(f'Load wave file')

    # convert to mel
    mel = inferencer.encode(wav_tensor)
    print(f'Make mel tensor, size: {str(mel.size())}')

    # convert back to wav
    pred_wav = inferencer.decode(mel)
    print(f'Predict wav tensor, size: {str(pred_wav.size())}')
