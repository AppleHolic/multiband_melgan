import fire
import librosa
import pandas as pd
import torch
from multiband_melgan import settings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from multiband_melgan.modules import LogMelSpectrogram


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        item = self.df[idx]
        path = item['audio_filename']
        return path


def run(meta_path: str):
    df = pd.read_json(meta_path)
    mel_func = LogMelSpectrogram(
        settings.SAMPLE_RATE, settings.MEL_SIZE, settings.N_FFT, settings.WIN_LENGTH, settings.HOP_LENGTH,
        float(settings.MEL_MIN), float(settings.MEL_MAX)
    )
    scaler = StandardScaler()

    for item in tqdm(list(df.iterrows())):
        wav, sr = librosa.load(item[1]['audio_filename'])
        mel_tensor = mel_func(torch.FloatTensor(wav).unsqueeze(0), eps=1e-10)[0].transpose(0, 1)
        mel = mel_tensor.numpy()
        scaler.partial_fit(mel)

    print('Mean : {}, STD : {} '.format(scaler.mean_.tolist(), scaler.scale_.tolist()))


if __name__ == '__main__':
    fire.Fire(run)
