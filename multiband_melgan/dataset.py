import numpy as np
import librosa
import os
from pytorch_sound.data.meta.ljspeech import LJSpeechMeta
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class AudioDataset(Dataset):

    def __init__(self, meta_frame, crop_length: int, seed: int = 1234):
        self.meta_frame = meta_frame
        self.column_name = 'audio_filename'
        self.crop_length = crop_length
        self.seed = seed
        np.random.seed(seed)

    def __getitem__(self, idx):
        # get selected file path
        file_path = self.meta_frame.iloc[idx][self.column_name]

        # load audio
        wav, _ = librosa.load(file_path, sr=None)

        # random crop
        rand_start = np.random.randint((len(wav) - self.crop_length))
        cropped_wav = wav[rand_start:rand_start + self.crop_length]

        # make mask
        wav_mask = np.ones_like(cropped_wav)

        return wav, wav_mask


def get_datasets(meta_dir: str, batch_size: int, num_workers: int, crop_length: int, random_seed: int
                 ) -> Tuple[DataLoader, DataLoader]:
    assert os.path.isdir(meta_dir), '{} is not valid directory path!'

    train_file, valid_file = LJSpeechMeta.frame_file_names[1:]

    # load meta file
    train_meta = LJSpeechMeta(os.path.join(meta_dir, train_file))
    valid_meta = LJSpeechMeta(os.path.join(meta_dir, valid_file))

    # create dataset
    train_dataset = AudioDataset(train_meta, crop_length=crop_length, seed=random_seed)
    valid_dataset = AudioDataset(valid_meta, crop_length=crop_length, seed=random_seed)

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader
