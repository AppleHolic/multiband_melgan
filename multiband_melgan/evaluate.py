import fire
import numpy as np
import torch
from torchaudio.transforms import Resample
from multiband_melgan.train_mb import match_dim
from pytorch_sound.utils.calculate import norm_mel
from tqdm import tqdm
from multiband_melgan.dataset import get_datasets
from pesq import pesq
from pytorch_sound import settings
from pytorch_sound.models import build_model
from pytorch_sound.models.transforms import LogMelSpectrogram, PQMF
from pytorch_sound.utils.commons import get_loadable_checkpoint


def load_model(model_name: str, pretrained_path: str) -> torch.nn.Module:
    print('Load model ...')
    model = build_model(model_name).cuda()
    chk = torch.load(pretrained_path)['generator']
    model.load_state_dict(get_loadable_checkpoint(chk))
    model.eval()
    return model


def main(meta_dir: str, pretrained_path: str, model_name: str = 'generator_mb'):
    # load model
    gen = load_model(model_name, pretrained_path).cuda()

    print(gen)
    print(f'Numb. Parameters : {sum(p.numel() for p in gen.parameters() if p.requires_grad)}')

    # make mel func
    mel_func = LogMelSpectrogram(
        settings.SAMPLE_RATE, settings.MEL_SIZE, settings.WIN_LENGTH, settings.WIN_LENGTH, settings.HOP_LENGTH,
        float(settings.MIN_DB), float(settings.MAX_DB), float(settings.MEL_MIN), float(settings.MEL_MAX)
    ).cuda()

    pqmf_func = PQMF().cuda()

    # get datasets
    _, valid_loader = get_datasets(
        meta_dir, batch_size=1, num_workers=1, crop_length=0, random_seed=1234
    )

    resample_func = Resample(22050, 16000).cuda()

    # score
    score_list = []

    for wav, _ in tqdm(valid_loader):
        wav = wav.cuda()

        # to mel
        mel = norm_mel(mel_func(wav))

        with torch.no_grad():
            pred_subbands = gen(mel)
            pred = pqmf_func.synthesis(pred_subbands)
        pred, wav = match_dim(pred, wav)

        # resample
        pred = resample_func(pred)
        wav = resample_func(wav)

        # to cpu
        wav = wav.cpu().numpy().squeeze()
        pred = pred.detach().cpu().numpy().squeeze()

        # resample
        item_score = pesq(16000, wav, pred.clip(-1., 1.), 'wb')
        score_list.append(item_score)

    print(
        f'mean : {np.mean(score_list)}, std : {np.std(score_list)}, '
        f'min : {np.min(score_list)}, max : {np.max(score_list)}'
    )


if __name__ == '__main__':
    fire.Fire(main)
