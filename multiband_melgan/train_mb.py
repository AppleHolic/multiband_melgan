import fire
import json
import torch
import os
import numpy as np
import torch.nn.functional as F
from multiband_melgan import models

from typing import Tuple
from pytorch_sound.models.transforms import LogMelSpectrogram, STFTTorchAudio
from pytorch_sound.utils.calculate import norm_mel
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_sound.models import build_model
from pytorch_sound import settings
from pytorch_sound.models.transforms import PQMF
from pytorch_sound.utils.commons import get_loadable_checkpoint, log
from multiband_melgan.dataset import get_datasets
from multiband_melgan.utils import repeat


def main(meta_dir: str, save_dir: str,
         save_prefix: str, pretrained_path: str = '',
         mb_batch_size: int = 32, num_workers: int = 8,
         lr: float = 1e-4, betas: Tuple[float] = (0.5, 0.9), weight_decay: float = 0.0, pretrain_step: int = 200000,
         max_step: int = 2000000, valid_max_step: int = 100, save_interval: int = 10000,
         log_scala_interval: int = 1, log_heavy_interval: int = 100,
         grad_norm: float = 10.0,
         gamma: float = 0.5, seed: int = 1234):
    #
    # prepare training
    #
    # create model
    mb_generator = build_model('generator_mb').cuda()
    discriminator = build_model('discriminator_base').cuda()

    # Multi-gpu is not required.

    # create optimizers
    mb_opt = torch.optim.Adam(mb_generator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    dis_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    # make scheduler
    mb_scheduler = MultiStepLR(mb_opt, list(range(300000, 900000 + 1, 100000)), gamma=gamma)
    dis_scheduler = MultiStepLR(dis_opt, list(range(100000, 700000 + 1, 100000)), gamma=gamma)

    # get datasets
    train_loader, valid_loader = get_datasets(
        meta_dir, batch_size=mb_batch_size, num_workers=num_workers, crop_length=settings.SAMPLE_RATE, random_seed=seed
    )

    # repeat
    train_loader = repeat(train_loader)
    valid_loader = repeat(valid_loader)

    # build mel function
    mel_func, stft_funcs_for_loss = build_mel_functions()

    # build pqmf
    pqmf_func = PQMF().cuda()

    # prepare logging
    writer, model_dir = prepare_logging(save_dir, save_prefix)

    # Training Saving Attributes
    best_loss = np.finfo(np.float32).max
    best_step = 0
    initial_step = 0

    # load model
    if pretrained_path:
        log(f'Pretrained path is given : {pretrained_path} . Loading...')
        chk = torch.load(pretrained_path)
        gen_chk, dis_chk = chk['generator'], chk['discriminator']
        gen_opt_chk, dis_opt_chk = chk['gen_opt'], chk['dis_opt']
        initial_step = chk['step']
        l = chk['loss']

        mb_generator.load_state_dict(gen_chk)
        discriminator.load_state_dict(dis_chk)
        mb_opt.load_state_dict(gen_opt_chk)
        dis_opt.load_state_dict(dis_opt_chk)
        best_loss = l

    #
    # Training !
    #
    # Pretraining generator
    for step in range(initial_step, pretrain_step):
        # data
        wav, _ = next(train_loader)
        wav = wav.cuda()

        # to mel
        mel = norm_mel(mel_func(wav))

        # pqmf
        target_subbands = pqmf_func.analysis(wav.unsqueeze(1))  # N, SUBBAND, T

        # forward
        pred_subbands = mb_generator(mel)
        pred_subbands, _ = match_dim(pred_subbands, target_subbands)

        # pqmf synthesis
        pred = pqmf_func.synthesis(pred_subbands)
        pred, wav = match_dim(pred, wav)

        # get stft loss
        loss, mb_loss, fb_loss = get_stft_loss(pred, wav, pred_subbands, target_subbands, stft_funcs_for_loss)

        # backward and update
        loss.backward()
        mb_opt.step()
        mb_scheduler.step()

        mb_opt.zero_grad()
        mb_generator.zero_grad()

        #
        # logging! save!
        #
        if step % log_scala_interval == 0 and step > 0:
            # log writer
            pred_audio = pred[0, 0]
            target_audio = wav[0]
            writer.add_scalar('train/pretrain_loss', loss.item(), global_step=step)
            writer.add_scalar('train/mb_loss', mb_loss.item(), global_step=step)
            writer.add_scalar('train/fb_loss', fb_loss.item(), global_step=step)

            if step % log_heavy_interval == 0:
                writer.add_audio('train/pred_audio', pred_audio, sample_rate=settings.SAMPLE_RATE, global_step=step)
                writer.add_audio('train/target_audio', target_audio, sample_rate=settings.SAMPLE_RATE, global_step=step)

            # console
            msg = f'train: step: {step} / loss: {loss.item()} / mb_loss: {mb_loss.item()} / fb_loss: {fb_loss.item()}'
            log(msg)

        if step % save_interval == 0:
            l = loss.item()
            is_best = l < best_loss
            if is_best:
                best_loss = l
            save_checkpoint(
                mb_generator, discriminator,
                mb_opt, dis_opt,
                model_dir, step, loss.item(), is_best=is_best
            )

    #
    # Train GAN
    #
    dis_block_layers = 6
    dis_numb = 3
    lambda_gen, lambda_feat = 2.5, 10.

    for step in range(max(pretrain_step, initial_step), max_step):
        # zero grad
        mb_opt.zero_grad()
        dis_opt.zero_grad()

        # data
        wav, _ = next(train_loader)
        wav = wav.cuda()

        # to mel
        mel = norm_mel(mel_func(wav))

        # pqmf
        target_subbands = pqmf_func.analysis(wav.unsqueeze(1))  # N, SUBBAND, T

        #
        # Train Discriminator
        #

        # forward
        pred_subbands = mb_generator(mel)
        pred_subbands, _ = match_dim(pred_subbands, target_subbands)

        # pqmf synthesis
        pred = pqmf_func.synthesis(pred_subbands)
        pred, wav = match_dim(pred, wav)

        with torch.no_grad():
            pred_mel = norm_mel(mel_func(pred.squeeze(1).detach()))
            mel_err = F.l1_loss(mel, pred_mel).item()

        d_fake_det = discriminator(pred.detach())
        d_real = discriminator(wav.unsqueeze(1))

        loss_D = 0
        for idx in range(dis_block_layers - 1, len(d_fake_det), dis_block_layers):
            loss_D += torch.mean((d_fake_det[idx] - 1) ** 2)

        for idx in range(dis_block_layers - 1, len(d_real), dis_block_layers):
            loss_D += torch.mean(d_real[idx] ** 2)

        # train
        loss_D.backward()
        dis_opt.step()
        dis_scheduler.step()

        #
        # Train Generator
        #
        d_fake = discriminator(pred)

        # calc generator loss
        loss_G = 0
        for idx in range(dis_block_layers - 1, len(d_fake), dis_block_layers):
            loss_G += ((d_fake[idx] - 1) ** 2)

        loss_G = (lambda_gen * loss_G).mean()

        # get stft loss
        stft_loss = get_stft_loss(pred, wav, pred_subbands, target_subbands, stft_funcs_for_loss)[0]
        loss_G += stft_loss


        #
        # Feature loss
        # calc weight
        feat_weights = 4.0 / (dis_block_layers + 1)
        d_weights = 1.0 / dis_numb
        wt = d_weights * feat_weights
        loss_feat = 0.

        for fake, real in zip(d_fake, d_real):
            loss_feat += wt * (real.detach() - fake).norm(p=1, dim=1).norm(p=1, dim=1).mean()

        final_loss = loss_G + lambda_feat * loss_feat
        final_loss.backward()
        mb_opt.step()
        mb_scheduler.step()
        mb_generator.zero_grad()

        #
        # logging! save!
        #
        if step % log_scala_interval == 0 and step > 0:
            # log writer
            pred_audio = pred[0, 0]
            target_audio = wav[0]
            writer.add_scalar('train/final_loss', final_loss.item(), global_step=step)
            writer.add_scalar('train/loss_G', loss_G.item(), global_step=step)
            writer.add_scalar('train/loss_D', loss_D.item(), global_step=step)
            writer.add_scalar('train/loss_feat', loss_feat.item(), global_step=step)
            writer.add_scalar('train/mel_err', mel_err, global_step=step)
            if step % log_heavy_interval == 0:
                writer.add_audio('train/pred_audio', pred_audio, sample_rate=settings.SAMPLE_RATE, global_step=step)
                writer.add_audio('train/target_audio', target_audio, sample_rate=settings.SAMPLE_RATE, global_step=step)

            # console
            msg = f'train: step: {step} / final_loss: {final_loss.item()} / ' \
                f'loss_G: {loss_G.item()} / loss_D: {loss_D.item()} / ' \
                f'loss_feat: {loss_feat.item()} / mel_err: {mel_err}'
            log(msg)

        if step % save_interval == 0:
            l = final_loss.item()
            is_best = l < best_loss
            if is_best:
                best_loss = l
            save_checkpoint(
                mb_generator, discriminator,
                mb_opt, dis_opt,
                model_dir, step, loss.item(), is_best=is_best
            )

    log('----- Finish ! -----')


def get_multi_resolution_params():
    origin_samplerate = 16000
    target_samplerate = settings.SAMPLE_RATE
    ratio = target_samplerate / origin_samplerate

    params_list = [
        [384, 150, 30], [683, 300, 60], [171, 60, 10]
    ]

    result = [[int(p * ratio) for p in params] for params in params_list]
    return result


def build_mel_functions():
    print('Build Mel Functions ...')
    params_for_loss = get_multi_resolution_params()

    # filter_length: int = 1024, hop_length: int = 512, win_length: int = None, n_fft: int = None,
    # window: str = 'hann'):

    mel_funcs_for_loss = [
        STFTTorchAudio(
            win, hop, win, fft
        ) for fft, win, hop in params_for_loss
    ]

    mel_func = LogMelSpectrogram(
        settings.SAMPLE_RATE, settings.MEL_SIZE, settings.WIN_LENGTH, settings.WIN_LENGTH, settings.HOP_LENGTH,
        float(settings.MIN_DB), float(settings.MAX_DB), float(settings.MEL_MIN), float(settings.MEL_MAX)
    ).cuda()
    return mel_func, mel_funcs_for_loss


def get_spec_losses(pred: torch.Tensor, target: torch.Tensor, stft_funcs_for_loss):
    loss, sc_loss, mag_loss = 0., 0., 0.

    for stft_idx, stft_func in enumerate(stft_funcs_for_loss):
        real, img = stft_func(pred.squeeze(1))
        p_stft = torch.sqrt(real ** 2 + img ** 2 + 1e-5)

        real, img = stft_func(target)
        t_stft = torch.sqrt(real ** 2 + img ** 2 + 1e-5)

        N = target.size(-1)
        sc_loss_ = ((torch.abs(t_stft) - torch.abs(p_stft)).norm(dim=1).norm(dim=1) / t_stft.norm(dim=1).norm(dim=1)).mean()
        mag_loss_ = torch.norm(torch.log(torch.abs(t_stft) + 1e-5) - torch.log(torch.abs(p_stft) + 1e-5), p=1, dim=1).norm(p=1, dim=1).mean() / N

        loss += sc_loss_ + mag_loss_
        sc_loss += sc_loss_
        mag_loss += mag_loss_

    return loss / len(stft_funcs_for_loss), sc_loss / len(stft_funcs_for_loss), mag_loss / len(stft_funcs_for_loss)


def get_stft_loss(pred, wav, pred_subband, target_subband, stft_funcs_for_loss):
    # calc full bank loss
    fb_loss, fb_sc_loss, fb_mag_loss = get_spec_losses(pred, wav, stft_funcs_for_loss)

    # calc multi bank losses
    T = pred_subband.size(-1)
    mb_loss, mb_sc_loss, mb_mag_loss = get_spec_losses(pred_subband.view(-1, T), target_subband.view(-1, T), stft_funcs_for_loss)

    # final loss
    loss = (fb_loss + mb_loss) / 2  # eq. 9
    return loss, mb_loss, fb_loss


def match_dim(pred, wav):
    pad_size = np.abs(pred.size(-1) - wav.size(-1)) // 2
    if pred.size(-1) > wav.size(-1):
        return pred[..., pad_size:-(pad_size + wav.size(-1) % 2)], wav
    elif pred.size(-1) < wav.size(-1):
        return pred, wav[..., pad_size:-(pad_size + wav.size(-1) % 2)]
    else:
        return pred, wav


def setup_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def prepare_logging(save_dir: str, save_prefix: str):
    # prepare directories
    log_dir = os.path.join(save_dir, 'logs', save_prefix)
    model_dir = os.path.join(save_dir, 'models', save_prefix)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # make writer
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
    return writer, model_dir


def save_checkpoint(
        generator, discriminator,
        gen_opt, dis_opt,
        out_dir: str, step: int, loss: float, is_best: bool = False
    ):
    # file path
    model_path = os.path.join(out_dir, f'models_{step}.pt')

    chk = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'gen_opt': gen_opt.state_dict(),
        'dis_opt': dis_opt.state_dict(),
        'step': step,
        'loss': loss
    }

    torch.save(chk, model_path)

    if is_best:
        best_path = os.path.join(out_dir, 'models_best.pt')
        os.system(f'cp {model_path} {best_path}')


def run_config(config_path: str):
    # load config file
    with open(config_path, 'r') as r:
        configs = json.load(r)

    # print configs
    print('--- Configuration ---')
    print(json.dumps(configs, indent=4))

    # eval tuple
    for key in configs.keys():
        if isinstance(configs[key], str):
            if ',' in configs[key] and '(' in configs[key] and ')' in configs[key]:
                configs[key] = eval(configs[key])

    # run main
    main(**configs)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    fire.Fire({
        'run': main,
        'run_config': run_config
    })
