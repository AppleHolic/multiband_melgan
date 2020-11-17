import fire
import json
import torch
import os
import numpy as np
import torch.nn.functional as F
from typing import Tuple

from multiband_melgan import models
from multiband_melgan.dataset import get_datasets
from multiband_melgan.utils import repeat
from multiband_melgan import settings
from multiband_melgan.modules import MelSpectrogramOther as LogMelSpectrogram
from pytorch_sound.models.transforms import STFT
from pytorch_sound.utils.plots import imshow_to_buf
from pytorch_sound.models import build_model
from pytorch_sound.models.transforms import PQMF
from pytorch_sound.utils.commons import log
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR


def main(meta_dir: str, save_dir: str,
         save_prefix: str, pretrained_path: str = '',
         batch_size: int = 32, num_workers: int = 8,
         lr: float = 1e-4, betas: Tuple[float, float] = (0.5, 0.9), weight_decay: float = 0.0,
         pretrain_step: int = 200000,
         max_step: int = 1000000, save_interval: int = 10000,
         log_scala_interval: int = 20, log_heavy_interval: int = 1000,
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
        meta_dir, batch_size=batch_size, num_workers=num_workers, crop_length=settings.SAMPLE_RATE, random_seed=seed
    )

    # repeat
    train_loader = repeat(train_loader)

    # build mel function
    mel_func, stft_funcs_for_loss = build_stft_functions()

    # build pqmf
    pqmf_func = PQMF().cuda()

    # prepare logging
    writer, model_dir = prepare_logging(save_dir, save_prefix)

    # Training Saving Attributes
    best_loss = np.finfo(np.float32).max
    initial_step = 0

    # load model
    if pretrained_path:
        log(f'Pretrained path is given : {pretrained_path} . Loading...')
        chk = torch.load(pretrained_path)
        gen_chk, dis_chk = chk['generator'], chk['discriminator']
        gen_opt_chk, dis_opt_chk = chk['gen_opt'], chk['dis_opt']
        initial_step = int(chk['step'])
        l = chk['loss']

        mb_generator.load_state_dict(gen_chk)
        discriminator.load_state_dict(dis_chk)
        mb_opt.load_state_dict(gen_opt_chk)
        dis_opt.load_state_dict(dis_opt_chk)
        if 'dis_scheduler' in chk:
            dis_scheduler_chk = chk['dis_scheduler']
            gen_scheduler_chk = chk['gen_scheduler']
            mb_scheduler.load_state_dict(gen_scheduler_chk)
            dis_scheduler.load_state_dict(dis_scheduler_chk)

        mb_opt._step_count = initial_step
        mb_scheduler._step_count = initial_step
        dis_opt._step_count = initial_step - pretrain_step
        dis_scheduler._step_count = initial_step - pretrain_step

        mb_scheduler.step(initial_step)
        dis_scheduler.step(initial_step - pretrain_step)
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
        mel = mel_func(wav)

        # pqmf
        target_subbands = pqmf_func.analysis(wav.unsqueeze(1))  # N, SUBBAND, T

        # forward
        pred_subbands = mb_generator(mel)
        pred_subbands, _ = match_dim(pred_subbands, target_subbands)

        # pqmf synthesis
        pred = pqmf_func.synthesis(pred_subbands)
        pred, wav = match_dim(pred, wav)

        # get multi-resolution stft loss   eq 9)
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

        if step % save_interval == 0 and step > 0:
            #
            # Validation Step !
            #
            valid_loss = 0.
            valid_mb_loss, valid_fb_loss = 0., 0.
            count = 0
            mb_generator.eval()

            for idx, (wav, _) in enumerate(valid_loader):
                # setup data
                wav = wav.cuda()
                mel = mel_func(wav)

                with torch.no_grad():
                    # pqmf
                    target_subbands = pqmf_func.analysis(wav.unsqueeze(1))  # N, SUBBAND, T

                    # forward
                    pred_subbands = mb_generator(mel)
                    pred_subbands, _ = match_dim(pred_subbands, target_subbands)

                    # pqmf synthesis
                    pred = pqmf_func.synthesis(pred_subbands)
                    pred, wav = match_dim(pred, wav)

                    # get stft loss
                    loss, mb_loss, fb_loss = get_stft_loss(
                        pred, wav, pred_subbands, target_subbands, stft_funcs_for_loss
                    )

                valid_loss += loss.item()
                valid_mb_loss += mb_loss.item()
                valid_fb_loss += fb_loss.item()
                count = idx

            valid_loss /= (count + 1)
            valid_mb_loss /= (count + 1)
            valid_fb_loss /= (count + 1)
            mb_generator.train()

            # log validation
            # log writer
            pred_audio = pred[0, 0]
            target_audio = wav[0]
            writer.add_scalar('valid/pretrain_loss', valid_loss, global_step=step)
            writer.add_scalar('valid/mb_loss', valid_mb_loss, global_step=step)
            writer.add_scalar('valid/fb_loss', valid_fb_loss, global_step=step)
            writer.add_audio('valid/pred_audio', pred_audio, sample_rate=settings.SAMPLE_RATE, global_step=step)
            writer.add_audio('valid/target_audio', target_audio, sample_rate=settings.SAMPLE_RATE, global_step=step)

            # console
            log(f'---- Valid loss: {valid_loss} / mb_loss: {valid_mb_loss} / fb_loss: {valid_fb_loss} ----')

            #
            # save checkpoint
            #
            is_best = valid_loss < best_loss
            if is_best:
                best_loss = valid_loss
            save_checkpoint(
                mb_generator, discriminator,
                mb_opt, dis_opt,
                mb_scheduler, dis_scheduler,
                model_dir, step, valid_loss, is_best=is_best
            )

    #
    # Train GAN
    #
    dis_block_layers = 6
    lambda_gen = 2.5
    best_loss = np.finfo(np.float32).max

    for step in range(max(pretrain_step, initial_step), max_step + 1):

        # data
        wav, _ = next(train_loader)
        wav = wav.cuda()

        # to mel
        mel = mel_func(wav)

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
            pred_mel = mel_func(pred.squeeze(1).detach())
            mel_err = F.l1_loss(mel, pred_mel).item()

        # if terminate_step > step:
        d_fake_det = discriminator(pred.detach())
        d_real = discriminator(wav.unsqueeze(1))

        # calculate discriminator losses  eq 1)
        loss_D = 0

        for idx in range(dis_block_layers - 1, len(d_fake_det), dis_block_layers):
            loss_D += torch.mean((d_fake_det[idx] - 1) ** 2)

        for idx in range(dis_block_layers - 1, len(d_real), dis_block_layers):
            loss_D += torch.mean(d_real[idx] ** 2)

        # train
        discriminator.zero_grad()
        loss_D.backward()
        dis_opt.step()
        dis_scheduler.step()

        #
        # Train Generator
        #
        d_fake = discriminator(pred)

        # calc generator loss   eq 8)
        loss_G = 0
        for idx in range(dis_block_layers - 1, len(d_fake), dis_block_layers):
            loss_G += ((d_fake[idx] - 1) ** 2).mean()

        loss_G *= lambda_gen

        # get multi-resolution stft loss
        loss_G += get_stft_loss(pred, wav, pred_subbands, target_subbands, stft_funcs_for_loss)[0]
        # loss_G += get_spec_losses(pred, wav, stft_funcs_for_loss)[0]

        mb_generator.zero_grad()
        loss_G.backward()
        mb_opt.step()
        mb_scheduler.step()

        #
        # logging! save!
        #
        if step % log_scala_interval == 0 and step > 0:
            # log writer
            pred_audio = pred[0, 0]
            target_audio = wav[0]
            writer.add_scalar('train/loss_G', loss_G.item(), global_step=step)
            writer.add_scalar('train/loss_D', loss_D.item(), global_step=step)
            writer.add_scalar('train/mel_err', mel_err, global_step=step)
            if step % log_heavy_interval == 0:
                target_mel = imshow_to_buf(mel[0].detach().cpu().numpy())
                pred_mel = imshow_to_buf(mel_func(pred[:1, 0])[0].detach().cpu().numpy())

                writer.add_image('train/target_mel', target_mel, global_step=step)
                writer.add_image('train/pred_mel', pred_mel, global_step=step)
                writer.add_audio('train/pred_audio', pred_audio, sample_rate=settings.SAMPLE_RATE, global_step=step)
                writer.add_audio('train/target_audio', target_audio, sample_rate=settings.SAMPLE_RATE, global_step=step)

            # console
            msg = f'train: step: {step} / loss_G: {loss_G.item()} / loss_D: {loss_D.item()} / ' \
                f' mel_err: {mel_err}'
            log(msg)

        if step % save_interval == 0 and step > 0:
            #
            # Validation Step !
            #
            valid_g_loss, valid_d_loss, valid_mel_loss = 0., 0., 0.
            count = 0
            mb_generator.eval()
            discriminator.eval()

            for idx, (wav, _) in enumerate(valid_loader):
                # setup data
                wav = wav.cuda()
                mel = mel_func(wav)

                with torch.no_grad():
                    # pqmf
                    target_subbands = pqmf_func.analysis(wav.unsqueeze(1))  # N, SUBBAND, T

                    # Discriminator
                    pred_subbands = mb_generator(mel)
                    pred_subbands, _ = match_dim(pred_subbands, target_subbands)

                    # pqmf synthesis
                    pred = pqmf_func.synthesis(pred_subbands)
                    pred, wav = match_dim(pred, wav)

                    # Mel Error
                    pred_mel = mel_func(pred.squeeze(1).detach())
                    mel_err = F.l1_loss(mel, pred_mel).item()

                    #
                    # discriminator part
                    #
                    d_fake_det = discriminator(pred.detach())
                    d_real = discriminator(wav.unsqueeze(1))

                    loss_D = 0

                    for idx in range(dis_block_layers - 1, len(d_fake_det), dis_block_layers):
                        loss_D += torch.mean((d_fake_det[idx] - 1) ** 2)

                    for idx in range(dis_block_layers - 1, len(d_real), dis_block_layers):
                        loss_D += torch.mean(d_real[idx] ** 2)

                    #
                    # generator part
                    #
                    d_fake = discriminator(pred)

                    # calc generator loss
                    loss_G = 0
                    for idx in range(dis_block_layers - 1, len(d_fake), dis_block_layers):
                        loss_G += ((d_fake[idx] - 1) ** 2).mean()

                    loss_G *= lambda_gen

                    # get stft loss
                    stft_loss = get_stft_loss(pred, wav, pred_subbands, target_subbands, stft_funcs_for_loss)[0]
                    loss_G += stft_loss

                valid_d_loss += loss_D.item()
                valid_g_loss += loss_G.item()
                valid_mel_loss += mel_err
                count = idx

            valid_d_loss /= (count + 1)
            valid_g_loss /= (count + 1)
            valid_mel_loss /= (count + 1)

            mb_generator.train()
            discriminator.train()

            # log validation
            # log writer
            pred_audio = pred[0, 0]
            target_audio = wav[0]
            target_mel = imshow_to_buf(mel[0].detach().cpu().numpy())
            pred_mel = imshow_to_buf(mel_func(pred[:1, 0])[0].detach().cpu().numpy())

            writer.add_image('valid/target_mel', target_mel, global_step=step)
            writer.add_image('valid/pred_mel', pred_mel, global_step=step)
            writer.add_scalar('valid/loss_G', valid_g_loss, global_step=step)
            writer.add_scalar('valid/loss_D', valid_d_loss, global_step=step)
            writer.add_scalar('valid/mel_err', valid_mel_loss, global_step=step)
            writer.add_audio('valid/pred_audio', pred_audio, sample_rate=settings.SAMPLE_RATE, global_step=step)
            writer.add_audio('valid/target_audio', target_audio, sample_rate=settings.SAMPLE_RATE, global_step=step)

            # console
            log(
                f'---- loss_G: {valid_g_loss} / loss_D: {valid_d_loss} / mel loss : {valid_mel_loss} ----'
            )

            #
            # save checkpoint
            #
            is_best = valid_g_loss < best_loss
            if is_best:
                best_loss = valid_g_loss
            save_checkpoint(
                mb_generator, discriminator,
                mb_opt, dis_opt,
                mb_scheduler, dis_scheduler,
                model_dir, step, valid_g_loss, is_best=is_best
            )

    log('----- Finish ! -----')


FB_STFT_PARAMS = [
    [1024, 600, 120], [2048, 1200, 240], [512, 240, 50]
]
MB_STFT_PARAMS = [
    [384, 30, 150], [683, 60, 300], [171, 10, 60]
]


def build_stft_functions():
    print('Build Mel Functions ...')
    mel_funcs_for_loss = [
        STFT(
            fft, hop, win
        ).cuda() for fft, win, hop in FB_STFT_PARAMS + MB_STFT_PARAMS
    ]

    mel_func = LogMelSpectrogram(
        settings.SAMPLE_RATE, settings.MEL_SIZE, settings.WIN_LENGTH, settings.WIN_LENGTH, settings.HOP_LENGTH,
        mel_min=float(settings.MEL_MIN), mel_max=float(settings.MEL_MAX)
    ).cuda()
    return mel_func, mel_funcs_for_loss


def get_spec_losses(pred: torch.Tensor, target: torch.Tensor, stft_funcs_for_loss, eps: float = 1e-5):
    loss, sc_loss, mag_loss = 0., 0., 0.

    for stft_idx, stft_func in enumerate(stft_funcs_for_loss):
        p_stft = stft_func.transform(pred.squeeze(1))[0]
        t_stft = stft_func.transform(target)[0]

        N = t_stft.size(1) * t_stft.size(2)
        sc_loss_ = ((t_stft - p_stft).norm(dim=(1, 2)) / t_stft.norm(dim=(1, 2))).mean()
        mag_loss_ = torch.norm(torch.log(t_stft + eps) - torch.log(p_stft + eps), p=1, dim=(1, 2)).mean() / N

        loss += sc_loss_ + mag_loss_
        sc_loss += sc_loss_
        mag_loss += mag_loss_

    return loss / len(stft_funcs_for_loss), sc_loss / len(stft_funcs_for_loss), mag_loss / len(stft_funcs_for_loss)


def get_stft_loss(pred, wav, pred_subband, target_subband, stft_funcs_for_loss):
    fb_stft_funcs = stft_funcs_for_loss[:len(stft_funcs_for_loss) // 2]
    mb_stft_funcs = stft_funcs_for_loss[len(stft_funcs_for_loss) // 2:]

    # calc full bank loss
    fb_loss, fb_sc_loss, fb_mag_loss = get_spec_losses(pred, wav, fb_stft_funcs)

    # calc multi bank losses
    T = pred_subband.size(-1)
    mb_loss, mb_sc_loss, mb_mag_loss = get_spec_losses(pred_subband.view(-1, T), target_subband.view(-1, T), mb_stft_funcs)

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
        gen_scheduler, dis_scheduler,
        out_dir: str, step: int, loss: float, is_best: bool = False
    ):
    # file path
    model_path = os.path.join(out_dir, f'models_{step}.pt')

    chk = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'gen_opt': gen_opt.state_dict(),
        'dis_opt': dis_opt.state_dict(),
        'gen_scheduler': gen_scheduler.state_dict(),
        'dis_scheduler': dis_scheduler.state_dict(),
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
