import fire
import json
import torch
import torch.nn as nn

from typing import Tuple
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_sound.models import build_model

from multiband_melgan.dataset import get_datasets


def main(meta_dir: str, save_dir: str,
         save_prefix: str, pretrained_path: str = '',
         fb_batch_size: int = 48, mb_batch_size: int = 128, num_workers: int = 16,
         lr: float = 1e-4, betas: Tuple[float] = (0.9, 0.99), weight_decay: float = 0.0,
         max_step: int = 2000000, valid_max_step: int = 100, save_interval: int = 10000, log_interval: int = 100,
         grad_clip: float = 0.0, grad_norm: float = 30.0, gamma: float = 0.2,
         sr: int = 22050):

    # create model
    mb_generator = build_model('generator_mb').cuda()
    full_generator = build_model('generator_full').cuda()
    discriminator = build_model('discriminator_base').cuda()

    # Multi-gpu is not required.

    # create optimizers
    mb_opt = torch.optim.Adam(mb_generator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    full_opt = torch.optim.Adam(full_generator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    dis_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    milestones = list(range(100000, 700000 + 1, 100000))

    apply_scheduler = lambda opt: MultiStepLR(opt, milestones, gamma=gamma)
    mb_scheduler = apply_scheduler(mb_opt)
    full_scheduler = apply_scheduler(full_opt)
    dis_scheduler = apply_scheduler(dis_opt)

    # get datasets
    train_mb_loader, valid_mb_loader = get_datasets(
        meta_dir, batch_size=mb_batch_size, num_workers=num_workers
    )

    train_fb_loader, valid_fb_loader = get_datasets(
        meta_dir, batch_size=fb_batch_size, num_workers=num_workers
    )


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
    fire.Fire({
        'run': main,
        'run_config': run_config
    })
