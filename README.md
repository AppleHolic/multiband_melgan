# Multi-Band MelGAN

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fappleholic%2Fmultiband_melgan)](https://hits.seeyoufarm.com)

It's an naive implementation of [Multi-band MelGAN: Faster Waveform Generation for High-Quality
Text-to-Speech](https://arxiv.org/abs/2005.05106).

*Under Developing*

### Goals

- Comparable Quality with other vocoders
- Mobile Inference Example

### TODO

- [x] Make inference code & example.
- [ ] Enhance vocoder quality.
- [ ] Make mobile example.

### Prerequisite

- install [pytorch_sound](https://github.com/appleholic/pytorch_sound)
  - More detail about installation is on repository.
```bash
git clone -b v0.0.4 https://github.com/appleholic/pytorch_sound
cd pytorch_sound
pip install -e .
```

- Preprocess vctk
  - After run it, you can find 'meta' directory in "OUT DIR"
  - [Download Link](http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz) 
```bash
python pytorch_sound/scripts/preprocess.py vctk [VCTK DIR] [OUT DIR] [[Sample Rate: default 22.05k]]
```

- Install multiband melgan
```bash
pip install -e .
```

### Environment

- Machine
  - pytorch 1.5.0
  - rtx titan 1 GPU / ryzen 3900x / 64GB
- Dataset
  - VCTK


### Train

```bash
python multiband_melgan/train_mb.py [META DIR] [SAVE DIR] [SAVE PREFIX] [[other arguments...]]
```

### Example

```python
import torch
import librosa
from multiband_melgan.inferencer import Inferencer

# make inferencer
inf = Inferencer()

# load sample audio
sample_path = ''
wav, sr = librosa.load(sample_path, sr=22050)
wav_tensor = torch.FloatTensor(wav).unsqueeze(0).cuda()

# convert to mel
mel = inf.encode(wav_tensor)  # (N, Cm, Tm)

# convert back to wav
pred_wav = inf.decode(mel, is_denoise=True)  # (N, Tw)
```

### Reference

- [descriptinc/melgan-neurips](https://github.com/descriptinc/melgan-neurips)
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)

### Others

- Evaluation Score
  - PESQ on validation set (VCTK)
    - (mean : 2.5367, std : 0.3372) on multi-band melgan 1000k. 
  
- Model
  - multiband generator (22.05k) : 
    - checkpoint file size : 4.7MB
    - numb. parameters : 1236248
  
- Audio Parameters

> - Sample Rate : 22.05k
> - Window Length & fft : 1024
> - Hop length : 256 
> - Mel dim : 80
> - Mel min/max : 80 / 7600
> - Crop Size (in training) : 1s

### Author

Ilji Choi [\@appleholic](https://github.com/appleholic)
