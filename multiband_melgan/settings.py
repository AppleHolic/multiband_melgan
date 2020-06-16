SAMPLE_RATE: int = 22050  # sample rate of target wave
WIN_LENGTH: int = 1024  # STFT window length
HOP_LENGTH: int = 256  # STFT hop length
HOP_STRIDE: int = WIN_LENGTH // HOP_LENGTH  # frames per window
SPEC_SIZE: int = WIN_LENGTH // 2 + 1  # spectrogram bands
MEL_SIZE: int = 80  # mel-spectrogram bands
MEL_MIN: int = 80  # mel minimum freq.
MEL_MAX: int = 7600  # mel maximum freq.
