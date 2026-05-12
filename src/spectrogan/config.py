import torch
from dataclasses import dataclass, asdict

@dataclass
class Config:
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training
    train_splits: tuple[int]  = (1,2,3,4)
    test_split: int          = 5
    batch_size: int          = 64
    epochs: int              = 6000
    steps_per_epoch: int     = 100
    
    # Spectrogram
    window_size_secs: float  = 2
    sample_rate: int         = 16000
    spec_freq_dim: int       = 120
    spec_time_dim: int       = 240
    
    # GAN
    nz: int                  = 128
    lr_g: float              = 2e-4
    lr_dw: float             = 1e-4
    lr_ds: float             = 1e-4
    lam_d: float             = 0.5 #waveform to spec discriminator loss ratio
    lam_g: float             = 0.5 #waveform to spec generator loss ratio
    k: int                   = 1
    noise_strength: float    = 0.05
    betas: tuple[float]      = (0.5, 0.999)

    def to_dict(self):
        return asdict(self)
