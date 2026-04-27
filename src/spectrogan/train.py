from tqdm import tqdm
import torch
from .utils import get_specs, create_noise, train_spectrogram_discriminator, train_waveform_discriminator, train_generator, plot_generated_vs_real

def train_loop(
    generator,
    waveform_discriminator,
    spectrogram_discriminator,
    train_gen,
    extractor,
    optim_g,
    optim_dw,
    optim_ds,
    config
):
    losses_g, losses_d = [], []

    for epoch in range(config.epochs):
        loss_g, loss_d = 0, 0

        for _ in tqdm(range(config.steps_per_epoch)):
            samps, _ = next(train_gen)
            samps = torch.tensor(samps)

            real_spectrograms = get_specs(samps, extractor, config.device)
            b_size = len(real_spectrograms)

            # Train D
            for _ in range(config.k):
                noise = create_noise(b_size, config.nz, config.device)
                fake_waveforms = generator(noise).detach()
                fake_spectrograms = get_specs(fake_waveforms, extractor, config.device)

                real_spectrograms = real_spectrograms.float()

                loss_dw = train_waveform_discriminator(
                    waveform_discriminator, optim_dw, real_spectrograms, fake_waveforms
                )

                loss_ds = train_spectrogram_discriminator(
                    spectrogram_discriminator, optim_ds, real_spectrograms, fake_spectrograms
                )

                loss_d += config.lam * loss_dw + loss_ds

            # Train G
            fake_waveforms = generator(create_noise(b_size, config.nz, config.device))
            fake_spectrograms = get_specs(fake_waveforms, extractor, config.device)

            loss_g += train_generator(waveform_discriminator, spectrogram_discriminator, optim_g, fake_waveforms, fake_spectrograms, config.lam)

        losses_g.append(loss_g / config.steps_per_epoch)
        losses_d.append(loss_d / config.steps_per_epoch)

        print(f"Epoch {epoch}: G={losses_g[-1]}, D={losses_d[-1]}")
        plot_generated_vs_real(fake_spectrograms.transpose(1,2)[:4], real_spectrograms.transpose(1,2)[:4])
    return losses_g, losses_d