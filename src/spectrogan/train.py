from tqdm import tqdm
import torch
import os
from .utils import get_specs, create_noise, discriminator_loss, plot_generated_vs_real, save_checkpoint, load_checkpoint

def train_loop(
    generator,
    waveform_discriminator,
    spectrogram_discriminator,
    train_gen,
    extractor,
    optim_g,
    optim_dw,
    optim_ds,
    config,
    save_file=None
):
    losses_g, losses_d = [], []
    start_epoch = 0

    # Load checkpoint if available
    if save_file is not None and os.path.isfile(save_file):
        print(f"Loading checkpoint from {save_file}")
        start_epoch, losses_g, losses_d = load_checkpoint(
            save_file, generator, waveform_discriminator, spectrogram_discriminator, optim_g, optim_dw, optim_ds
        )
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, config.epochs):
        loss_g, loss_d = 0, 0

        for _ in tqdm(range(config.steps_per_epoch)):
            real_waveforms, _ = next(train_gen)
            real_waveforms = torch.tensor(real_waveforms).squeeze().float().to(config.device)

            real_spectrograms = get_specs(real_waveforms, extractor)
            b_size = len(real_spectrograms)

            # Train D
            for _ in range(config.k):
                noise = create_noise(b_size, config.nz, config.device).float()
                fake_waveforms = generator(noise)
                fake_spectrograms = get_specs(fake_waveforms, extractor)

                real_spectrograms = real_spectrograms

                optim_dw.zero_grad()
                optim_ds.zero_grad()

                epoch_loss_dw = discriminator_loss(
                    waveform_discriminator, real_waveforms, fake_waveforms
                )

                epoch_loss_ds = discriminator_loss(
                    spectrogram_discriminator, real_spectrograms, fake_spectrograms
                )

                epoch_loss_d = (config.lam_d * epoch_loss_dw) + epoch_loss_ds

                epoch_loss_d.backward()
                optim_dw.step()
                optim_ds.step()

                loss_d += epoch_loss_d.detach()

            # Train G
            fake_waveforms = generator(create_noise(b_size, config.nz, config.device))
            fake_spectrograms = get_specs(fake_waveforms, extractor)

            optim_g.zero_grad()
            output_w = waveform_discriminator(fake_waveforms)
            output_s = spectrogram_discriminator(fake_spectrograms)
            epoch_loss_g = -(config.lam_g*torch.mean(output_w)) - torch.mean(output_s)
            epoch_loss_g.backward()
            optim_g.step()

            loss_g += epoch_loss_g.detach()

        losses_g.append(loss_g / config.steps_per_epoch)
        losses_d.append(loss_d / config.steps_per_epoch)

        print(f"Epoch {epoch}: G={losses_g[-1]}, D={losses_d[-1]}")
        plot_generated_vs_real(fake_spectrograms.transpose(1,2)[:4], real_spectrograms.transpose(1,2)[:4])

        # Save checkpoint at end of each epoch
        if save_file is not None:
            save_checkpoint(save_file, epoch+1, generator, waveform_discriminator, spectrogram_discriminator, optim_g, optim_dw, optim_ds, losses_g, losses_d)

    return losses_g, losses_d