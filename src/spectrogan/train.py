from tqdm import tqdm
import torch
from .utils import get_specs, create_noise, train_discriminator, train_generator, plot_generated_vs_real

def train_loop(
    generator,
    discriminator,
    train_gen,
    extractor,
    optim_g,
    optim_d,
    config
):
    losses_g, losses_d = [], []

    for epoch in range(config.epochs):
        loss_g, loss_d = 0, 0

        for _ in tqdm(range(config.steps_per_epoch)):
            samps, _ = next(train_gen)
            samps = torch.tensor(samps)

            specs = get_specs(samps, extractor, config.device)
            b_size = len(specs)

            # Train D
            for _ in range(config.k):
                noise = create_noise(b_size, config.nz, config.device)
                fake = generator(noise).detach()
                fake = get_specs(fake, extractor, config.device)

                real = specs.float()

                loss_d += train_discriminator(
                    discriminator, optim_d, real, fake
                )

            # Train G
            fake = generator(create_noise(b_size, config.nz, config.device))
            fake = get_specs(fake, extractor, config.device)

            loss_g += train_generator(discriminator, optim_g, fake)

        losses_g.append(loss_g / config.steps_per_epoch)
        losses_d.append(loss_d / config.steps_per_epoch)

        print(f"Epoch {epoch}: G={losses_g[-1]}, D={losses_d[-1]}")
        plot_generated_vs_real(fake.transpose(1,2)[:4], real.transpose(1,2)[:4])
    return losses_g, losses_d