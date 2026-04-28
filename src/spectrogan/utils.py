import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.style.use("dark_background")

def get_specs(samps, extractor):
    specs = extractor(samps)['input_values']
    specs = torch.log(specs + 1e-8)
    return (specs - specs.mean()) / (specs.std() + 1e-5)

def label_real(size, device):
    return (0.9 + 0.1 * torch.rand(size, 1)).to(device)

def label_fake(size, device):
    return (0.0 + 0.1 * torch.rand(size, 1)).to(device)

def create_noise(sample_size, nz, device):
    return torch.randn(sample_size, nz).to(device)


def discriminator_loss(discriminator, data_real, data_fake):
    output_real = discriminator(data_real)
    loss_real = torch.mean(F.relu(1. - output_real))

    output_fake = discriminator(data_fake)
    loss_fake = torch.mean(F.relu(1. + output_fake))

    return loss_real + loss_fake


# plot examples each epoch
def plot_generated_vs_real(gen_specs, real_specs, n=4):
    gen_specs = gen_specs[:n].detach().cpu().numpy()
    real_specs = real_specs[:n].detach().cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(12, 3))

    for i in range(n):
        # Generated (top row)
        axes[0, i].imshow(gen_specs[i], aspect="auto", origin="lower")
        axes[0, i].axis("off")
        
        axes[1, i].imshow(real_specs[i], aspect="auto", origin="lower")
        axes[1, i].axis("off")

    plt.figtext(-0.01, 0.75, "Gen", va='center', rotation='vertical')
    plt.figtext(-0.01, 0.25, "Real", va='center', rotation='vertical')

    plt.tight_layout()
    plt.show()