import torch
import matplotlib.pyplot as plt
plt.style.use("dark_background")

def get_specs(samps, extractor):
    specs = extractor(samps)['input_values']
    specs = torch.log(specs + 1e-8)
    return (specs - specs.mean()) / (specs.std() + 1e-5)

# plot examples each epoch
def plot_generated_vs_real(gen_specs, real_specs, n=4):
    gen_specs = gen_specs[:n].detach().cpu().numpy()
    real_specs = real_specs[:n].detach().cpu().numpy()

    _, axes = plt.subplots(2, n, figsize=(12, 3))

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

# just plot generated data without real data for a cleaner look
def plot_generated_data(specs):
    """Plot only generated spectrograms (or similar 2D data)."""
    specs = specs.detach().cpu().numpy()
    n = len(specs)
    _, axes = plt.subplots(1, n, figsize=(3 * n, 2))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(specs[i], aspect="auto", origin="lower")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()