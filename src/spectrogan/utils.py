import torch
from .config import Config


# Save the entire model and config
def save_model(model, path, config=None):
    """Save the entire model (not just state_dict) and optional config as a .pt or .pth file."""
    save_dict = {'model': model}
    if config is not None:
        save_dict['config'] = config.to_dict()
    torch.save(save_dict, path)


# Load a model and config
def load_model(path, map_location=None):
    """Load a model and config from a .pt or .pth file. Returns (model, config)."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model = checkpoint['model']
    config_dict = checkpoint.get('config', None)
    config = Config(**config_dict) if config_dict is not None else None
    return model, config
