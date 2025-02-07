from .gaussian_diffusion import GaussianDiffusion
from .models.diffusionMLP import DiffusionMLP
from .models.diffusionMLP_condCNN import DiffusionMLPConditionalCNN
from .models.diffusionCNN import DiffusionCNN

def load_diffusion_model(model_name, **kwargs):
    if model_name == 'diffusionMLP':
        return DiffusionMLP(**kwargs)
    elif model_name == 'diffusionMLP_condCNN':
        return DiffusionMLPConditionalCNN(**kwargs)
    elif model_name == 'diffusionCNN':
        return DiffusionCNN(**kwargs)
    else:
        raise ValueError(f"Invalid model name: {model_name}")