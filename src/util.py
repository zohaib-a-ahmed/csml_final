import torch
from pathlib import Path
from .definitions import ConvolutionalNN, LinearExample, DenseNetModel, ResNetModel, DenseNetRawModel, ResNetRawModel, VGG16Model, VGG16RawModel

DIR = Path(__file__).resolve()
MODEL_FACTORY = {
    "cnn": ConvolutionalNN,
    "example": LinearExample,
    "dense": DenseNetModel,
    "res": ResNetModel,
    "dense_raw": DenseNetRawModel,
    "res_raw": ResNetRawModel,
    "vgg": VGG16Model,
    "vgg_raw": VGG16RawModel
}

def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            print(e)
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    model_size = calculate_model_size_mb(model)
    print(f"Model saved to {output_path}. Size: {model_size} mb")

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


if __name__ == "__main__":
    pass