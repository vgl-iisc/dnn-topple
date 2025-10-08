"""Model and transform loader for classification experiments.

Provides standard torchvision model constructors for classification experiments:
- densenet121
- resnet18
- vgg16
- wide_resnet50_2

Each constructor can optionally load a checkpoint and will return a torch.nn.Module
on the requested device.

Usage:
    model = create_model('resnet20', num_classes=10, checkpoint='...')
"""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as tv_models

# ---------------------- Standard torchvision wrappers ----------------------
# Use torchvision's standard implementations rather than custom CIFAR variants.
# Canonical torchvision constructors are provided; create_model accepts several
# legacy aliases for backwards compatibility.


def resnet18(num_classes: int = 10) -> nn.Module:
    """Return torchvision's resnet18"""
    model = tv_models.resnet18()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def wide_resnet50_2(num_classes: int = 10) -> nn.Module:
    """Return torchvision's wide_resnet50_2"""
    model = tv_models.wide_resnet50_2()
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        print("replaced a linear layer")
    else:
        # fallback if different naming
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                # can't safely replace arbitrary module; fallback to adding head
                model.fc = nn.Linear(in_features, num_classes)
                print("added a linear layer at 'fc'")
                break
    return model


# ---------------------- DenseNet / VGG helpers ----------------------
def densenet121(num_classes: int = 10) -> nn.Module:
    """Return torchvision's densenet121"""
    model = tv_models.densenet121()
    # adapt classifier
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def vgg16(num_classes: int = 10) -> nn.Module:
    """Return torchvision's vgg16"""
    model = tv_models.vgg16()
    # adapt classifier: replace the last linear layer while preserving the Sequential
    if isinstance(model.classifier, nn.Sequential):
        modules = list(model.classifier.children())
        if len(modules) == 0:
            # fallback: replace with a single linear
            model.classifier = nn.Sequential(nn.Linear(512, num_classes))
        else:
            last = modules[-1]
            # Prefer Linear layers (have int .in_features). Otherwise fall back to 4096.
            in_features = 4096
            if isinstance(last, nn.Linear):
                in_features = last.in_features
                print("replaced a linear layer")
            else:
                if hasattr(last, 'in_features') and isinstance(last.in_features, int):
                    in_features = last.in_features
                    print("using last.in_features")
                elif hasattr(last, 'out_features') and isinstance(last.out_features, int):
                    in_features = last.out_features
                    print("using last.out_features, seems suspect")
            modules[-1] = nn.Linear(in_features, num_classes)
            model.classifier = nn.Sequential(*modules)
    else:
        # ensure classifier becomes a Sequential with a final Linear
        in_features = getattr(model.classifier, 'in_features', 4096)
        print("classifier was not a Sequential, replacing with a single Linear")
        model.classifier = nn.Sequential(nn.Linear(in_features, num_classes))
    return model


# ---------------------- Factory + utils ----------------------
def _load_checkpoint(model: nn.Module, checkpoint_path: Optional[str], device: torch.device):
    if checkpoint_path is None:
        return model
    state = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in state:
        sd = state['state_dict']
    elif 'model_state_dict' in state:
        sd = state['model_state_dict']
    else:
        sd = state
    # try to adapt common 'module.' prefix
    keys = list(sd.keys())
    if any(k.startswith('module.') for k in keys):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f'Loaded checkpoint {checkpoint_path}. Missing/Unexpected keys: {msg}')
    return model


def create_model(
    arch: str,
    num_classes: int = 10,
    checkpoint: Optional[str] = None,
    device: str = 'cpu',
) -> nn.Module:
    """Create a model and CIFAR transforms.

    arch: one of 'densenet121', 'resnet18', 'vgg16', 'wide_resnet50_2'
    Returns: (model_on_device, (train_transform, test_transform))
    """
    dev = torch.device(device)
    arch = arch.lower()
    # Accept only canonical torchvision names (no aliases)
    if arch == 'densenet121':
        model = densenet121(num_classes=num_classes)
    elif arch == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif arch == 'vgg16':
        model = vgg16(num_classes=num_classes)
    elif arch == 'wide_resnet50_2':
        model = wide_resnet50_2(num_classes=num_classes)
    else:
        raise ValueError(f'Unknown architecture: {arch}. Use canonical names: densenet121, resnet18, vgg16, wide_resnet50_2')

    model = _load_checkpoint(model, checkpoint, dev)
    model.to(dev)

    return model


if __name__ == '__main__':
    print('Creating example models:')
    for name in ('resnet18', 'densenet121', 'vgg16', 'wide_resnet50_2'):
        try:
            model = create_model(name, num_classes=10)
            print(f'  - {name}: OK')
        except Exception as e:
            print(f'  - {name}: FAILED -> {e}')
