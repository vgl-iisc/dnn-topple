from pathlib import Path
import numpy as np
import torch

def resolve_run_dir(model_dataset: str, candidates: list[Path]) -> Path:
    for root in candidates:
        run_dir = root / model_dataset
        if run_dir.exists():
            return run_dir
    checked = "\n".join(str(p / model_dataset) for p in candidates)
    raise FileNotFoundError(
        f"Could not find multifield run directory for {model_dataset}. Checked:\n{checked}"
    )


def load_epoch_latents_and_labels(
    run_dir: Path, split: str, epoch: int, tensor_tag: int
) -> tuple[np.ndarray, np.ndarray]:
    tensor_path = run_dir / "Tensors" / split / f"a{tensor_tag}_e{epoch}.pt"
    labels_path = run_dir / "Labels" / split / f"labels_e{epoch}.pt"

    if not tensor_path.exists():
        raise FileNotFoundError(f"Missing latent tensor file: {tensor_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    latents = torch.load(tensor_path, map_location="cpu")
    labels = torch.load(labels_path, map_location="cpu")

    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    latents = np.asarray(latents)
    labels = np.asarray(labels).astype(int)

    if latents.ndim != 2:
        raise ValueError(f"Expected 2D latent matrix, got shape {latents.shape}")
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if latents.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Latent/label length mismatch: {latents.shape[0]} vs {labels.shape[0]}"
        )

    return latents, labels