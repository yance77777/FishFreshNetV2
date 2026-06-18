from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit


CLASS_NAMES = ["Highly Fresh", "Fresh", "Not Fresh"]


def get_transforms(input_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """Get data transforms for training or evaluation.

    Args:
        input_size: Target image size (square).
        is_train: Whether to apply training augmentations.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# Default transforms for backward compatibility
TRAIN_TRANSFORM = get_transforms(224, is_train=True)
EVAL_TRANSFORM = get_transforms(224, is_train=False)


class FishFreshImageFolder(datasets.ImageFolder):
    """Fish freshness ImageFolder with deterministic class ordering.

    Both MFED and FFE use the same three class folders, but relying on the
    filesystem's alphabetical order would silently remap labels. This dataset
    keeps the class ids fixed across datasets:
    0=Highly Fresh, 1=Fresh, 2=Not Fresh.
    """

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        missing = [c for c in CLASS_NAMES if not (Path(directory) / c).is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing fish freshness class folders: {missing}")
        return CLASS_NAMES, {c: i for i, c in enumerate(CLASS_NAMES)}


class MFEDImageFolder(FishFreshImageFolder):
    """Backward-compatible alias for older training code."""


class FishFreshSubsetDataset(Dataset):
    """Subset dataset with optional per-split decoded-image caching."""

    def __init__(
        self,
        samples: list[tuple[str, int]],
        indices: list[int],
        transform=None,
        cache_images: bool = False,
    ) -> None:
        self.samples = [samples[i] for i in indices]
        self.transform = transform
        self.cached_images = None
        if cache_images:
            self.cached_images = []
            for path, _ in self.samples:
                with Image.open(path) as img:
                    self.cached_images.append(img.convert("RGB").copy())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        if self.cached_images is None:
            with Image.open(path) as img:
                image = img.convert("RGB")
        else:
            image = self.cached_images[index].copy()
        if self.transform is not None:
            image = self.transform(image)
        return image, target


MFEDSubsetDataset = FishFreshSubsetDataset


def create_split_indices(
    dataset_size: int,
    labels: list[int],
    runs: int = 5,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> list[dict[str, list[int]]]:
    """Create stratified train/val/test split indices.

    Uses stratified sampling to ensure each class is proportionally
    represented in every split. Each run uses a different random seed.

    Args:
        dataset_size: Total number of samples.
        labels: List of integer labels for each sample.
        runs: Number of independent splits.
        seed: Base random seed.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation (rest is test).

    Returns:
        List of dicts, each with 'train', 'val', 'test' index lists.
    """
    labels = np.array(labels)
    split_indices = []

    for run_index in range(runs):
        rng = seed + run_index

        # First split: train+val vs test
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio - val_ratio, random_state=rng)
        train_val_idx, test_idx = next(sss1.split(np.zeros(dataset_size), labels))

        # Second split: train vs val
        train_val_labels = labels[train_val_idx]
        val_fraction = val_ratio / (train_ratio + val_ratio)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=rng)
        train_idx_local, val_idx_local = next(sss2.split(np.zeros(len(train_val_idx)), train_val_labels))

        train_idx = train_val_idx[train_idx_local].tolist()
        val_idx = train_val_idx[val_idx_local].tolist()
        test_idx = test_idx.tolist()

        split_indices.append({"train": train_idx, "val": val_idx, "test": test_idx})

    return split_indices


def create_dataloaders(
    data_dir: Path,
    split_indices: dict[str, list[int]],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    input_size: int = 224,
    prefetch_factor: int = 8,
    cache_images: bool = True,
) -> dict[str, DataLoader]:
    """Create train/val/test dataloaders.

    Args:
        data_dir: Path to dataset root.
        split_indices: Dict with 'train', 'val', 'test' index lists.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory (recommended for GPU).
        input_size: Target image size.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders.
    """
    train_transform = get_transforms(input_size, is_train=True)
    eval_transform = get_transforms(input_size, is_train=False)

    base_dataset = FishFreshImageFolder(str(data_dir))
    train_dataset = FishFreshSubsetDataset(base_dataset.samples, split_indices["train"], train_transform, cache_images)
    val_dataset = FishFreshSubsetDataset(base_dataset.samples, split_indices["val"], eval_transform, cache_images)
    test_dataset = FishFreshSubsetDataset(base_dataset.samples, split_indices["test"], eval_transform, cache_images)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return {
        "train": DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    }


def get_labels_from_dataset(data_dir: Path) -> list[int]:
    """Extract integer labels from the MFED dataset directory.

    Args:
        data_dir: Path to dataset root.

    Returns:
        List of integer labels.
    """
    dataset = FishFreshImageFolder(str(data_dir))
    return [label for _, label in dataset.samples]


def create_full_dataloader(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    input_size: int = 224,
    prefetch_factor: int = 8,
    cache_images: bool = True,
) -> DataLoader:
    """Create an evaluation dataloader over the full dataset."""
    eval_transform = get_transforms(input_size, is_train=False)
    base_dataset = FishFreshImageFolder(str(data_dir))
    indices = list(range(len(base_dataset.samples)))
    dataset = FishFreshSubsetDataset(base_dataset.samples, indices, eval_transform, cache_images)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def create_subset_dataloader(
    data_dir: Path,
    indices: list[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    input_size: int = 224,
    prefetch_factor: int = 8,
    cache_images: bool = True,
) -> DataLoader:
    """Create an evaluation dataloader over selected sample indices."""
    eval_transform = get_transforms(input_size, is_train=False)
    base_dataset = FishFreshImageFolder(str(data_dir))
    dataset = FishFreshSubsetDataset(base_dataset.samples, indices, eval_transform, cache_images)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def describe_dataset(data_dir: Path) -> dict[str, int]:
    """Return class counts for a fish freshness dataset."""
    labels = get_labels_from_dataset(data_dir)
    return {name: labels.count(i) for i, name in enumerate(CLASS_NAMES)}
