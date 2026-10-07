"""Model definitions, training utilities, and CIFAR-100 data loading."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

import torch
import torchvision.transforms.v2 as transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sized

CONV1_CH = 64
CONV2_CH = 64

LIN1_CH = 384
LIN2_CH = 192

IMG_WH = 24


class LeNet5(nn.Module):
    """LeNet-style convolutional network for CIFAR images.

    Parameters
    ----------
    num_classes : int, default=100
        Number of output classes.
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()

        self._feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=CONV1_CH, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=0.2),  # noqa: ERA001
            nn.Conv2d(in_channels=CONV1_CH, out_channels=CONV2_CH, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=0.2),  # noqa: ERA001
        )

        self._classifier = nn.Sequential(
            nn.Linear(in_features=CONV2_CH * (IMG_WH // (2 * 2) - 2) ** 2, out_features=LIN1_CH),
            nn.ReLU(),
            nn.Linear(in_features=LIN1_CH, out_features=LIN2_CH),
            nn.ReLU(),
            nn.Linear(in_features=LIN2_CH, out_features=num_classes),
        )

        self._softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, *, apply_softmax: bool = False) -> torch.Tensor:
        """Compute class scores for a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Image batch with shape ``(batch, channels, height, width)``.
        apply_softmax : bool, default=False
            Whether to normalize class scores into probabilities.

        Returns
        -------
        torch.Tensor
            Class scores, or probabilities when ``apply_softmax`` is true.
        """
        x = self._feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self._classifier(x)

        if apply_softmax:
            x = self._softmax(x)

        return x

    def save(self, path: str, other: dict | None = None) -> None:
        """
        Save the model parameters and class metadata to a checkpoint.

        Parameters
        ----------
        path : str
            Destination checkpoint path.
        other : dict, optional
            Additional values to store in the checkpoint.
        """
        if other is None:
            other = {}
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({"model_state_dict": self.state_dict(), "model_class": self.__class__.__name__, **other}, path)

    @classmethod
    def load(cls, path: str) -> LeNet5:
        """
        Load model parameters from a checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint created by :meth:`save`.

        Returns
        -------
        LeNet5
            A model with the checkpoint parameters loaded.
        """
        checkpoint = torch.load(path)

        model = cls()

        model.load_state_dict(checkpoint["model_state_dict"])

        return model


def average_model_params(out: nn.Module, inp: list[nn.Module]) -> None:
    """
    Average corresponding parameters and store them in the output model.

    Parameters
    ----------
    out : nn.Module
        Model that receives the averaged parameters.
    inp : list[nn.Module]
        Models whose parameters are averaged. All models must have matching
        parameter counts and shapes.
    """
    if not inp:
        msg = "The list of input models is empty."
        raise ValueError(msg)

    for i in range(1, len(inp)):
        if len(list(inp[0].parameters())) != len(list(inp[i].parameters())):
            msg = "All models must have the same structure."
            raise ValueError(msg)

    for out_param, *inp_params in zip(out.parameters(), *[m.parameters() for m in inp], strict=False):
        if not all(out_param.shape == inp_param.shape for inp_param in inp_params):
            msg = "Mismatch in parameter shapes among models."
            raise ValueError(msg)

        avg_param = torch.mean(torch.stack([inp_param.data for inp_param in inp_params]), dim=0)

        out_param.data.copy_(avg_param)


def set_model_params(out: list[nn.Module], inp: nn.Module) -> None:
    """
    Copy the input model parameters to each model in the output list.

    Parameters
    ----------
    out : list[nn.Module]
        Models to update. Each must have the same parameter structure as
        ``inp``.
    inp : nn.Module
        Model whose parameters are copied.
    """
    if not out:
        msg = "The list of models is empty."
        raise ValueError(msg)

    for model in out:
        if len(list(model.parameters())) != len(list(inp.parameters())):
            msg = "Mismatch in the structure of models and the input model."
            raise ValueError(msg)

    inp_params = list(inp.parameters())

    for model in out:
        for model_param, inp_param in zip(model.parameters(), inp_params, strict=False):
            if model_param.shape != inp_param.shape:
                msg = "Mismatch in parameter shapes between models and the input model."
                raise ValueError(msg)

            model_param.data.copy_(inp_param.data)


def evaluate_model(
    model: nn.Module,
    eval_data: Dataset,
    device: torch.device,
    *,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Evaluate the model on a dataset.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    eval_data : Dataset
        Dataset yielding image and label pairs.
    device : torch.device
        Device used for model inference.
    verbose : bool, default=True
        Whether to log the resulting loss and accuracy.

    Returns
    -------
    tuple[float, float]
        Mean cross-entropy loss and classification accuracy.
    """
    eval_size = len(cast("Sized", eval_data))
    eval_data_loader = DataLoader(eval_data, batch_size=2048, shuffle=False, pin_memory=True)

    model.eval()
    correct = 0
    running_loss = 0

    with torch.no_grad():
        for image_batch, label_batch in eval_data_loader:
            images = image_batch.to(device, non_blocking=True)
            labels = label_batch.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels, reduction="sum")

            # Accumulate loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    av_loss = running_loss / eval_size
    accuracy = correct / eval_size

    if verbose:
        logger.info("Validation loss/acc: %.3f/%.2f%%", av_loss, accuracy * 100)

    return av_loss, accuracy


T_co = TypeVar("T_co", covariant=True)


class Instantiator[T_co]:
    """Store a constructor and keyword arguments for deferred instantiation.

    Parameters
    ----------
    var_class : type
        Class or callable to instantiate.
    var_kwargs : dict, optional
        Keyword arguments passed to the constructor.
    """

    def __init__(self, var_class: type[T_co], var_kwargs: dict[str, Any] | None = None) -> None:
        self.var_class = var_class
        self.kwargs = var_kwargs or {}

    def instantiate(self, first_arg: object, **kwargs: object) -> T_co:
        """Instantiate the configured class.

        Parameters
        ----------
        first_arg : object
            First positional constructor argument.
        **kwargs : object
            Extra keyword arguments, which override stored values with the
            same names.

        Returns
        -------
        T_co
            The constructed object.
        """
        factory = cast("Callable[..., T_co]", self.var_class)
        return factory(first_arg, **self.kwargs, **kwargs)


class Trainer:
    """Manage training and evaluation for one model and optimizer."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_I: Instantiator[Any],
        device: torch.device,
        scheduler_I: Instantiator[Any] | None = None,
        *,
        verbose: bool = True,
    ) -> None:
        """
        Initialize a trainer for a model and its optimizer.

        Parameters
        ----------
        model : nn.Module
            Model to train.
        optimizer_I : Instantiator
            Factory that creates the model optimizer.
        device : torch.device
            Device used for training and evaluation.
        scheduler_I : Instantiator, optional
            Factory that creates a learning-rate scheduler. The scheduler steps
            after each optimizer update.
        verbose : bool, default=True
            Whether to log training progress.
        """
        self.model = model.to(device)

        self.optimizer = optimizer_I.instantiate(model.parameters())

        if scheduler_I is not None:
            self.scheduler = scheduler_I.instantiate(self.optimizer)
        else:
            self.scheduler = None

        self.device = device
        self.verbose = verbose

    def train_model(  # noqa: C901
        self,
        train_loader: Iterator[DataLoader],
        n_steps: int,
        eval_data: Dataset | None = None,
    ) -> tuple[float, float] | tuple[float, float, float, float]:
        """
        Train the model for a fixed number of steps.

        Parameters
        ----------
        train_loader : Iterator
            Iterator yielding batches of images and labels.
        n_steps : int
            Maximum number of optimization steps.
        eval_data : Dataset, optional
            Dataset evaluated after each training step.

        Returns
        -------
        tuple[float, float] or tuple[float, float, float, float]
            Mean training loss and accuracy. When ``eval_data`` is supplied,
            also returns mean evaluation loss and accuracy.
        """
        self.model.train()

        train_loss = torch.empty(n_steps)
        train_acc = torch.empty(n_steps)
        eval_loss = torch.empty(n_steps)
        eval_acc = torch.empty(n_steps)

        start_time = time.monotonic()

        for step in range(n_steps):
            try:
                images, labels = next(train_loader)
            except StopIteration:
                if eval_data is not None:
                    return (
                        train_loss.mean().item(),
                        train_acc.mean().item(),
                        eval_loss.mean().item(),
                        eval_acc.mean().item(),
                    )
                return train_loss.mean().item(), train_acc.mean().item()

            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

            batch_size = labels.size(0)

            # Forward pass
            outputs = self.model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels, reduction="mean")

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()

            if torch.isnan(loss):
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item()

                logger.warning("Gradient norm: %s", total_norm)
                msg = "Loss is NaN. Stopping..."
                raise ValueError(msg)

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()

            train_loss[step] = loss.item()
            train_acc[step] = correct / batch_size

            if eval_data is not None:
                e_loss, e_acc = self.evaluate(eval_data)
                eval_loss[step] = e_loss
                eval_acc[step] = e_acc
                self.model.train()

            if self.verbose and step == 0:
                logger.info("Training progress: [0/%s]", n_steps)
            if self.verbose:
                logger.info("Training step %s: Learning rate %.6f", step + 1, self.optimizer.param_groups[0]["lr"])
                logger.info(
                    "Training progress: [%s/%s], %.2fs per step (batch size: %s)",
                    step + 1,
                    n_steps,
                    (time.monotonic() - start_time) / (step + 1),
                    batch_size,
                )

        if eval_data is not None:
            return train_loss.mean().item(), train_acc.mean().item(), eval_loss.mean().item(), eval_acc.mean().item()

        return train_loss.mean().item(), train_acc.mean().item()

    def evaluate(self, eval_data: Dataset | None = None) -> tuple[float, float]:
        """
        Evaluate the model on a dataset.

        Parameters
        ----------
        eval_data : Dataset, optional
            Dataset yielding image and label pairs. If omitted, raises
            ``ValueError``.

        Returns
        -------
        tuple[float, float]
            Mean cross-entropy loss and classification accuracy.
        """
        if eval_data is None:
            msg = "Pass an evaluation dataset to evaluate()."
            raise ValueError(msg)

        return evaluate_model(self.model, eval_data, self.device, verbose=self.verbose)


@overload
def load_data(
    data_dir: str | Path | None = None,
    random_seed: int | None = None,
    *,
    test_data: Literal[True],
) -> Dataset: ...


@overload
def load_data(
    data_dir: str | Path | None = None,
    random_seed: int | None = None,
    *,
    test_data: Literal[False] = False,
) -> tuple[Dataset, Dataset]: ...


def load_data(
    data_dir: str | Path | None = None,
    random_seed: int | None = None,
    *,
    test_data: bool = False,
) -> tuple[Dataset, Dataset] | Dataset:
    """Load CIFAR-100 training and validation splits or the test set.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Dataset root directory. Defaults to ``data`` in the current working
        directory.
    random_seed : int, optional
        Seed used to create the training and validation split.
    test_data : bool, default=False
        If true, load and return the test dataset.

    Returns
    -------
    tuple[Dataset, Dataset] or Dataset
        The training and validation datasets, or the test dataset when
        ``test_data`` is true.
    """
    if data_dir is None:
        data_dir = Path("data").resolve()

    generator = torch.Generator().manual_seed(random_seed) if random_seed is not None else None

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop((IMG_WH, IMG_WH)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.ToImage(),  # noqa: ERA001
            # transforms.ToDtype(torch.float32, scale=True), # to tensor is faster  # noqa: ERA001
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # noqa: ERA001
            transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.CenterCrop((IMG_WH, IMG_WH)),
            transforms.ToTensor(),
            # transforms.ToImage(),  # noqa: ERA001
            # transforms.ToDtype(torch.float32, scale=True),  # noqa: ERA001
            transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
        ]
    )

    dataset = datasets.CIFAR100(root=str(data_dir), train=not test_data, download=True)

    if test_data:
        dataset.transform = train_transforms
        return dataset

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_dataset = DatasetFromSubset(train_dataset, train_transforms)
    val_dataset = DatasetFromSubset(val_dataset, val_transforms)

    return train_dataset, val_dataset


class DatasetFromSubset(Dataset):
    """Apply an optional transform to items from a dataset subset.

    Parameters
    ----------
    subset : Subset
        Dataset subset to wrap.
    transform : callable, optional
        Transform applied to each item before it is returned.
    """

    def __init__(self, subset: Subset, transform: Callable | None = None) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Return a subset item after applying the configured transform."""
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        """Return the number of items in the wrapped subset."""
        return len(self.subset)
