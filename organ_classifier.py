#!/usr/bin/env python3
"""EfficientNet-V2-S organ classifier adapted from capybara."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ORGAN_NAMES = (
    "abdominal wall",
    "bladder",
    "gallbladder",
    "pelvic lymph nodes",
    "rectum",
    "sigmoid colon",
    "unspecified",
    "uterine horn",
)


@dataclass(frozen=True)
class OrganPrediction:
    organ_name: str
    frame_organs: list[str]
    frame_confidences: list[float]


class _EfficientNetOrganModel:
    """Keep the module names identical to the training checkpoint."""

    def __new__(cls):
        import torch.nn as nn
        import torchvision

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torchvision.models.efficientnet_v2_s(weights=None)
                in_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(in_features, len(ORGAN_NAMES))

            def forward(self, images):
                return self.model(images)

        return Model()


class OrganClassifier:
    """Classify sampled RGB frames and aggregate them with majority voting."""

    def __init__(self, checkpoint_path: Path, *, device: str = "auto") -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Organ classification requires torch and torchvision") from error

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Organ classifier checkpoint does not exist: {checkpoint_path}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested for organ classification but is unavailable: {device}")

        self._torch = torch
        self.device = torch.device(device)
        self.model = _EfficientNetOrganModel()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()

    def _preprocess(self, frames):
        torch = self._torch
        try:
            from torchvision.transforms.functional import gaussian_blur
        except ImportError as error:
            raise RuntimeError("Organ classification requires torchvision") from error

        if getattr(frames, "ndim", None) != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape (frames, height, width, 3)")

        images = torch.as_tensor(frames).permute(0, 3, 1, 2).float().div_(255.0)
        if images.shape[-1] > 384:
            images = images[..., 192:-192]
        bottom = min(45, images.shape[-2])
        if bottom:
            blurred = gaussian_blur(images, kernel_size=[51, 51])
            images[..., -bottom:, :] = blurred[..., -bottom:, :]
        images = torch.nn.functional.interpolate(
            images, size=(512, 512), mode="bicubic", align_corners=False
        )
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (images - mean) / std

    def predict(self, frames, *, batch_size: int = 8) -> OrganPrediction:
        if batch_size <= 0:
            raise ValueError("organ classifier batch_size must be positive")
        probabilities = []
        with self._torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                images = self._preprocess(frames[start : start + batch_size])
                logits = self.model(images.to(self.device))
                probabilities.append(logits.softmax(dim=1).cpu())
        probabilities = self._torch.cat(probabilities)
        confidences, class_ids = probabilities.max(dim=1)
        counts = self._torch.bincount(class_ids, minlength=len(ORGAN_NAMES))
        video_class_id = int(counts.argmax().item())
        return OrganPrediction(
            organ_name=ORGAN_NAMES[video_class_id],
            frame_organs=[ORGAN_NAMES[index] for index in class_ids.tolist()],
            frame_confidences=confidences.tolist(),
        )
