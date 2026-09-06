#!/usr/bin/env python3
"""YOLOv10 surgical-tool detection for already sampled RGB video frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


COMPONENT_SUFFIXES = ("tip", "clevis", "shaft")
USED_COMPONENTS = frozenset(("tip", "clevis"))


@dataclass(frozen=True)
class ToolDetection:
    class_name: str
    instrument_name: str
    component: str
    confidence: float
    bbox_xyxy_normalized: tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameToolDetections:
    sample_number: int
    source_frame_index: int
    timestamp_seconds: float
    detections: tuple[ToolDetection, ...]


@dataclass(frozen=True)
class InstrumentStatistics:
    instrument_name: str
    frame_count: int
    max_confidence: float


@dataclass(frozen=True)
class ToolDetectionResult:
    frames: tuple[FrameToolDetections, ...]
    instrument_frame_count_threshold: int = 0

    def __post_init__(self) -> None:
        """Discard unused components and instruments seen too few times."""
        if (
            isinstance(self.instrument_frame_count_threshold, bool)
            or not isinstance(self.instrument_frame_count_threshold, int)
            or self.instrument_frame_count_threshold < 0
        ):
            raise ValueError("instrument frame count threshold must be a non-negative integer")

        component_filtered_frames = tuple(
            FrameToolDetections(
                sample_number=frame.sample_number,
                source_frame_index=frame.source_frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                detections=tuple(
                    detection
                    for detection in frame.detections
                    if detection.component in USED_COMPONENTS
                ),
            )
            for frame in self.frames
        )

        source_frames_by_instrument: dict[str, set[int]] = {}
        for frame in component_filtered_frames:
            for detection in frame.detections:
                source_frames_by_instrument.setdefault(detection.instrument_name, set()).add(
                    frame.source_frame_index
                )
        retained_instruments = {
            instrument_name
            for instrument_name, source_frames in source_frames_by_instrument.items()
            if len(source_frames) > self.instrument_frame_count_threshold
        }
        filtered_frames = tuple(
            FrameToolDetections(
                sample_number=frame.sample_number,
                source_frame_index=frame.source_frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                detections=tuple(
                    detection
                    for detection in frame.detections
                    if detection.instrument_name in retained_instruments
                ),
            )
            for frame in component_filtered_frames
        )
        object.__setattr__(self, "frames", filtered_frames)

    @property
    def detection_count(self) -> int:
        return sum(len(frame.detections) for frame in self.frames)

    @property
    def instrument_names(self) -> tuple[str, ...]:
        """Return unique instrument names, merging retained tip/clevis detections."""
        return tuple(statistic.instrument_name for statistic in self.instrument_statistics)

    @property
    def instrument_statistics(self) -> tuple[InstrumentStatistics, ...]:
        """Summarize each instrument across frames, merging its detected components."""
        source_frames_by_instrument: dict[str, set[int]] = {}
        max_confidences: dict[str, float] = {}
        for frame in self.frames:
            for detection in frame.detections:
                source_frames_by_instrument.setdefault(detection.instrument_name, set()).add(
                    frame.source_frame_index
                )
                max_confidences[detection.instrument_name] = max(
                    max_confidences.get(detection.instrument_name, 0.0),
                    detection.confidence,
                )
        return tuple(
            InstrumentStatistics(
                instrument_name=instrument_name,
                frame_count=len(source_frames),
                max_confidence=max_confidences[instrument_name],
            )
            for instrument_name, source_frames in source_frames_by_instrument.items()
        )

    def to_prompt_context(self) -> str:
        # A per-frame, per-detection listing (one sentence with a bounding box
        # for every detection in every sampled frame) used to be built here.
        # Its length scales with num_frames * simultaneous instruments/
        # components in the scene, which is unbounded and highly
        # video-dependent -- for busy multi-instrument clips it grew to
        # 5,000+ LLM tokens, on top of the ~3,000-token video itself. That
        # doubled the total sequence length for a handful of real cases and
        # was the actual cause of CUDA OOMs on Grand Challenge's T4 (the
        # video/resolution settings were sized without accounting for this
        # text). Summarizing to one line per distinct instrument bounds the
        # length by the number of instrument classes (small and fixed by
        # surgvu.yaml) instead of by frame count or detections-per-frame, at
        # the cost of dropping precise per-frame bounding boxes.
        total_frames = len(self.frames)
        statistics = sorted(
            self.instrument_statistics, key=lambda item: item.frame_count, reverse=True
        )
        if not statistics:
            return (
                f"Surgical tool detection across {total_frames} sampled video frames: "
                "no surgical tool detected."
            )
        lines = [f"Surgical tool detection across {total_frames} sampled video frames:"]
        for statistic in statistics:
            lines.append(
                f"- {statistic.instrument_name}: detected in {statistic.frame_count} of "
                f"{total_frames} sampled frames (max confidence {statistic.max_confidence:.2f})."
            )
        return "\n".join(lines)


# The checkpoint's "needle_driver" class is trained on what the official
# SurgVU/da Vinci taxonomy calls the "large needle driver" -- there is no
# separate "regular-size" needle driver in this instrument set (confirmed
# against 1_prepare/segment_video_surgicalsam.py and 2_vlm/qa_templates.json,
# which both use "large_needle_driver"). Several real evaluation questions
# name it that way explicitly, so present it under its official name here;
# a generic "needle driver" question still matches fine since "large needle
# driver" contains "needle driver" as a substring.
INSTRUMENT_DISPLAY_NAME_OVERRIDES = {"needle driver": "large needle driver"}


def _split_class_name(class_name: str) -> tuple[str, str]:
    for component in COMPONENT_SUFFIXES:
        suffix = f"_{component}"
        if class_name.endswith(suffix):
            name = class_name[: -len(suffix)].replace("_", " ")
            return INSTRUMENT_DISPLAY_NAME_OVERRIDES.get(name, name), component
    name = class_name.replace("_", " ")
    return INSTRUMENT_DISPLAY_NAME_OVERRIDES.get(name, name), "whole"


def load_class_names(path: Path) -> dict[int, str]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("Tool detection requires PyYAML") from error

    if not path.is_file():
        raise FileNotFoundError(f"Tool class configuration does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    names = data.get("names") if isinstance(data, dict) else None
    if not isinstance(names, (dict, list)):
        raise ValueError(f"Tool class configuration has no valid names section: {path}")
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {int(index): str(name) for index, name in names.items()}


class ToolDetector:
    """Detect tool parts and expose human-readable instruments and positions."""

    def __init__(
        self,
        model_path: Path,
        class_config_path: Path,
        *,
        device: str = "auto",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Tool detection requires ultralytics") from error

        if not model_path.is_file():
            raise FileNotFoundError(f"Tool detector checkpoint does not exist: {model_path}")
        self.class_names = load_class_names(class_config_path)
        self.model = YOLO(str(model_path))
        model_names = {int(index): str(name) for index, name in self.model.names.items()}
        if model_names != self.class_names:
            raise ValueError("Tool detector checkpoint classes do not match surgvu.yaml")
        self.device = None if device == "auto" else device

    def predict(
        self,
        frames: Any,
        *,
        frame_indices: list[int],
        source_fps: float,
        confidence: float = 0.2,
        batch_size: int = 8,
        instrument_frame_count_threshold: int = 0,
    ) -> ToolDetectionResult:
        if getattr(frames, "ndim", None) != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape (frames, height, width, 3)")
        if len(frames) != len(frame_indices):
            raise ValueError("frames and frame_indices must have the same length")
        if source_fps <= 0:
            raise ValueError("source_fps must be positive")
        if batch_size <= 0:
            raise ValueError("tool detector batch_size must be positive")
        if not 0 <= confidence <= 1:
            raise ValueError("tool detector confidence must be between 0 and 1")
        if (
            isinstance(instrument_frame_count_threshold, bool)
            or not isinstance(instrument_frame_count_threshold, int)
            or instrument_frame_count_threshold < 0
        ):
            raise ValueError("instrument frame count threshold must be a non-negative integer")

        all_results = []
        for start in range(0, len(frames), batch_size):
            batch = [frame for frame in frames[start : start + batch_size]]
            all_results.extend(
                self.model.predict(
                    source=batch,
                    conf=confidence,
                    device=self.device,
                    verbose=False,
                )
            )

        frame_results = []
        for sample_index, (frame, source_index, result) in enumerate(
            zip(frames, frame_indices, all_results), start=1
        ):
            height, width = frame.shape[:2]
            detections = []
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = self.class_names[class_id]
                instrument_name, component = _split_class_name(class_name)
                left, top, right, bottom = box.xyxy[0].detach().cpu().tolist()
                normalized = (
                    min(max(float(left) / width, 0.0), 1.0),
                    min(max(float(top) / height, 0.0), 1.0),
                    min(max(float(right) / width, 0.0), 1.0),
                    min(max(float(bottom) / height, 0.0), 1.0),
                )
                detections.append(
                    ToolDetection(
                        class_name=class_name,
                        instrument_name=instrument_name,
                        component=component,
                        confidence=float(box.conf[0].item()),
                        bbox_xyxy_normalized=normalized,
                    )
                )
            detections.sort(key=lambda item: item.confidence, reverse=True)
            frame_results.append(
                FrameToolDetections(
                    sample_number=sample_index,
                    source_frame_index=int(source_index),
                    timestamp_seconds=float(source_index) / source_fps,
                    detections=tuple(detections),
                )
            )
        return ToolDetectionResult(
            frames=tuple(frame_results),
            instrument_frame_count_threshold=instrument_frame_count_threshold,
        )
