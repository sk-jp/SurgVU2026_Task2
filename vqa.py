#!/usr/bin/env python3
"""Run video question answering with a local Qwen3.5 vision-language model."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# A local directory (pre-downloaded checkpoint) is loaded fully offline. A
# bare Hugging Face Hub repo id such as this default is downloaded (and
# cached under ~/.cache/huggingface) automatically on first use -- see
# VideoQAModel.__init__. Point --model-path / model_path at a local directory
# instead to use a checkpoint you downloaded yourself, e.g. with
# `huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./Qwen3.5-4B`.
DEFAULT_MODEL_PATH = Path("Qwen/Qwen3.5-4B")
DEFAULT_PROMPT_CONFIG = Path(__file__).with_name("prompt_config.json")
DEFAULT_ORGAN_MODEL_PATH = (
    Path(__file__).with_name("organs_classifier")
    / "efficientnet_v2_s_v3"
    / "epoch=29_test_loss=0.04914_f1_avg=0.98064.ckpt"
)
DEFAULT_TOOL_MODEL_PATH = Path(__file__).with_name("tool_detection") / "best.pt"
DEFAULT_TOOL_CLASS_CONFIG = Path(__file__).with_name("tool_detection") / "surgvu.yaml"
# None defers to rtdetr_tool_detector.RTDetrToolDetector's own default, which is
# resolved relative to wherever it actually finds the category1 checkout (see
# rtdetr_tool_detector._resolve_backend_location) rather than to vqa.py's own
# location -- the two can differ, e.g. 4_docker's copy of this file. Imported lazily
# (see main()) so vqa.py works without a category1 checkout when
# --tool-detector-backend stays at its "yolo" default.
DEFAULT_TOOL_ENSEMBLE_CONFIG = None


DEFAULT_ARGUMENTS: dict[str, Any] = {
    "config": None,
    "video": None,
    "question": None,
    "question_file": None,
    "model_path": DEFAULT_MODEL_PATH,
    "prompt_config": DEFAULT_PROMPT_CONFIG,
    "few_shot_file": None,
    "context": "",
    "context_file": None,
    "disable_organ_classifier": False,
    "organ_model_path": DEFAULT_ORGAN_MODEL_PATH,
    "organ_device": "auto",
    "organ_batch_size": 8,
    "disable_tool_detector": False,
    "tool_detector_backend": "yolo",
    "tool_model_path": DEFAULT_TOOL_MODEL_PATH,
    "tool_class_config": DEFAULT_TOOL_CLASS_CONFIG,
    "tool_ensemble_config": DEFAULT_TOOL_ENSEMBLE_CONFIG,
    "tool_disable_clevis_correction": False,
    "tool_max_detections_per_image": 3,
    "tool_duplicate_iou_threshold": 0.5,
    "rtdetr_category1_root": None,
    "tool_device": "auto",
    "tool_batch_size": 8,
    "tool_confidence": 0.2,
    "tool_frame_count_threshold": 1,
    "num_frames": 32,
    "fps": None,
    "max_pixels": 7_000_000,
    "max_new_tokens": 256,
    "temperature": 0.0,
    "enable_thinking": False,
    "device_map": "auto",
    "dtype": "auto",
    "quantization": "none",
    "output": None,
}


@dataclass(frozen=True)
class PromptConfig:
    system_prompt: str
    user_prompt_template: str
    few_shots: list[dict[str, Any]]

    @classmethod
    def from_json(cls, path: Path) -> "PromptConfig":
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        required = {"system_prompt", "user_prompt_template"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Missing prompt config keys: {', '.join(sorted(missing))}")
        if "{question}" not in data["user_prompt_template"]:
            raise ValueError("user_prompt_template must contain {question}")
        return cls(
            system_prompt=str(data["system_prompt"]),
            user_prompt_template=str(data["user_prompt_template"]),
            few_shots=list(data.get("few_shots", [])),
        )


def _user_content(video: str | None, text: str) -> list[dict[str, str]]:
    content: list[dict[str, str]] = []
    if video:
        content.append({"type": "video", "path": video})
    content.append({"type": "text", "text": text})
    return content


def build_messages(
    *,
    config: PromptConfig,
    video_path: Path,
    question: str,
    additional_context: str = "",
    extra_few_shots: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build Qwen chat messages while keeping prompt details data-driven."""
    format_values = {
        "question": question.strip(),
        "additional_context": additional_context.strip(),
    }
    try:
        current_prompt = config.user_prompt_template.format(**format_values)
    except KeyError as error:
        raise ValueError(f"Unknown placeholder in user_prompt_template: {error}") from error

    messages: list[dict[str, Any]] = []
    if config.system_prompt.strip():
        messages.append({"role": "system", "content": config.system_prompt.strip()})

    few_shots = [*config.few_shots, *(extra_few_shots or [])]
    for index, shot in enumerate(few_shots, start=1):
        if "question" not in shot or "answer" not in shot:
            raise ValueError(f"few-shot #{index} must contain question and answer")
        shot_context = str(shot.get("additional_context", "")).strip()
        shot_text = config.user_prompt_template.format(
            question=str(shot["question"]).strip(),
            additional_context=shot_context,
        )
        shot_video = shot.get("video")
        messages.append(
            {"role": "user", "content": _user_content(str(shot_video) if shot_video else None, shot_text)}
        )
        messages.append({"role": "assistant", "content": str(shot["answer"]).strip()})

    messages.append(
        {"role": "user", "content": _user_content(str(video_path.resolve()), current_prompt)}
    )
    return messages


def load_few_shots(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("few-shot file must contain a JSON array")
    return data


def decode_video(video_path: str, *, num_frames: int | None, fps: float | None):
    """Decode only selected RGB frames and preserve their original timestamps."""
    try:
        import numpy as np
        from decord import VideoReader, cpu
    except ImportError as error:
        raise RuntimeError("Video decoding requires decord and numpy") from error

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    reader = VideoReader(str(path), ctx=cpu(0))
    total = len(reader)
    if total == 0:
        raise ValueError(f"Video contains no decodable frames: {path}")
    source_fps = float(reader.get_avg_fps())
    if source_fps <= 0:
        raise ValueError(f"Could not determine frame rate: {path}")

    if fps is not None:
        step = max(source_fps / fps, 1.0)
        indices = np.arange(0, total, step).astype(np.int64)
    else:
        requested = min(num_frames or total, total)
        indices = np.linspace(0, total - 1, requested, dtype=np.int64)

    # Qwen's temporal patch size is 2. Duplicate the final frame when necessary.
    if len(indices) % 2:
        indices = np.append(indices, indices[-1])
    frames = reader.get_batch(indices).asnumpy()
    metadata = {
        "total_num_frames": total,
        "fps": source_fps,
        "duration": total / source_fps,
        "frames_indices": indices.tolist(),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "video_backend": "decord",
    }
    return frames, metadata


def materialize_videos(
    messages: list[dict[str, Any]],
    *,
    num_frames: int | None,
    fps: float | None,
    video_cache: dict[str, tuple[Any, dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace video paths with sampled frames and matching timestamp metadata."""
    result = copy.deepcopy(messages)
    metadata: list[dict[str, Any]] = []
    for message in result:
        if not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if block.get("type") != "video":
                continue
            key = next((name for name in ("path", "video", "url") if name in block), None)
            if key is None or not isinstance(block[key], str):
                continue
            resolved_path = str(Path(block[key]).resolve())
            cached = (video_cache or {}).get(resolved_path)
            if cached is None:
                frames, video_metadata = decode_video(block[key], num_frames=num_frames, fps=fps)
            else:
                frames, video_metadata = cached
            block.clear()
            block.update({"type": "video", "video": frames})
            metadata.append(video_metadata)
    return result, metadata


class VideoQAModel:
    def __init__(
        self, model_path: Path, *, device_map: str, dtype: str, quantization: str = "none"
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from error

        # A local directory is loaded fully offline. Anything else -- a bare
        # Hugging Face Hub repo id such as the default "Qwen/Qwen3.5-4B" -- is
        # resolved by letting transformers download (and cache under
        # ~/.cache/huggingface) the weights itself on first use.
        local_files_only = model_path.is_dir()
        model_source = model_path if local_files_only else str(model_path)

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_source, local_files_only=local_files_only, trust_remote_code=False
        )

        # A checkpoint produced by diagnostics/prequantize.py already has its
        # weights packed as NF4 tensors, with bitsandbytes' own metadata
        # recorded in config.json ("quantization_config"). Re-quantizing such
        # a checkpoint on every boot would be redundant (it reads the full
        # fp16/bf16-sized tensors again just to immediately requantize them)
        # and is what makes cold starts on Grand Challenge slow. Detect this
        # case and let from_pretrained load the packed tensors directly.
        already_quantized = False
        config_path = model_path / "config.json"
        if quantization != "none" and local_files_only and config_path.is_file():
            try:
                with open(config_path) as f:
                    already_quantized = "quantization_config" in json.load(f)
            except (OSError, ValueError):
                already_quantized = False

        quantization_config = None
        if quantization != "none" and not already_quantized:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as error:
                raise RuntimeError(
                    "Install dependencies with: pip install -r requirements.txt "
                    "(quantization requires bitsandbytes)"
                ) from error
            compute_dtype = torch.bfloat16 if dtype in ("auto", "bfloat16") else dtype_map[dtype]
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"Unknown quantization mode: {quantization}")

        model_kwargs: dict[str, Any] = dict(
            local_files_only=local_files_only,
            trust_remote_code=False,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        if quantization_config is not None:
            # bitsandbytes owns the weight dtype/placement once quantized;
            # passing an explicit dtype alongside quantization_config is
            # rejected by transformers.
            model_kwargs["quantization_config"] = quantization_config
        elif not already_quantized:
            model_kwargs["dtype"] = dtype_map[dtype]
        # else: pre-quantized checkpoint — no quantization_config and no
        # dtype override; bitsandbytes metadata baked into the checkpoint
        # drives both.

        self.model = AutoModelForImageTextToText.from_pretrained(model_source, **model_kwargs)
        self.model.eval()

    def answer(
        self,
        messages: list[dict[str, Any]],
        *,
        num_frames: int | None,
        fps: float | None,
        max_pixels: int | None,
        max_new_tokens: int,
        temperature: float,
        enable_thinking: bool,
        video_cache: dict[str, tuple[Any, dict[str, Any]]] | None = None,
    ) -> str:
        video_kwargs: dict[str, Any] = {"do_sample_frames": False}
        if max_pixels is not None:
            # Qwen3VLVideoProcessor._preprocess() (transformers>=5.0's video
            # pipeline) does not accept a bare "max_pixels" kwarg -- its
            # signature only reads "size" (a dict/SizeDict with
            # "shortest_edge"/"longest_edge"). Passing "max_pixels" directly
            # used to be silently swallowed into **kwargs and ignored, so
            # this setting had ZERO effect: frames were resized only by
            # smart_resize()'s checkpoint-default budget (~25M px total for
            # this model), not by vqa_config.yaml's much smaller intended
            # value. That's also why max_pixels appeared to make no
            # difference in earlier testing -- it was never reaching the
            # processor. "longest_edge" is the actual upper bound we want;
            # "shortest_edge" is a lower bound smart_resize will upscale
            # to if the video is smaller still, so keep it tiny (well
            # below any real video) so it never kicks in in practice.
            video_kwargs["size"] = {"shortest_edge": 16384, "longest_edge": max_pixels}

        messages, video_metadata = materialize_videos(
            messages, num_frames=num_frames, fps=fps, video_cache=video_cache
        )
        video_kwargs["video_metadata"] = video_metadata

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=video_kwargs,
        )
        inputs = inputs.to(self.model.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        # Qwen3.5 interleaves a handful of standard full-attention layers
        # among its (mostly linear/Gated-Delta-Rule) decoder layers. Those
        # full-attention layers go through torch's scaled_dot_product_attention,
        # which picks a backend (flash / memory-efficient / cuDNN / math) at
        # runtime based on hardware and input shape. On a T4 (Grand
        # Challenge's GPU) none of the fast (linear-memory) backends are
        # available for this workload -- confirmed by explicitly excluding
        # "math" and observing torch raise "No available kernel" rather than
        # silently choosing one -- so math is the *only* backend that works
        # there, independent of sequence length. math materializes the full
        # [heads, seq_len, seq_len] attention-score matrix, so its memory
        # grows quadratically with sequence length; that quadratic blow-up
        # for our long, video-derived token sequences is what caused a
        # 12 GiB single-allocation CUDA OOM on Grand Challenge (never seen
        # on a dev GPU, where flash/memory-efficient kernels are chosen
        # instead of math). The fix that actually works on T4 is keeping
        # math available and shrinking the sequence length -- via
        # num_frames and (now that the bug above is fixed) max_pixels in
        # vqa_config.yaml -- so math's quadratic footprint fits in the
        # ~14.5 GiB budget. See that file for the current numbers.
        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        generated = generated[:, inputs["input_ids"].shape[1] :]
        answer = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        if "</think>" in answer:
            answer = answer.split("</think>", 1)[1].strip()
        return answer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer a question about an MP4 with Qwen3.5-4B.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--config", type=Path, help="YAML file containing command-line arguments.")
    parser.add_argument("--video", type=Path, help="Input video (for example, MP4).")
    question = parser.add_mutually_exclusive_group()
    question.add_argument("--question", help="Question text.")
    question.add_argument("--question-file", type=Path, help="UTF-8 text file containing the question.")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--prompt-config", type=Path)
    parser.add_argument("--few-shot-file", type=Path, help="Additional few-shot examples as a JSON array.")
    parser.add_argument("--context", help="Extra information such as instrument/organ names.")
    parser.add_argument("--context-file", type=Path, help="UTF-8 file containing extra information.")
    parser.add_argument("--disable-organ-classifier", action="store_true")
    parser.add_argument("--organ-model-path", type=Path)
    parser.add_argument("--organ-device", help="Organ classifier device (default: auto).")
    parser.add_argument("--organ-batch-size", type=int)
    parser.add_argument("--disable-tool-detector", action="store_true")
    parser.add_argument(
        "--tool-detector-backend",
        choices=["yolo", "rtdetrv2"],
        help=(
            "Instrument detection backend: 'yolo' (default, --tool-model-path/"
            "--tool-class-config) or 'rtdetrv2' (the tip_and_clevis3 RT-DETRv2 SSL "
            "ensemble, --tool-ensemble-config; requires a category1 checkout, see "
            "rtdetr_tool_detector.py)."
        ),
    )
    parser.add_argument("--tool-model-path", type=Path, help="YOLO checkpoint (backend=yolo).")
    parser.add_argument("--tool-class-config", type=Path, help="YOLO class names YAML (backend=yolo).")
    parser.add_argument(
        "--tool-ensemble-config",
        type=Path,
        help="RT-DETRv2 ensemble manifest, see category1/6_test/ensemble_tip_and_clevis3.yaml (backend=rtdetrv2).",
    )
    parser.add_argument(
        "--tool-disable-clevis-correction",
        action="store_true",
        help="Disable the isolated-clevis cross-tool correction for the rtdetrv2 backend (backend=rtdetrv2).",
    )
    parser.add_argument("--tool-max-detections-per-image", type=int, help="backend=rtdetrv2 only.")
    parser.add_argument("--tool-duplicate-iou-threshold", type=float, help="backend=rtdetrv2 only.")
    parser.add_argument(
        "--rtdetr-category1-root",
        type=Path,
        help="Override the category1 checkout used by the rtdetrv2 backend (default: auto-detected).",
    )
    parser.add_argument("--tool-device", help="Tool detector device (default: auto).")
    parser.add_argument("--tool-batch-size", type=int)
    parser.add_argument("--tool-confidence", type=float)
    parser.add_argument(
        "--tool-frame-count-threshold",
        type=int,
        help="Ignore instruments detected in at most this many distinct source frames.",
    )
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--num-frames", type=int, help="Frames sampled across the video (default: 32)."
    )
    sampling.add_argument("--fps", type=float, help="Sample at this rate instead of a fixed frame count.")
    parser.add_argument("--max-pixels", type=int, help="Maximum pixels per sampled frame.")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float, help="0 selects deterministic decoding.")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--device-map")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--quantization",
        choices=["none", "8bit", "4bit"],
        help="Load the VLM weights quantized via bitsandbytes to reduce GPU memory use.",
    )
    parser.add_argument("--output", type=Path, help="Write the answer to this UTF-8 text file.")
    return parser


def load_argument_config(path: Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("YAML configuration requires PyYAML") from error

    try:
        with path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML syntax: {error}") from error
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must contain a mapping at the top level")

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    result: dict[str, Any] = {}
    for raw_key, value in data.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"YAML config key must be a string: {raw_key!r}")
        key = raw_key.replace("-", "_")
        if key in result:
            raise ValueError(f"Duplicate YAML config key after normalization: {raw_key}")
        if key not in actions:
            raise ValueError(f"Unknown YAML config key: {raw_key}")

        action = actions[key]
        if action.nargs == 0:
            if not isinstance(value, bool):
                raise ValueError(f"YAML config key '{raw_key}' must be true or false")
        elif value is not None and action.type is not None:
            try:
                value = action.type(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid value for YAML config key '{raw_key}': {value!r}"
                ) from error
        elif value is not None and not isinstance(value, str):
            raise ValueError(f"YAML config key '{raw_key}' must be a string")
        if value is not None and action.choices is not None and value not in action.choices:
            choices = ", ".join(map(str, action.choices))
            raise ValueError(
                f"Invalid value for YAML config key '{raw_key}': {value!r} "
                f"(choose from {choices})"
            )
        result[key] = value

    for first, second in (("question", "question_file"), ("num_frames", "fps")):
        if result.get(first) is not None and result.get(second) is not None:
            raise ValueError(f"YAML config cannot set both '{first}' and '{second}'")
    return result


def _apply_exclusive_override(
    arguments: dict[str, Any], overrides: dict[str, Any], first: str, second: str
) -> None:
    if overrides.get(first) is not None:
        arguments[second] = None
    elif overrides.get(second) is not None:
        arguments[first] = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_argument_parser()
    cli_arguments = vars(parser.parse_args(argv))

    config_arguments: dict[str, Any] = {}
    config_path = cli_arguments.get("config")
    if config_path is not None:
        try:
            config_arguments = load_argument_config(config_path, parser)
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(f"could not load --config {config_path}: {error}")

    arguments = {**DEFAULT_ARGUMENTS, **config_arguments}
    _apply_exclusive_override(arguments, config_arguments, "question", "question_file")
    _apply_exclusive_override(arguments, config_arguments, "num_frames", "fps")
    arguments.update(cli_arguments)
    _apply_exclusive_override(arguments, cli_arguments, "question", "question_file")
    _apply_exclusive_override(arguments, cli_arguments, "num_frames", "fps")

    if arguments["video"] is None:
        parser.error("--video is required (on the command line or in --config)")
    if arguments["question"] is None and arguments["question_file"] is None:
        parser.error(
            "one of --question or --question-file is required "
            "(on the command line or in --config)"
        )
    return argparse.Namespace(**arguments)


def main() -> int:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video does not exist: {args.video}")
    if args.num_frames is not None and args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if args.organ_batch_size <= 0:
        raise ValueError("--organ-batch-size must be positive")
    if args.tool_batch_size <= 0:
        raise ValueError("--tool-batch-size must be positive")
    if not 0 <= args.tool_confidence <= 1:
        raise ValueError("--tool-confidence must be between 0 and 1")
    if args.tool_frame_count_threshold < 0:
        raise ValueError("--tool-frame-count-threshold must be non-negative")
    if args.tool_max_detections_per_image < 1:
        raise ValueError("--tool-max-detections-per-image must be at least 1")
    if not 0 <= args.tool_duplicate_iou_threshold <= 1:
        raise ValueError("--tool-duplicate-iou-threshold must be between 0 and 1")

    question = args.question
    if args.question_file:
        question = args.question_file.read_text(encoding="utf-8").strip()
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    context_parts = [args.context.strip()]
    if args.context_file:
        context_parts.append(args.context_file.read_text(encoding="utf-8").strip())
    video_cache = None
    frames = None
    video_metadata = None
    if not args.disable_organ_classifier or not args.disable_tool_detector:
        frames, video_metadata = decode_video(
            str(args.video), num_frames=args.num_frames, fps=args.fps
        )
        video_cache = {str(args.video.resolve()): (frames, video_metadata)}

    if not args.disable_organ_classifier:
        from organ_classifier import OrganClassifier

        classifier = OrganClassifier(args.organ_model_path, device=args.organ_device)
        prediction = classifier.predict(frames, batch_size=args.organ_batch_size)
        context_parts.append(f"Detected organ in the sampled video frames: {prediction.organ_name}.")
        print(f"Detected organ: {prediction.organ_name}", file=sys.stderr)
        del classifier

    if not args.disable_tool_detector:
        if args.tool_detector_backend == "rtdetrv2":
            from rtdetr_tool_detector import RTDetrToolDetector

            detector = RTDetrToolDetector(
                args.tool_ensemble_config,
                device=args.tool_device,
                disable_clevis_correction=args.tool_disable_clevis_correction,
                max_detections_per_image=args.tool_max_detections_per_image,
                duplicate_iou_threshold=args.tool_duplicate_iou_threshold,
                category1_root=args.rtdetr_category1_root,
            )
        else:
            from tool_detector import ToolDetector

            detector = ToolDetector(
                args.tool_model_path,
                args.tool_class_config,
                device=args.tool_device,
            )
        print(f"Tool detector backend: {args.tool_detector_backend}", file=sys.stderr)
        tool_predictions = detector.predict(
            frames,
            frame_indices=video_metadata["frames_indices"],
            source_fps=video_metadata["fps"],
            confidence=args.tool_confidence,
            batch_size=args.tool_batch_size,
            instrument_frame_count_threshold=args.tool_frame_count_threshold,
        )
        context_parts.append(tool_predictions.to_prompt_context())
        tool_statistics = tool_predictions.instrument_statistics
        detected_tools = ", ".join(
            statistic.instrument_name for statistic in tool_statistics
        ) or "none"
        print(f"Detected tools: {detected_tools}")
        if tool_statistics:
            print("Tool statistics:")
            for statistic in tool_statistics:
                print(
                    f"  {statistic.instrument_name}: frame_count={statistic.frame_count}, "
                    f"max_confidence={statistic.max_confidence:.3f}"
                )
        else:
            print("Tool statistics: none")
        print(
            f"Detected {tool_predictions.detection_count} tool parts in "
            f"{len(tool_predictions.frames)} sampled frames",
            file=sys.stderr,
        )
        del detector
    context = "\n".join(part for part in context_parts if part)

    config = PromptConfig.from_json(args.prompt_config)
    messages = build_messages(
        config=config,
        video_path=args.video,
        question=question,
        additional_context=context,
        extra_few_shots=load_few_shots(args.few_shot_file),
    )
    model = VideoQAModel(
        args.model_path,
        device_map=args.device_map,
        dtype=args.dtype,
        quantization=args.quantization,
    )
    answer = model.answer(
        messages,
        num_frames=args.num_frames,
        fps=args.fps,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        video_cache=video_cache,
    )
    print(answer)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(answer + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
