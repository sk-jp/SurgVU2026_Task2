# SurgVU2026 Category2 — Video VQA (fs2 / v2 configuration)

A VQA (Visual Question Answering) pipeline that takes one surgical video
(MP4) and one question as input and outputs a single text answer. The base
model is [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (a
vision-language model that supports images and video); no fine-tuning is
performed, inference only.

This repository is a standalone, externally runnable extraction of the
single best-scoring configuration from the original project's `2_vlm`
(the refined "fs" system prompt = **v2**, few-shot set **fs2**; BERTScore
0.6488 on the Grand Challenge leaderboard, clearly ahead of every other
variant tried).

Beyond the video and the question text, the VLM input is augmented with two
kinds of auxiliary context:

- **Organ classification** (EfficientNet-V2-S, 8 classes): each sampled
  frame is classified, and the majority-vote organ name across the whole
  video is added to the context.
- **Surgical instrument detection** (YOLO): from the same frames, instrument
  name, part (`tip`/`clevis`; `shaft` is discarded), confidence, and
  normalized bounding box are detected and added to the context per frame.

Video frames are decoded only once and shared across the three
processes (VLM, organ classification, instrument detection).

## Directory layout

```
5_github/
├── README.md                    this file
├── requirements.txt              dependencies
├── vqa.py                        main script (frame extraction, prompt building, VLM inference)
├── organ_classifier.py           organ classification (EfficientNet-V2-S)
├── tool_detector.py               surgical instrument/part detection (YOLO)
├── vqa_config.yaml                run configuration (already reflects the recommended v2/fs2 values)
├── prompt_config.json             system prompt (v2: instructs the model to commit to a single answer without hedging)
├── few_shots.json                 25 few-shot examples (fs2)
├── test_vqa.py                    unit tests
├── organs_classifier/
│   └── efficientnet_v2_s_v3/
│       └── (epoch=29_test_loss=0.04914_f1_avg=0.98064.ckpt)   organ classifier weights (download separately)
└── tool_detection/
    ├── best.pt                    YOLO instrument detector weights
    └── surgvu.yaml                42-class definition for instrument detection (14 instruments × tip/clevis/shaft)
```

The Qwen3.5-4B weights and the organ classifier weights are large and are
not included in this repository. Obtain them separately as described in
"Preparing the Qwen model" and "Preparing the organ classifier model"
below (the instrument detector weights `tool_detection/best.pt` are small
enough to be included in this repository).

## Setup

```bash
cd 5_github
pip install -r requirements.txt
```

Running on a GPU (CUDA) is recommended. `--*-device auto` (the default)
uses CUDA when available and falls back to CPU otherwise, but running
Qwen3.5-4B on CPU alone is very slow.

## Preparing the Qwen model

`model_path` in `vqa_config.yaml` (or `--model-path`) supports either of the
following ways of specifying the model.

### Method A: Automatic download from Hugging Face (default)

The default value in `vqa_config.yaml` is the Hugging Face Hub repo ID
(`Qwen/Qwen3.5-4B`). If the value passed does not exist as a local
directory, `transformers` downloads it directly from the Hugging Face Hub
and caches it under `~/.cache/huggingface`. A network connection is
required on first run, and the model is roughly 9GB (subsequent runs load
from the cache and work offline). If the download requires authentication,
run `huggingface-cli login` beforehand.

```bash
python vqa.py --config vqa_config.yaml \
  --video /path/to/case.mp4 \
  --question "Which instrument is manipulating the tissue?"
```

### Method B: Download in advance and point to it yourself

If you want to run fully offline or would rather download the model ahead
of time, fetch it yourself with `huggingface-cli` or similar and point to
that local directory. When a local directory exists, it is loaded with
`local_files_only` (offline mode) and no network access occurs at all.

```bash
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./Qwen3.5-4B
```

```yaml
# vqa_config.yaml
model_path: ./Qwen3.5-4B
```

Or specify it directly on the command line:

```bash
python vqa.py --config vqa_config.yaml --model-path ./Qwen3.5-4B \
  --video /path/to/case.mp4 --question "..."
```

## Preparing the organ classifier model

The trained weights (.ckpt) for organ classification (EfficientNet-V2-S)
are large and not included in this repository. Download them from the
Google Drive link below and place them under
`organs_classifier/efficientnet_v2_s_v3/`.

- https://drive.google.com/drive/folders/1Fbnf1htcuoRPk3iGnnMliPTs9ULSkdPT

After placing the file, make sure `organ_model_path` in `vqa_config.yaml`
(default:
`organs_classifier/efficientnet_v2_s_v3/epoch=29_test_loss=0.04914_f1_avg=0.98064.ckpt`)
points to it (if the downloaded file's name differs from the default,
update `organ_model_path` in `vqa_config.yaml`, or `--organ-model-path`, to
match it).

If you want to run without organ classification, pass
`--disable-organ-classifier` (or set `disable_organ_classifier: true` in
`vqa_config.yaml`), and this weight file is not needed. Note that this
changes behavior relative to the v2/fs2 configuration.

## Usage

```bash
cd 5_github
python vqa.py --config vqa_config.yaml \
  --video /path/to/case.mp4 \
  --question "Which instrument is manipulating the tissue?"
```

`vqa_config.yaml` already specifies `prompt_config.json` and
`few_shots.json` (the 25 fs2 few-shot examples), so it reproduces v2/fs2
behavior with no extra configuration. Arguments given explicitly on the
command line take precedence over the values in the YAML. Relative paths
are resolved relative to the current directory.

Instead of `--question`, you can also pass the question from a text file
via `--question-file`. If you only need the answer text, use `--output
answer.txt` (standard output also includes the list of detected
instruments and other stats, so it is not designed to be used as the
answer verbatim).

Detected organs are printed to standard error (e.g.
`Detected organ: sigmoid colon`).

```bash
python vqa.py --config vqa_config.yaml \
  --video /path/to/case.mp4 \
  --question "Is a needle driver involved in the procedure?" \
  --output answer.txt
```

### Main options

| Option | Description | Default |
|---|---|---|
| `--model-path` | Qwen model (local directory or Hub ID) | `Qwen/Qwen3.5-4B` |
| `--prompt-config` | System prompt configuration | `prompt_config.json` |
| `--few-shot-file` | Additional few-shot examples (JSON array) | `few_shots.json` (set in the YAML) |
| `--context` / `--context-file` | Manually supplied extra context | none |
| `--disable-organ-classifier` | Disable organ classification | enabled |
| `--disable-tool-detector` | Disable instrument detection | enabled |
| `--tool-confidence` | Instrument detection confidence threshold | `0.8` (tuned in vqa_config.yaml) |
| `--tool-frame-count-threshold` | Drop instruments detected in this many frames or fewer | `10` |
| `--num-frames` / `--fps` | Number of sampled frames / FPS (mutually exclusive) | `num_frames=32` |
| `--max-pixels` | Spatial memory footprint | `262144` |
| `--quantization` | `none` / `4bit` / `8bit` (for limited GPU memory) | `none` |
| `--output` | Where to save the answer | stdout only |

All options can be listed with `python vqa.py --help`.

## Testing

```bash
python -m unittest test_vqa -v
```

These are unit tests for prompt construction, formatting of instrument
detection results, etc., and do not exercise the model itself. 1 out of 8
tests (`test_tool_detections_are_formatted_for_every_frame`) was already
failing in the original project, because the expected format in the test
had drifted from the actual output of `tool_detector.to_prompt_context()`.
This does not affect the behavior of the VQA pipeline itself.

To verify actual end-to-end behavior with a real video, prepare a GPU
environment, the Qwen model, and a sample video, then run the commands
under "Usage" above.

## Notes and known limitations

- **The instrument detection backend is YOLO only.** The original project
  also had an alternative backend, an RT-DETRv2 ensemble, but that requires
  a separate checkout of another repository (`category1`) and is not
  included here (keep `--tool-detector-backend` set to `yolo`).
- No fine-tuning or online API calls are performed. Other than fetching the
  Qwen model weights (downloading from the Hugging Face Hub), no network
  access occurs.
- `tool_detection/best.pt` (about 40MB) is included in this repository. If
  you'd rather not rely on the copy checked into git (e.g. a shallow
  clone, or the file is missing for some other reason), it is also
  available from Dropbox:
  https://www.dropbox.com/scl/fi/uwhar54exvn955yqcl95o/best.pt?rlkey=28x71n3jlwur2pcxff8u85no8&st=q7rzv18z&dl=0
  — download it and place it at `tool_detection/best.pt`.
  The trained weights under `organs_classifier/` (about 78MB) are large and
  not included; obtain them separately from Google Drive as described in
  "Preparing the organ classifier model" above.
- `prompt_config.json` / `few_shots.json` are exactly
  `prompt_config.v2.json` / `few_shots.generated.v2.json` from the original
  project (the configuration that scored best on the Grand Challenge
  submission). The system prompt instructs the model to commit to a single
  answer even when the evidence is insufficient, and not to hedge.

## License and credits

- Base model: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0)
- Organ classifier: ported from capybara's EfficientNet-V2-S training code
- Instrument detector: trained with Ultralytics YOLO (class definitions in `tool_detection/surgvu.yaml`)
