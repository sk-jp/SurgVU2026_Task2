# SurgVU2026 Category2 — Video VQA (fs2 / v2 configuration)

手術動画（MP4）1本と質問文1つを入力として、テキストの回答を1つ出力する VQA
（Visual Question Answering）パイプラインです。ベースモデルは
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)（画像・動画対応の
Vision-Language Model）で、ファインチューニングは行わず推論のみを行います。

このリポジトリは、元プロジェクトの `2_vlm` で最もスコアが良かった設定
（system prompt "fs" の改良版 = **v2**、few-shot セット **fs2**。Grand
Challenge の BERTScore で 0.6488、他の全バリエーションより明確に高スコア）
だけを、外部でも動かせる形に切り出したものです。

VLM への入力は動画と質問文だけでなく、次の2つの補助情報でプロンプトを
補強します。

- **臓器分類**（EfficientNet-V2-S、8クラス）: サンプリングした各フレームを
  分類し、動画全体で多数決した臓器名をコンテキストに追加します。
- **手術器具検出**（YOLO）: 同じフレームから器具名・部位（`tip`/`clevis`。
  `shaft` は破棄）・信頼度・正規化バウンディングボックスを検出し、フレーム
  ごとにコンテキストへ追加します。

動画フレームは1回だけデコードし、VLM・臓器分類・器具検出の3処理で共有します。

## ディレクトリ構成

```
5_github/
├── README.md                    このファイル
├── requirements.txt              依存パッケージ
├── vqa.py                        メインスクリプト（動画抽出・プロンプト構築・VLM推論）
├── organ_classifier.py           臓器分類（EfficientNet-V2-S）
├── tool_detector.py               手術器具・部位検出（YOLO）
├── vqa_config.yaml                実行設定（v2/fs2 の推奨値を反映済み）
├── prompt_config.json             システムプロンプト（v2: hedgeせず単一の回答に断定させる設定）
├── few_shots.json                 few-shot 例 25件（fs2）
├── test_vqa.py                    単体テスト
├── organs_classifier/
│   └── efficientnet_v2_s_v3/
│       └── (epoch=29_test_loss=0.04914_f1_avg=0.98064.ckpt)   臓器分類の学習済み重み（別途ダウンロード）
└── tool_detection/
    ├── best.pt                    YOLO器具検出の学習済み重み
    └── surgvu.yaml                器具検出の42クラス定義（14器具 × tip/clevis/shaft）
```

Qwen3.5-4B 本体の重み、および臓器分類モデルの重みはサイズが大きいため
このリポジトリには含まれていません。下記「Qwenモデルの準備」「臓器分類
モデルの準備」のとおり別途取得してください（器具検出モデル `tool_detection/best.pt`
はサイズが小さいためこのリポジトリに含まれています）。

## セットアップ

```bash
cd 5_github
pip install -r requirements.txt
```

GPU（CUDA）での実行を推奨します。`--*-device auto`（既定値）は CUDA が
使えればそれを、なければ CPU を使いますが、Qwen3.5-4B を CPU だけで動かすと
非常に低速です。

## Qwenモデルの準備

`vqa_config.yaml` の `model_path`（または `--model-path`）は、次のどちらの
指定方法にも対応しています。

### 方法A: Hugging Face から自動ダウンロード（既定）

`vqa_config.yaml` の既定値は Hugging Face Hub のリポジトリID
（`Qwen/Qwen3.5-4B`）です。ローカルディレクトリとして存在しない値を渡すと、
`transformers` がそのまま Hugging Face Hub からダウンロードし、
`~/.cache/huggingface` 以下に自動でキャッシュします。初回実行時にネット
ワーク接続が必要で、モデルサイズはおよそ 9GB です（2回目以降はキャッシュ
から読み込むためオフラインで動作します）。ダウンロードに認証が必要な場合は
事前に `huggingface-cli login` を実行してください。

```bash
python vqa.py --config vqa_config.yaml \
  --video /path/to/case.mp4 \
  --question "Which instrument is manipulating the tissue?"
```

### 方法B: 利用者が事前にダウンロードして指定

完全オフラインで実行したい場合や、ダウンロードを事前に済ませておきたい場合は、
`huggingface-cli` などで自分でモデルを取得し、そのローカルディレクトリを
指定してください。ローカルディレクトリが存在する場合は `local_files_only`
（オフラインモード）で読み込まれ、ネットワークへは一切アクセスしません。

```bash
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./Qwen3.5-4B
```

```yaml
# vqa_config.yaml
model_path: ./Qwen3.5-4B
```

またはコマンドラインから直接指定できます。

```bash
python vqa.py --config vqa_config.yaml --model-path ./Qwen3.5-4B \
  --video /path/to/case.mp4 --question "..."
```

## 臓器分類モデルの準備

臓器分類（EfficientNet-V2-S）の学習済み重み（.ckpt）はサイズが大きいため
このリポジトリには含まれていません。下記のGoogle Driveからダウンロードし、
`organs_classifier/efficientnet_v2_s_v3/` に配置してください。

- https://drive.google.com/drive/folders/1Fbnf1htcuoRPk3iGnnMliPTs9ULSkdPT

配置後、`vqa_config.yaml` の `organ_model_path`（既定値
`organs_classifier/efficientnet_v2_s_v3/epoch=29_test_loss=0.04914_f1_avg=0.98064.ckpt`）
がそのファイルを指すようにしてください（ダウンロードしたファイル名が既定値と
異なる場合は、`vqa_config.yaml` の `organ_model_path` または
`--organ-model-path` をそのファイル名に合わせて書き換えてください）。

臓器分類を使わずに実行したい場合は `--disable-organ-classifier`
（または `vqa_config.yaml` の `disable_organ_classifier: true`）を指定すれば
この重みは不要です。ただしその場合 v2/fs2 の構成とは挙動が変わります。

## 使い方

```bash
cd 5_github
python vqa.py --config vqa_config.yaml \
  --video /path/to/case.mp4 \
  --question "Which instrument is manipulating the tissue?"
```

`vqa_config.yaml` は `prompt_config.json` と `few_shots.json`（fs2 の
25件の few-shot 例）をすでに指定しているため、追加設定なしで v2/fs2 の
挙動を再現します。コマンドラインで明示的に指定した引数は YAML の値より
優先されます。相対パスはカレントディレクトリからの相対パスとして解決
されます。

質問は `--question` の代わりに `--question-file` でテキストファイルから
渡すこともできます。回答本文だけが必要な場合は `--output answer.txt` を
指定してください（標準出力には検出された器具一覧や統計も出力されるため、
標準出力全体をそのまま回答として使う設計にはなっていません）。

検出された臓器は標準エラー出力に表示されます（例:
`Detected organ: sigmoid colon`）。

```bash
python vqa.py --config vqa_config.yaml \
  --video /path/to/case.mp4 \
  --question "Is a needle driver involved in the procedure?" \
  --output answer.txt
```

### 主なオプション

| オプション | 説明 | 既定値 |
|---|---|---|
| `--model-path` | Qwenモデル（ローカルディレクトリまたはHub ID） | `Qwen/Qwen3.5-4B` |
| `--prompt-config` | システムプロンプト設定 | `prompt_config.json` |
| `--few-shot-file` | 追加 few-shot 例（JSON配列） | `few_shots.json`（YAML内で指定） |
| `--context` / `--context-file` | 手動で追加するコンテキスト | なし |
| `--disable-organ-classifier` | 臓器分類を無効化 | 有効 |
| `--disable-tool-detector` | 器具検出を無効化 | 有効 |
| `--tool-confidence` | 器具検出の信頼度閾値 | `0.8`（vqa_config.yamlでチューニング済み） |
| `--tool-frame-count-threshold` | この数以下のフレームでしか検出されなかった器具を除外 | `10` |
| `--num-frames` / `--fps` | フレームサンプリング数／FPS（どちらか一方） | `num_frames=32` |
| `--max-pixels` | 空間方向のメモリ使用量 | `262144` |
| `--quantization` | `none` / `4bit` / `8bit`（GPUメモリが少ない場合） | `none` |
| `--output` | 回答の保存先 | 標準出力のみ |

すべてのオプションは `python vqa.py --help` で確認できます。

## 動作確認（テスト）

```bash
python -m unittest test_vqa -v
```

モデル本体を使わない、プロンプト構築・器具検出結果の整形などの単体テスト
です。8件中1件（`test_tool_detections_are_formatted_for_every_frame`）は
元プロジェクトの時点ですでに `tool_detector.to_prompt_context()` の出力
フォーマットとテストの期待値がずれており失敗します。VQA本体の動作には
影響しません。

実際に動画を使ったエンドツーエンド動作は、GPU環境・Qwenモデル・サンプル
動画を用意したうえで、上記「使い方」のコマンドで確認してください。

## 補足・既知の制限

- **器具検出のバックエンドは YOLO のみ**です。元プロジェクトには
  RT-DETRv2 アンサンブルという代替バックエンドもありますが、そちらは
  別リポジトリ（`category1`）のチェックアウトが別途必要なため、本
  リポジトリには含めていません（`--tool-detector-backend` は
  `yolo` のまま使用してください）。
- ファインチューニングやオンラインAPI呼び出しは行いません。Qwenモデルの
  重み取得（Hugging Face Hubからのダウンロード）以外はネットワークに
  アクセスしません。
- `tool_detection/best.pt`（約40MB）はこのリポジトリに含まれています。
  `organs_classifier/` の学習済み重み（約78MB）はサイズが大きいため含まれて
  おらず、上記「臓器分類モデルの準備」のとおりGoogle Driveから別途取得が
  必要です。
- `prompt_config.json` / `few_shots.json` は元プロジェクトの
  `prompt_config.v2.json` / `few_shots.generated.v2.json`（Grand
  Challenge 提出でスコア最良だった構成）そのものです。system prompt は
  「根拠不十分でも単一の回答に断定し、ヘッジ（曖昧な言い回し）をしない」
  よう指示しています。

## ライセンス・出典

- ベースモデル: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)（Apache-2.0）
- 臓器分類器: capybara の EfficientNet-V2-S 学習コードから移植
- 器具検出器: Ultralytics YOLO で学習（クラス定義は `tool_detection/surgvu.yaml`）
