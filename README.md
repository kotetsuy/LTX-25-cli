# LTX-25-cli

## ライセンス

本プロジェクトのコードは [Apache License 2.0](LICENSE) で公開しています。
使用する外部ライブラリ・モデル・重みには、それぞれのライセンス・利用条件が適用されます。

LTX-2.5 の動画・音声生成を、ComfyUI を使わず Python スクリプトから実行する実験用プロジェクトです。diffusers の `LTX2Pipeline` を使用し、AMD ROCm / gfx1151（Radeon 8060S）環境を対象にしています。

## 現在の状況

2026-09-25 時点のコードとローカル成果物に基づく状況です。

- テキストの埋め込み生成、distilled の 8 ステップ推論、中間 latent の保存を実装済みです。
- `outputs/fox_5s.mp4` があり、768×512、121 フレーム、24 fps（約 5 秒）の映像と音声ストリームを確認しています。
- `t2v_resume.py` は空間・時間方向の VAE 分割デコードと、動画書き出し前の `uint8` 変換に対応しています。
- 統合 CLI `ltx25 generate` / `ltx25 decode` を実装しています。2026-09-26 に `generate --image` による画像からの動画生成を追加し、latent 生成と別プロセスでの MP4 書き出しを検証しました。補間は未実装です。

2026-09-25 に修正後の `t2v_resume.py` を `--phase both --offload none --vae-tiling` で実行し、新規の埋め込み生成から MP4・計測 JSON の保存まで終了コード 0 で完走しました。出力の全 121 映像フレームと 48 kHz 音声を PyAV でデコードして確認しています。保存済みの `fox_5s.log` は修正前の型エラーの記録です。[HANDOFF.md](HANDOFF.md) は 2026-08-16 時点の調査記録で、デコード未完という記述などは現在の成果物より古い情報です。

## 対象環境

開発時の環境記録は次のとおりです。他の GPU での動作は未検証です。

| 項目 | 構成 |
| --- | --- |
| 本体 | GMKtec NucBox EVO X2 / Ryzen AI MAX+ 395 |
| GPU | Radeon 8060S Graphics / gfx1151 |
| メモリ | 96 GB 統合メモリ、BIOS で VRAM に 48 GB 割り当て、OS 側は約 45 GiB |
| OS | Ubuntu 26.04 |
| Python | 3.12（`>=3.12,<3.13`） |
| パッケージ管理 | uv、`uv.lock` 同梱 |
| PyTorch | AMD の gfx1151 専用 wheel を使用 |
| diffusers | Git リポジトリから取得（解決済みリビジョンは `uv.lock`） |

モデルの保存には約 67 GiB に加え、Python 環境・ダウンロードキャッシュ・生成物用の空き容量が必要です。この容量はダウンロードスクリプトと開発記録に基づく目安です。

## セットアップ

以下はプロジェクトのルートディレクトリで実行します。uv と、GPU を利用できるホスト側の AMD ドライバ環境を事前に用意してください。

```bash
unset HSA_OVERRIDE_GFX_VERSION
uv sync --locked
uv run python scripts/check_env.py
```

`pyproject.toml` は torch / torchvision / torchaudio を `https://repo.amd.com/rocm/whl/gfx1151/` から取得する設定です。開発記録では汎用 wheel で `hipErrorInvalidImage` が発生しています。ネイティブ gfx1151 ビルドを使用するため、`HSA_OVERRIDE_GFX_VERSION` は設定しません。

`check_env.py` は GPU の認識、bf16 行列演算、必要な diffusers クラスの存在を確認します。ROCm 環境でも PyTorch の API 名は `torch.cuda` です。

### モデルの取得

```bash
uv run python scripts/download_models.py --dry-run
uv run python scripts/download_models.py
```

取得元は `Lightricks/LTX-2.5-Diffusers`、標準の保存先は `models/ltx-2.5-diffusers/` です。シャードの index を参照し、重複する重みと非 distilled の `transformer_full/` を通常の取得対象から除外します。`--dry-run` も index の取得にネットワークを使用します。

アクセス制限によるエラーが出る場合は、対象モデルの利用条件を確認し、承認された Hugging Face アカウントで `uv run hf auth login` を実行してください。

| オプション | 内容 |
| --- | --- |
| `--dest PATH` | 保存先を変更 |
| `--dry-run` | 取得パターンを表示して終了 |
| `--full-transformer` | 非 distilled transformer も取得（約 35 GiB 追加） |
| `--lora` | distilled LoRA も取得（約 9 GiB 追加） |

追加の重みを取得しても、生成スクリプトが自動的に使用するわけではありません。保存先を変えた場合は、生成時にも `--model PATH` を指定します。

必要に応じて、各モデル部品を順番にロードして確認できます。大きな重みを読み込むため、実行には時間とメモリを使います。

```bash
uv run python scripts/check_env.py --model models/ltx-2.5-diffusers
```

## 統合 CLI

`uv sync --locked` でコマンドを登録し、プロジェクトのルートから実行します。

```bash
uv run ltx25 --help
uv run ltx25 generate --prompt "A red fox walking through a snowy forest." --output outputs/demo.mp4
uv run ltx25 decode --input outputs/demo.latents.pt --output outputs/demo_decoded.mp4
```

`generate` は埋め込み生成・denoise・動画と音声の書き出し・計測 JSON 保存までを実行します。`decode` は指定した latent から書き出しのみを行い、出力と同じベース名の `.metrics.json` を保存します。`python -m ltx25_cli` からも実行できます。

CLI と互換スクリプトでは空間・時間方向の VAE 分割デコードが標準で有効です。明示的に無効化する場合は `--no-vae-tiling` を指定します。生成オプションは下記のスクリプト版と共通ですが、CLI の出力既定値は `outputs/generated.mp4` です。幅・高さは 32 の倍数、フレーム数は `8n+1`、fps は正の整数、出力拡張子は `.mp4` を指定します。

モデルの既定パスは実行ディレクトリ基準の `models/ltx-2.5-diffusers` です。別の場所から実行する場合は `--model` を指定してください。既存の同名出力は上書きされます。

### 写真から動画を生成する

`--image` に開始フレームとなる画像を指定すると、`LTX2ImageToVideoPipeline` に切り替わります。モデルの追加ダウンロードは不要です。テキストの埋め込み生成、8 ステップ推論、latent 保存、分割デコードはテキスト生成と共通です。

```bash
uv run ltx25 generate \
  --image inputs/race_delorean.png \
  --prompt "A silver DeLorean drives ahead of a group of racehorses on a racetrack. The car and horses move forward together. The camera tracks alongside them smoothly." \
  --width 768 --height 512 --frames 121 --fps 24 \
  --offload none \
  --output outputs/race_delorean.mp4
```

`inputs/race_delorean.png` は利用者が用意する画像です。レース風景と車が別々の写真の場合は、先に希望の位置へ車を合成した 1 枚の画像を作成してください。このコマンドは 2 枚の写真の自動合成や、既存動画への物体追加を行いません。

- `--image-fit contain`（標準）：縦横比を維持し、画像全体が収まるよう黒い余白を追加します。
- `--image-fit cover`：縦横比を維持して画面を埋め、はみ出す部分を中央基準で切り取ります。
- 写真の EXIF 回転を反映し、透過部分は黒背景に変換します。アニメーション画像は最初のフレームを使用します。
- 入力画像のパスとサイズ調整方法を latent と計測 JSON に記録します。`decode` での再書き出しには元画像は不要です。

画像は開始状態を指定します。以後の車の形状や馬の動きの維持は保証されません。画像の前処理・CLI・パイプラインへの画像引き渡しを含む CPU テスト 8 件が通過しています。

2026-09-26 に `inputs/race_delorean.png` から画像条件付き推論を実行し、latent 保存まで完了しました（推論区間 161.0 秒、ピーク GPU メモリ 45.5 GiB）。その後の一括デコードが数分経っても完了しなかったため実行を終了し、以下の設定で保存済み latent からデコードを再試行しました。再試行は正常終了し、映像デコード 15.3 秒、音声デコード 1.6 秒、MP4 書き出し 1.0 秒でした。設定変更とプロセス分離の両方を行ったため、遅延原因の切り分けは未完了です。

```bash
MIOPEN_FIND_MODE=FAST MIOPEN_USER_DB_PATH="$PWD/miopen-cache" \
  uv run ltx25 decode --input outputs/race_delorean.latents.pt \
  --output outputs/race_delorean.mp4
```

`outputs/race_delorean.mp4` は 768×512 / 24 fps / 121 フレーム（約 5 秒）、48 kHz 音声を PyAV で全件読み戻しました。検証結果は `outputs/race_delorean.validation.json`、デコード計測は `outputs/race_delorean.metrics.json` にあります。途中と最終フレームで車が馬群の前にあることを確認しましたが、車や馬の細部には生成による変化があります。

### CLI の検証結果（2026-09-25）

`ltx25 generate` を新規埋め込みから実行し、その latent を `ltx25 decode` に渡して、両コマンドの正常終了と計測 JSON の保存を確認しました。両方の MP4 は全 121 フレーム（768×512 / 24 fps）と 48 kHz 音声を PyAV で読み戻せています。計測区間の合計は生成 182.8 秒、再デコード 21.2 秒でした。

成果物とログは `outputs/cli_e2e_20260925.*`、`outputs/cli_decode_20260925.*` にあります。GPU を使用しない 4 件の CLI テストも通過しています。

```bash
uv run python -m unittest discover -s tests -v
```

## テキストから動画を生成する（互換スクリプト）

互換スクリプト `t2v_resume.py` も統合 CLI の生成処理を呼び出します。開発環境では過去に VAE デコード中のシステム全体のフリーズが記録されているため、`--vae-tiling` を付けて実行してください。改良版では空間タイリングと時間方向の分割デコードの両方を有効にします。

```bash
uv run python scripts/t2v_resume.py \
  --prompt "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot." \
  --width 768 --height 512 --frames 121 --fps 24 \
  --seed 42 --offload none --vae-tiling \
  --output outputs/demo.mp4
```

標準の `--phase both` は、テキストエンコードを別プロセスで実行し、メモリを解放してから動画生成に進みます。推論後は latent を保存し、transformer と connectors を解放してから映像・音声をデコードします。

| オプション | 既定値 / 内容 |
| --- | --- |
| `--phase` | `both`。`encode` / `generate` の個別実行も可能 |
| `--model` | プロジェクト内の `models/ltx-2.5-diffusers` |
| `--prompt` | 雪の森を歩く赤いキツネの英文 |
| `--negative-prompt` | diffusers の `DEFAULT_NEGATIVE_PROMPT` |
| `--max-sequence-length` | `1024` |
| `--embeds` | 出力名の拡張子を `.embeds.pt` に置換したパス |
| `--width` / `--height` | `768` / `512` |
| `--frames` / `--fps` | `121` / `24.0` |
| `--seed` | `42` |
| `--offload` | `none`。`model` / `sequential` も選択可能 |
| `--vae-tiling` | 標準で有効。`--no-vae-tiling` で無効化 |
| `--output` | `outputs/t2v_smoke.mp4` |

ステップ数は `DISTILLED_SIGMA_VALUES` による 8 ステップで、変更用の引数はありません。書き出し時の fps は整数へ変換されるため、整数値を指定してください。

### 埋め込みキャッシュの再利用

`both` はキャッシュ内のプロンプト、ネガティブプロンプト、最大系列長が一致すればエンコードを省略します。モデルの同一性は検査しないため、モデルを変える場合は別の `--embeds` パスを使用してください。

このフォルダにある既存キャッシュを、スクリプトの既定プロンプトで再利用する例です。

```bash
uv run python scripts/t2v_resume.py \
  --phase generate \
  --embeds outputs/t2v_smoke.embeds.pt \
  --offload none --vae-tiling \
  --output outputs/fox_retry.mp4
```

`--phase generate` は有効なキャッシュがなければ終了します。任意のプロンプトで先にキャッシュだけ作る場合は `--phase encode --prompt "..." --embeds outputs/custom.embeds.pt` を指定し、生成時も同じプロンプトとキャッシュを指定します。

### 保存済み latent からデコードを再開する

```bash
uv run python scripts/decode_saved.py
```

この互換スクリプトの既定入力は `outputs/fox_5s.latents.pt`、既定出力は `outputs/fox_5s.mp4` です。統合 CLI の `decode` に処理を委譲し、`--input` / `--output` / `--model` で変更できます。既存の同名 MP4 は上書きされます。

`generate` 自体は保存済み latent の自動読み込みには対応していません。デコードのみの再試行には `ltx25 decode` を使います。

## 出力と計測

`--output outputs/demo.mp4` の場合、次のファイルを使用・作成します。

| ファイル | 内容 |
| --- | --- |
| `demo.embeds.pt` | プロンプト埋め込み。`--embeds` 指定時はそちらを使用 |
| `demo.latents.pt` | 映像・音声 latent と生成条件。denoise 後、デコード前に保存・fsync |
| `demo.mp4` | 音声付き動画 |
| `demo.metrics.json` | 一括生成が正常終了した場合の生成条件・各段階の時間・メモリ情報 |

ログの `peak_vram_gib` は PyTorch が計測する割り当て済み GPU メモリのピーク、`rss_gib` は各段階終了時のプロセス RSS です。システム全体のメモリ使用量ではありません。キャッシュを再利用した場合、集計には保存済みのエンコード時間も含まれます。

今回の一括検証では denoise が 115.9 秒、デコードが 18.2 秒、各計測区間の合計が 180.9 秒、ピーク GPU メモリが 44.93 GiB でした。計測区間の合計にはプロセス起動など区間外の時間を含みません。特定環境での一回の結果であり、画質・音質の主観評価や繰り返し実行の安定性検証は含みません。

検証成果物は `outputs/e2e_20260925.mp4`、`.log`、`.metrics.json`、`.validation.json` に保存しています。同じベース名の `.embeds.pt` と `.latents.pt` も保存済みです。実行時は `MIOPEN_FIND_MODE=FAST` と、プロジェクト内の `miopen-cache/` を `MIOPEN_USER_DB_PATH` に指定しました。MIOpen の perf DB 警告は出ましたが、デコード・書き出しとも正常終了しました。

## メモリと既知の注意点

- モデル全体を一度にロードすると、開発機のシステム RAM / VRAM のどちらにも収まりません。エンコードと生成の 2 段階構成を前提にしています。
- 開発機では CPU オフロードによるメモリ圧迫とクラッシュが記録されています。実行例の `--offload none` は、生成側の重み約 43 GiB を GPU に載せる構成です。
- `t2v_smoke.py` は旧実装です。latent 保存・時間方向の分割デコード・書き出し時の `uint8` 変換がなく、過去のフリーズ記録もあるため、通常は改良版を使ってください。
- gfx1151 用 MIOpen perf DB が読めない警告が記録されています。必要に応じ、以下の環境変数で探索モードと書き込み先を指定できます。フリーズの解消を保証する設定ではありません。

  ```bash
  mkdir -p miopen-cache
  export MIOPEN_FIND_MODE=FAST
  export MIOPEN_USER_DB_PATH="$PWD/miopen-cache"
  ```

生成スクリプトは torch の import 前に、未設定の場合のみ `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` と `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定します。

## ファイル構成

```text
pyproject.toml             依存関係と gfx1151 向け取得設定
uv.lock                    Python 依存関係のロックファイル
scripts/
  check_env.py             GPU・ライブラリ・モデル部品の確認
  download_models.py       モデルの選択的ダウンロード
  t2v_smoke.py             初期の生成スクリプト
  t2v_resume.py            共通 runner を呼ぶ生成用互換スクリプト
  decode_saved.py          保存済み latent からの動画・音声書き出し
src/ltx25_cli/
  cli.py                  generate / decode の引数と入力検証
  runner.py               生成・デコードの共通処理
  __main__.py             python -m ltx25_cli の入口
tests/test_cli.py         GPU を使わない CLI テスト
models/                    ローカルモデル（Git 管理対象外）
outputs/                   キャッシュ・動画・ログ（Git 管理対象外）
miopen-cache/              ローカルの MIOpen キャッシュ用ディレクトリ
HANDOFF.md                 過去の環境構築・障害調査・今後の計画
```
