# HANDOFF.md — LTX-25-cli

> 2026-09-25 追記：下記は 8 月時点の履歴です。現在は一括生成の完走を確認し、`ltx25 generate` / `ltx25 decode` を実装しています。最新の実行方法・検証状況は README.md を参照してください。

最終更新: 2026-08-16

## プロジェクト概要

LTX-2.5（Lightricks製オープン動画生成AI）を、ComfyUIのGUIを介さずコマンドラインから直接実行するための自作CLIツールを実装する。diffusersライブラリの`LTX2Pipeline`をベースに、AMD ROCm環境（Strix Halo / gfx1151）での動作を前提とする。

第一弾のユースケースは「アハ体験」動画デモ（じわじわ変化する画像→動画）だが、通常のtext-to-video / image-to-videoにも対応した汎用CLIとする。

## 現状サマリ（2026-08-16 時点）

**Phase 1（環境構築）完了。Phase 2（最小推論）は denoise まで到達したが、VAE decode でマシンごとハードフリーズして未完。この時点でいったん中断。**

| Phase | 状態 | 備考 |
|---|---|---|
| 1. 環境構築と疎通確認 | ✅ 完了 | `scripts/check_env.py` |
| 2. 最小推論（t2v 短尺） | ⚠️ **中断** | text encode ✅ / denoise ✅ / **VAE decode でシステム全体がフリーズ** |
| 3. CLI化 | ❌ 未着手 | `src/ltx25_cli/` は空の`__init__.py`のみ |
| 4. 検証と最適化 | ❌ 未着手 | 計測の枠組み（`Stage`クラス）だけは Phase 2 に実装済み |

Gitは**未コミット**（`master`にコミットゼロ、全ファイルuntracked）。

### 動いているもの

インストール済みバージョン（`.venv`, Python 3.12）:

```
torch         2.11.0+rocm7.13.0     # repo.amd.com/rocm/whl/gfx1151/ から
diffusers     0.40.0.dev0           # github.com/huggingface/diffusers (main)
transformers  5.15.0
torch.cuda.is_available() → True / Radeon 8060S Graphics
```

モデル: `models/ltx-2.5-diffusers/` に67GB取得済み（`Lightricks/LTX-2.5-Diffusers`、`LTX2Pipeline`レイアウト）。`scripts/download_models.py`が重複シャードと`transformer_full/`を除外して落とす。

`scripts/t2v_smoke.py`（768x512 / 121フレーム / 24fps / distilled 8ステップ）の実測:

```
[generate:load]         3.0s   peak VRAM  0.0 GiB
[generate:offload:none] 21.3s  peak VRAM 42.97 GiB
[generate:denoise]      109.4s peak VRAM 44.93 GiB   ← 8 steps, 約13s/it
[generate:release]      0.3s   peak VRAM 43.01 GiB
[generate:decode]       ← ここでマシンごと落ちる
```

text embeddings は `outputs/t2v_smoke.embeds.pt`（770MB）にキャッシュ済み。プロンプトを変えなければ再エンコード不要。

## ブロッカー: VAE decode でハードフリーズ

`run.log` の記録は `[generate:decode] ...` の直後で途切れている。Pythonの例外ではなく**マシンごとのハードフリーズ**。根拠:

1. `run.log` の末尾764バイトがNULバイト — ファイルサイズは伸びたのにデータブロックがディスクに届いていない = fsync前の強制リセットの痕跡。例外終了ならトレースバックが残る。
2. `journalctl --list-boots` で前ブートが `11:14:33` に**shutdownシーケンスなしで**途切れ、`11:20:05` に再起動。
3. OOM killerの記録も amdgpu の GPU reset / ring timeout も**なし** — カーネルがハンドラを走らせる前に固まった。
4. フリーズ直前のカーネルメッセージ:
   ```
   11:13:45 kernel: workqueue: svm_range_deferred_list_work [amdgpu] hogged CPU for >10000us 5 times
   ```
   SVM = ユニファイドメモリのレンジ管理。VRAM↔システムRAM のページング圧がかかっていた合図。

原因は untiled な VAE decode（121x512x768 の conv3d）。VRAM 48GB に対しシステムRAM 45GB / GTT 31GB しかないため、はみ出した瞬間 OOM killer が間に合わずロックする。`t2v_smoke.py:204` のコメントが「untiled conv3d over 121x512x768 then asks for another 9.65 GiB」と予告していたとおり。

**悪化要因**: MIOpen に gfx1151 の perf DB が存在しない。

```
$ ls .venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1151/share/miopen/db/ | grep gfx11
（何も出ない — gfx908 / gfx90a / gfx942 のみ）
$ grep MIOpen run.log
MIOpen(HIP): Warning [ParseAndLoadDb] File is unreadable: ".../gfx1151_20.HIP.fdb.txt"
```

gfx1151は毎回アルゴリズム探索にフォールバックし、候補ごとにワークスペースを確保する。transformerはattention/linear主体でconvを通らないため、**このrunで最初にconv3dを叩いたのがdecode段** — 落ちた場所と一致する。

### 再開するときの手順（未検証）

1. **`--vae-tiling` を付ける。** 768x512なら latent 24x16 に対し `tile_latent_min_width = 512//32 = 16` なので `24 > 16` で空間タイリングは発動する（`autoencoder_kl_ltx2.py:1281`）。
2. **時間方向も切る。** `enable_tiling()` は `use_framewise_decoding` を触らない（`autoencoder_kl_ltx2.py:1174` で `False`）ので、このままだと121フレームが丸ごと一発でdecodeされる。`pipe.vae.use_framewise_decoding = True` を併せて立てる。
3. **denoise後にlatentsをディスクへ落とす。** 現状はdecodeで落ちるたびに109秒のdenoiseをやり直しになる。embedsと同じくキャッシュすべき。
4. `MIOPEN_FIND_MODE=FAST` と書き込み可能な `MIOPEN_USER_DB_PATH` を設定し、探索時のワークスペース確保を抑える。
5. それでも駄目ならフレーム数を落として（例: 121 → 49）decodeの絶対量を下げ、切り分ける。

## 実行環境

- ハードウェア: GMKtec NucBox EVO X2（Ryzen AI MAX+ 395, gfx1151, 96GB統合メモリ）
  - 内訳: BIOSで**48GBをVRAMに固定割当**、残り約45GBがOS側システムRAM（別途GTT 31GB）。VRAMに載せる場合の上限は96GBではなく48GB
- OS: Ubuntu 26.04（resolute）
- GPU stack: ROCm 7.14.0（`/opt/rocm`）。ただし torch は自前のROCm 7.13ランタイムを同梱している
- `HSA_OVERRIDE_GFX_VERSION`は**設定しない**（ネイティブgfx1151ビルドなのでarch上書きは動作を壊す）
- Python: `uv`管理のvenv、3.12固定（`.python-version`）

## ファイル構成

```
scripts/check_env.py       Phase 1 smoke test（ROCm疎通 + チェックポイントロード）
scripts/download_models.py 選択的ダウンロード（重複シャード / transformer_full を除外）
scripts/t2v_smoke.py       Phase 2 smoke test（encode/generate 二段構成）★現在の主戦場
src/ltx25_cli/__init__.py  空。CLI本体は未着手
outputs/t2v_smoke.embeds.pt  キャッシュ済みprompt embeddings（770MB）
run.log                    フリーズしたrunのログ（末尾NUL）
```

`models/`・`outputs/`・`*.mp4` は`.gitignore`済み。

## これまでに判明した実装上の要点

`t2v_smoke.py` のdocstringとコメントに詳細があるが、要点は以下。**どれも実際に踏んだ穴**。

- **チェックポイントはどこにも丸ごと載らない。** bf16で67GiB、対してシステムRAM 45GiB / VRAM 48GiB。そこで2フェーズに分割した:
  - `encode`: tokenizer + text_encoder のみ（23GiB）→ prompt embeddings をディスクへ
  - `generate`: text_encoder以外（43GiB）→ mp4
  - `both`（デフォルト）は encode をサブプロセスで回す。23GiBがアロケータの善意ではなくプロセス終了でOSに戻る上、generateが失敗しても再エンコードが要らない。
- **オフロードは使わない（`--offload none`が正解）。** 43GiBを45GiBのシステムRAMに載せる構成こそが最初にマシンを落とした原因。`enable_model_cpu_offload()` は RSS 43.3GiB + swap 6.3GiB まで行き、`amdgpu: SVM mapping failed, exceeds resident system memory limit` でカーネルログを埋めてgnome-terminal-serverごと落ちた。
- **denoise後にtransformer/connectorsをNoneにして解放する。** これがVAEにdecodeの余地を作る（`LTX2Pipeline.__init__`は各コンポーネントをガードしているのでNoneは正当な状態）。
- **`encode_prompt` には自前で `torch.no_grad()` が要る。** `@torch.no_grad()` が付いているのは `__call__` と `enhance_prompt` だけ。単体で呼ぶとGemma 48層ぶんのautogradグラフ（約24GiB）が積まれて48GBカードがOOMする。
- **distilled推論は `sigmas=DISTILLED_SIGMA_VALUES` で駆動する。** `num_inference_steps` を渡すと汎用の線形スケジュールになり、静かに品質を落とす。guidance系は全部off（`guidance_scale=1.0`, `stg_scale=0.0` 等）。
- **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` が必要。** 無いとROCmがmemory-efficient SDPAカーネルを報告せず、mathバックエンドにフォールバックして `[B, heads, S, S]` を実体化する（この解像度で1回のattentionに4.5GiB）。
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。** 43GiBを48GiBに詰めるので、アロケータがセグメント間に取り残す約1.5GiBが収まるかどうかの分かれ目になる。
- **音声テンソルは `.float().cpu()` してからmuxerに渡す。** GPU上のbf16のまま渡すのが ComfyUI-LTXVideo #361 のクラッシュ。
- **長時間runは端末から切り離す。** 最初のクラッシュは端末ごと道連れにした。
  ```bash
  nohup .venv/bin/python scripts/t2v_smoke.py > run.log 2>&1 &
  ```
  ただし今回のフリーズはマシン全体のロックなので、nohupでは守れない。

## 当初の計画（残り）

### Phase 3: CLI化

1. `argparse`または`click`でサブコマンド構成:
   - `generate t2v --prompt "..." --width --height --frames --output`
   - `generate i2v --image path --prompt "..." --output`
   - `generate interpolate --image-start path --image-end path --output`（アハ体験用、対応パイプラインの有無を要調査）
2. 設定ファイル（YAML）でデフォルトパラメータ（解像度、フレーム数、ステップ数、ネガティブプロンプト等）を管理
3. モデルパスやROCm関連の環境変数は設定ファイルで一元管理。`HSA_OVERRIDE_GFX_VERSION`は設定しないこと

### Phase 4: 検証と最適化

1. 生成時間・メモリ使用量を計測（`Stage`クラスで実装済み、`*.metrics.json`に吐く）
2. distilledパイプライン（8ステップ）とTwoStagesパイプライン（高品質・低速）の両対応を検討
3. 必要に応じてFP8量子化（`quantization=QuantizationPolicy.fp8_cast()`）を試す

## 非ゴール（今回は対応しない）

- ComfyUIとの連携・ノード化
- LoRAトレーニング（ltx-trainerパッケージ）
- FP8量子化などの高度な最適化（Phase 4で検討）

## 既知のリスク・要検証事項

| 項目 | 状態 | 内容 |
|---|---|---|
| VAE decodeでのハードフリーズ | 🔴 **未解決・現在のブロッカー** | 上記「ブロッカー」節を参照。tiling + framewise decoding が次の一手 |
| MIOpenにgfx1151のperf DBが無い | 🔴 未解決 | conv3dのたびにアルゴリズム探索。速度とワークスペース両方に効く |
| `enable_sequential_cpu_offload` | ✅ 解決（不採用） | オフロード系は45GiBのシステムRAMに収まらず、これがクラッシュ源だった。`--offload none`で運用 |
| メモリオフロード戦略 | ✅ 解決 | 2フェーズ分割（encode/generate）＋denoise後のtransformer解放で対応 |
| distilled pipelineのdiffusers対応 | ✅ 解決 | `diffusers.pipelines.ltx2.utils.DISTILLED_SIGMA_VALUES` が存在。8ステップ実測109秒 |
| 音声テンソルのCPU転送忘れ | ⚪️ 未到達 | 実装済みだが、decodeに到達していないため未検証 |
| KeyframeInterpolationPipeline | ⚪️ 未調査 | 「アハ体験」用途に最適か未検証。無ければRetakePipelineやIC-LoRA系で代替検討 |

## 参考リンク

- https://github.com/Lightricks/LTX-2
- https://huggingface.co/Lightricks/LTX-2.5
- https://huggingface.co/Lightricks/LTX-2.5-Diffusers
- https://github.com/Lightricks/ComfyUI-LTXVideo/issues/361（AMD動作確認・音声バグ報告）
- https://docs.ltx.io/open-source-model/usage-guides/prompting-guide

## 完了の定義（Definition of Done）

- [x] `uv`ベースのvenvで独立した環境を構築、`torch.cuda.is_available()` が True
- [x] LTX-2.5のdistilledモデル一式をダウンロード
- [x] `LTX2Pipeline.from_pretrained` でロードでき、denoiseが完走する
- [ ] **mp4（音声付き）が1本でも書き出せる** ← ここで止まっている
- [ ] `t2v` で動画が生成できる
- [ ] `i2v` で画像を起点とした動画が生成できる
- [ ] ROCm環境（gfx1151）でクラッシュせず最後まで実行できる
- [ ] 生成時間・メモリ使用量のログが記録される（枠組みは実装済み）
- [ ] README.mdに使い方とインストール手順を記載（現在は空ファイル）
