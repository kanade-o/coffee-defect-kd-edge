# coffee-defect-kd-edge

高専本科5年と専攻科で取り組んだ、深層学習と知識蒸留（Knowledge Distillation）を用いた、エッジコンピュータ（Raspberry Pi 5）向けコーヒー生豆欠陥検出システムの研究リポジトリです。

大規模な教師モデル（Swin Transformer Small, ResNet-50 など）から軽量な生徒 CNN へ知識蒸留を行い、Raspberry Pi 5 上でリアルタイム推論（約 9ms/枚, 約 110FPS）を実現しています。独自に構築した 72,000 枚規模のコーヒー生豆画像データセットを使用しています。

## 環境構築

- Python 3.13.3（`.python-version` で指定）
- 依存パッケージのインストール:

```bash
pip install -r requirements.txt
```

> **注意**: PyTorch は CUDA 11.8 向けビルド（`torch==2.7.0+cu118`）が `requirements.txt` に記載されています。環境に合わせて適宜変更してください。

---

## ディレクトリ構成

```
.
├── README.md                 # 本ファイル
├── .python-version           # Python バージョン指定（3.13.3）
├── requirements.txt          # Python 依存パッケージ一覧
├── configs/                  # Hydra 設定ファイル群
├── src/                      # ソースコード（学習・蒸留・ユーティリティ）
├── eval/                     # エッジデバイス（Raspberry Pi 5）上での評価スクリプト・結果
├── outputs/                  # 学習実行時の出力（ログ・モデル重み・グラフ等）
└── paper/                    # 論文・発表資料
```

---

## configs/ — Hydra 設定ファイル

[Hydra](https://hydra.cc/) を用いて、モデル・データセット・学習条件を YAML ファイルで管理しています。

```
configs/
├── config.yaml                    # 通常学習用メイン設定（train.py 用）
├── config_kd_cnn224.yaml          # 知識蒸留用設定: CNN224 生徒 + Swin 教師（AT+KD）
├── config_kd_cnn_student050.yaml  # 知識蒸留用設定: cnn_student（軽量版）+ Swin 教師（AT+KD）
├── config_kd_cnn_student100.yaml  # 知識蒸留用設定: cnn_student100 + Swin 教師（AT+KD）
├── config_kd_resnet.yaml          # 知識蒸留用設定: cnn_student100 + ResNet-50 教師（AT+KD, 複数層）
├── config_kd_resnet_soft.yaml     # 知識蒸留用設定: cnn_student100 + ResNet-50 教師（ソフトラベルのみ）
├── config_kd_soft.yaml            # 知識蒸留用設定: cnn_student100 + Swin 教師（ソフトラベルのみ）
├── data/                          # データセット設定
│   ├── bean_224.yaml              # 224×224 コーヒー生豆データセット（train/val/test パス・入力サイズ）
│   └── bean_180.yaml              # 180×180 コーヒー生豆データセット（train/val パス・入力サイズ）
├── model/                         # モデル定義設定（Hydra instantiate 用）
│   ├── cnn.yaml                   # 自作 CNN（180×180 入力）
│   ├── cnn224.yaml                # 自作 CNN（224×224 入力）
│   ├── cnn_student050.yaml        # 軽量生徒 CNN
│   ├── cnn_student100.yaml        # 生徒 CNN（3 ブロック構成）
│   ├── resnet18.yaml              # ResNet-18（timm 事前学習済み）
│   ├── resnet50.yaml              # ResNet-50（torchvision 事前学習済み）
│   ├── swin_b.yaml                # Swin Transformer Base
│   ├── swin_s.yaml                # Swin Transformer Small
│   ├── swin_t.yaml                # Swin Transformer Tiny
│   ├── deit_b.yaml                # DeiT Base
│   ├── deit_bd.yaml               # DeiT Base Distilled
│   ├── deit_s.yaml                # DeiT Small
│   ├── efficientnet_b0.yaml       # EfficientNet-B0
│   ├── mobilenetv2.yaml           # MobileNetV2
│   ├── mobilenetv2_050_np.yaml    # MobileNetV2 (width=0.5, 事前学習なし)
│   ├── mobilenetv2_np.yaml        # MobileNetV2（事前学習なし）
│   ├── tiny_vit.yaml              # TinyViT
│   ├── tiny_vit_np.yaml           # TinyViT（事前学習なし）
│   ├── vit.yaml                   # ViT
│   ├── vit_b16.yaml               # ViT-B/16
│   ├── vit_second.yaml            # ViT（別構成）
│   ├── vit_third.yaml             # ViT（別構成）
│   └── test.yaml                  # テスト用設定
└── train/                         # 学習条件設定
    ├── base.yaml                  # 基本設定（epochs: 100）
    ├── batchsize_finder.yaml      # バッチサイズ探索用
    ├── find_lr.yaml               # 学習率探索用（epochs: 100, lr: 0.001）
    ├── test.yaml                  # テスト用設定
    ├── test2.yaml
    └── test3.yaml
```

### 設定の仕組み

- `config.yaml` が通常学習（`src/train.py`）のメイン設定で、`defaults` で `data`, `train`, `model` を組み合わせます。
- `config_kd_*.yaml` は知識蒸留用の設定で、教師モデル（`teacher`）と蒸留パラメータ（`kd`）を追加で定義しています。
- 蒸留設定の `kd` セクションでは以下を指定します:
  - `T`: 温度パラメータ（ソフトラベルの滑らかさ）
  - `alpha`: KL ダイバージェンス損失の重み
  - `beta`: Attention Transfer (AT) 損失の重み（AT 使用時）
  - `teacher_feat` / `student_feat`: AT で比較する中間層の名前

---

## src/ — ソースコード

学習・知識蒸留・ユーティリティスクリプトが含まれます。

```
src/
├── train.py                  # 通常モデル学習スクリプト（cosine スケジューラ付き）
├── train_simple.py           # 通常モデル学習スクリプト（スケジューラなし、シンプル版）
├── train_distill.py          # 知識蒸留学習スクリプト（AT + ソフトラベル KD）
├── utils.py                  # 評価指標計算・ROC/PR 曲線描画ユーティリティ
├── lr_finder.py              # Optuna を用いた学習率・weight_decay の自動探索
├── batchsize_finder.py       # PyTorch Lightning Tuner を用いた最大バッチサイズ探索
├── batchsize_finder_kd.py    # 蒸留時のバッチサイズ探索（教師+生徒の VRAM を考慮）
├── download_models.py        # timm 事前学習済みモデルのダウンロード・保存
├── download_models_no_pre.py # timm モデルの重み保存（事前学習なし）
├── normalize_finder.py       # データセットの Mean/Std 値を算出するスクリプト
├── normalize_value.txt       # 算出した正規化値の記録
├── view_model_detail.py      # モデルの計算量（GMAC）・パラメータ数（M）を一覧表示
├── models/                   # モデル定義
│   ├── cnn.py                # 自作 CNN（180×180 入力対応）
│   ├── cnn_224.py            # 自作 CNN（224×224 入力対応）
│   ├── cnn_student050.py     # 軽量生徒 CNN（チャンネル数半分）
│   ├── cnn_student100.py     # 生徒 CNN（3 ブロック, 8→16→32ch）— 最終的なエッジ推論用モデル
│   ├── resnet18.py           # ResNet-18 ラッパー（timm 事前学習済み重みロード）
│   ├── resnet50.py           # ResNet-50 ラッパー（torchvision 事前学習済み重みロード）
│   ├── swin_b.py             # Swin Transformer Base ラッパー
│   ├── swin_s.py             # Swin Transformer Small ラッパー — 教師モデルとして使用
│   ├── swin_t.py             # Swin Transformer Tiny ラッパー
│   ├── deit_b.py             # DeiT Base ラッパー
│   ├── deit_bd.py            # DeiT Base Distilled ラッパー
│   ├── deit_s.py             # DeiT Small ラッパー
│   ├── efficientnet_b0.py    # EfficientNet-B0 ラッパー
│   ├── mobilenetv2.py        # MobileNetV2 ラッパー
│   ├── mobilenetv2_050_np.py # MobileNetV2 (width=0.5, 事前学習なし)
│   ├── mobilenetv2_np.py     # MobileNetV2（事前学習なし）
│   ├── tiny_vit.py           # TinyViT ラッパー
│   ├── tiny_vit_np.py        # TinyViT（事前学習なし）
│   ├── vit_b16_pretrained.py # ViT-B/16 ラッパー
│   ├── timm_cnn.py           # timm ベース CNN ユーティリティ
│   ├── timm_model.py         # timm ベースモデル汎用ラッパー
│   └── old/                  # 過去バージョンのモデル定義（参考用）
└── kd/                       # 知識蒸留用学習スクリプト群
    ├── kd_soft.py            # ソフトラベル KD のみ（Swin 教師 → cnn_student100）
    ├── kd_resnet.py          # AT + KD（ResNet-50 教師 → cnn_student100, 複数層対応）
    ├── kd_resnet_soft.py     # ソフトラベル KD のみ（ResNet-50 教師 → cnn_student100）
    ├── kd_cnn_224.py         # AT + KD（Swin 教師 → CNN224）
    ├── kd_cnn_student100.py  # AT + KD（Swin 教師 → cnn_student100）
    └── kd_cnn_student100_simple.py # KD 簡易版（cnn_student100）
```

### 主要スクリプトの説明

#### `src/train.py` — 通常学習

Hydra で設定を読み込み、指定したモデルを train/val/test データで学習します。Cosine スケジューラ（10% ウォームアップ）付き。学習終了後にテストデータで Accuracy, Precision, Recall, F1, AUC を算出し、ROC 曲線・PR 曲線を保存します。

#### `src/train_simple.py` — 通常学習（スケジューラなし）

`train.py` のスケジューラを除いたシンプル版。固定学習率で学習を行います。

#### `src/train_distill.py` — 知識蒸留学習

Attention Transfer (AT) とソフトラベル KD を組み合わせた蒸留学習スクリプト。教師モデルの中間特徴マップと生徒モデルの中間特徴マップを Attention Map に変換し MSE で比較します。

#### `src/kd/kd_soft.py` — ソフトラベル KD

中間層の AT を使わず、ロジットレベルのソフトラベル KD のみを行うスクリプト。`(1-α)*CE + α*KD` の損失関数で学習します。

#### `src/kd/kd_resnet.py` — ResNet-50 教師による複数層 AT+KD

ResNet-50 を教師とし、`kd.pairs` で指定した複数の中間層ペアに対して AT 損失を計算します。

#### `src/utils.py` — 評価ユーティリティ

テストデータに対する Accuracy, Precision, Recall, F1, AUC の計算と、ROC 曲線・Precision-Recall 曲線の描画を行います。

#### `src/lr_finder.py` — 学習率探索

Optuna を用いて学習率（`lr`）と weight_decay のハイパーパラメータ最適化を行います。F1 スコアを最大化する方向で 30 トライアル実行します。

#### `src/batchsize_finder.py` — バッチサイズ探索

PyTorch Lightning の Tuner を使い、GPU メモリに収まる最大バッチサイズを二分探索で求めます。

#### `src/batchsize_finder_kd.py` — 蒸留時バッチサイズ探索

通常学習と蒸留学習（教師+生徒の両方が VRAM を消費）の両方で最大バッチサイズを探索し、公平比較のための共通バッチサイズを決定します。

#### `src/normalize_finder.py` — 正規化値算出

データセット全体の Mean/Std を算出します。結果は `normalize_value.txt` に記録されています。

#### `src/view_model_detail.py` — モデル情報一覧

ptflops を使い、各モデルの計算量（GMAC）とパラメータ数（M）を一覧表示します。モデル選定の参考として使用します。

#### `src/download_models.py` / `src/download_models_no_pre.py` — モデルダウンロード

timm の事前学習済みモデルの重みをローカルに保存するスクリプトです。`download_models_no_pre.py` は事前学習なしの重みを保存します。

---

## eval/ — エッジデバイス評価

Raspberry Pi 5 上でモデルの推論性能をベンチマークするためのスクリプトと結果が含まれます。

```
eval/
├── eval_on_rasp.py               # Raspberry Pi 5 上でのベンチマークスクリプト
├── benchmark_rpi5_results.csv    # ベンチマーク結果（レイテンシ・FPS・CPU使用率・メモリ）
├── student.pt                    # 生徒モデル（通常学習済み）の重み
├── distilled_from_swin.pt        # Swin 教師から蒸留した生徒モデルの重み
└── distilled_from_resnet.pt      # ResNet-50 教師から蒸留した生徒モデルの重み
```

`eval_on_rasp.py` は、教師モデル（Swin-Small, ResNet-50）と生徒モデル（通常学習・蒸留済み）のそれぞれに対して、ダミー入力でウォームアップ後に 100 回推論を実行し、平均レイテンシ・中央値・p95・FPS・CPU 使用率・最大メモリ使用量を計測して CSV に出力します。

---

## outputs/ — 学習出力

Hydra の出力ディレクトリとして、学習実行ごとにタイムスタンプ付きディレクトリが自動生成されます。各ディレクトリには以下が含まれます:

- `best.pt`: バリデーション精度が最良のモデル重み
- `loss_curve.png`, `accuracy_curve.png`: 学習曲線グラフ
- `roc_curve.png`, `pr_curve.png`: テスト評価の ROC / PR 曲線
- `train.log`: 学習ログ

```
outputs/
├── bean_180/                  # 180×180 データセットでの学習結果
├── bean_224/                  # 224×224 データセットでの学習結果（モデルごとにサブディレクトリ）
│   ├── cnn224/
│   ├── resnet50/
│   ├── swin_s/
│   ├── cnn_student100/
│   └── ...（各モデル名のディレクトリ）
├── batch_finder/              # バッチサイズ探索の結果
├── kd/                        # 知識蒸留の学習結果
│   ├── cnn_student100/        # cnn_student100 の蒸留結果
│   ├── resnet/                # ResNet-50 教師による蒸留結果
│   └── swin/                  # Swin 教師による蒸留結果
└── 2025-09-22/, 2025-09-24/   # 日付ごとの実行結果
```

---

## paper/ — 論文・発表資料

```
paper/
├── full_paper/                # 卒業論文（LaTeX）
│   ├── paper.tex              # メインファイル
│   ├── chapters/              # 各章の LaTeX ファイル
│   │   ├── 00_abstract.tex    # 論文要旨
│   │   ├── 01_introduction.tex # 序論
│   │   ├── 02_related_method.tex # 関連手法
│   │   ├── 03_method.tex      # 提案手法
│   │   ├── 04_dataset.tex     # データセット
│   │   ├── 05_experiment.tex  # 実験
│   │   ├── 06_discussion.tex  # 考察
│   │   ├── 07_conclusion.tex  # 結論
│   │   ├── acknowledgments.tex # 謝辞
│   │   └── references.tex     # 参考文献
│   ├── figures/               # 図表
│   ├── styles/                # LaTeX スタイルファイル
│   ├── references.bib         # BibTeX 参考文献
│   ├── .latexmkrc             # latexmk 設定
│   └── out/                   # コンパイル出力
└── final_presentation/        # 最終発表スライド（LaTeX）
    ├── template.tex           # 発表用テンプレート
    ├── jpreprint.cls          # スタイルクラス
    ├── bio.bib                # 参考文献
    ├── system.pdf             # システム構成図
    ├── cloudrain.png          # 画像素材
    └── out/                   # コンパイル出力
```

---

## 実行方法

すべてのスクリプトはリポジトリルートから実行します。

### 通常学習（train.py）

```bash
python src/train.py model=<model_name> data=<dataset_name> gpu=<gpu_id>
```

- `model`: `configs/model/` 内の YAML ファイル名（拡張子なし）を指定（例: `swin_s`, `resnet50`, `cnn_student100`）
- `data`: `configs/data/` 内の YAML ファイル名（拡張子なし）を指定（例: `bean_224`）
- `gpu`: 使用する GPU の ID を指定

**例:**

```bash
# Swin Transformer Small を 224×224 データで学習（GPU 0）
python src/train.py model=swin_s data=bean_224 gpu=0

# cnn_student100 を学習
python src/train.py model=cnn_student100 data=bean_224 gpu=0
```

### スケジューラなし通常学習（train_simple.py）

```bash
python src/train_simple.py model=<model_name> data=<dataset_name> gpu=<gpu_id>
```

### 知識蒸留（src/kd/ 配下のスクリプト）

蒸留用スクリプトはそれぞれ専用の `config_kd_*.yaml` を使用します。

```bash
# Swin 教師 → cnn_student100（ソフトラベル KD のみ）
python src/kd/kd_soft.py

# ResNet-50 教師 → cnn_student100（AT + KD, 複数層）
python src/kd/kd_resnet.py

# ResNet-50 教師 → cnn_student100（ソフトラベル KD のみ）
python src/kd/kd_resnet_soft.py

# Swin 教師 → cnn_student100（AT + KD）
python src/kd/kd_cnn_student100.py
```

> **注意**: 蒸留スクリプトでは、事前に学習済みの教師モデルの重みファイルパスを対応する `config_kd_*.yaml` 内の `teacher.weights_path` に設定する必要があります。

### 学習率探索（lr_finder.py）

```bash
python src/lr_finder.py model=<model_name> data=<dataset_name> gpu=<gpu_id>
```

Optuna で 30 トライアルの学習率・weight_decay 最適化を実行します。

### バッチサイズ探索（batchsize_finder.py）

```bash
python src/batchsize_finder.py <gpu_id>
```

スクリプト内の `model_cfgs` 辞書で対象モデルを指定します。

### 蒸留時バッチサイズ探索（batchsize_finder_kd.py）

```bash
python src/batchsize_finder_kd.py <gpu_id>
```

### 正規化値の算出（normalize_finder.py）

```bash
python src/normalize_finder.py
```

### モデル情報の表示（view_model_detail.py）

```bash
python src/view_model_detail.py
```

### 事前学習済みモデルのダウンロード

```bash
python src/download_models.py
```

### エッジデバイスでのベンチマーク（eval_on_rasp.py）

Raspberry Pi 5 上で実行します:

```bash
python eval/eval_on_rasp.py
```

`eval/` ディレクトリに `.pt` ファイル（モデル重み）を配置した状態で実行してください。結果は `eval/benchmark_rpi5_results.csv` に出力されます。

### バックグラウンドで実行したい場合

```bash
nohup sh -c "python src/train.py model=swin_s data=bean_224 gpu=0" > train.log &
```

複数の学習を順次実行する場合:

```bash
nohup sh -c "python src/train.py model=swin_s data=bean_224 gpu=0; python src/train.py model=resnet50 data=bean_224 gpu=0" > train.log &
```

> **注意**: `&` をつけないとバックグラウンド実行にならないので注意してください。
