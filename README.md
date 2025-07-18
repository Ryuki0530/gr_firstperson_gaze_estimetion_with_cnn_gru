# firstperson gaze estimetion with cnn gru

このリポジトリは、**GTEA Gaze+ データセット**を使用した主観視点映像における視線位置予測モデル（CNN + GRU）の実装です。6連続フレームから (x, y) の視線座標を予測し、学習・推論します。

---

## ディレクトリ構成

```
.
├─ checkpoints/             # 学習済みモデル保存先
├─ configs/                # 各種設定ファイル
├─ data/                   # データセット (raw, processed, 分割済)
├─ datasets/               # データローダ・前処理
├─ logs/                   # ログ
├─ models/                 # モデル定義 (CNN, GRUなど)
├─ preprocess/             # 前処理スクリプト
│
├─ eval.py                 # 評価スクリプト
├─ inference_sample.py     # AVIファイルへの推論
├─ inference_realtime.py   # Webカメラによるリアルタイム推論
├─ train.py                # 学習スクリプト
├─ requirements.txt        # 必要ライブラリ一覧
└─ README.md               # このファイル
```

---

## インストール方法

```bash
git clone https://github.com/yourusername/egocentric-gaze-estimation.git
cd egocentric-gaze-estimation
pip install -r requirements.txt
```

---

## データセットの準備

### GTEA Gaze+
- ダウンロード元: http://cbi.gatech.edu/GTEA_Gaze.html
- `.avi` 動画と `.txt` ラベルを下記に配置してください:

```
data/raw/gtea/
```

---

## 前処理

動画を `.npz` に変換するには以下を実行：

```bash
python preprocess/generate_sequences_gtea.py \
  --input_dir data/raw/gtea \
  --output data/processed/gtea/train_split.npz \
  --seed 42
```

訓練/検証分割を行う：

```bash
python preprocess/split_dataset.py
```

---

## 学習

```bash
python train.py --epochs 100
```

- MobileNetV2 を CNNエンコーダとして使用
- 6フレームの時系列を GRU で処理
- モデルは `checkpoints/` に保存
- 学習/検証ロスをリアルタイムで表示・グラフ保存 (`loss_plot.png`)

---

## ロスグラフの可視化

各エポックの学習・検証ロスを折れ線グラフで表示し、最終的に次の場所に保存されます：

```
loss_plot.png
```

---

## 推論の実行

### AVIファイルでの推論
```bash
python inference_sample.py --video_path data/raw/gtea/Alireza_Pizza.avi
```


---

##  モデル構成

- **エンコーダ**: MobileNetV2（ImageNet事前学習済）
- **時系列処理**: GRU（1層、隠れ次元=128）
- **出力**: 全結合層 → (x, y) 視線座標

```
入力:  (B, 6, 3, 224, 224)
出力: (B, 2)
```

---

## 要求ライブラリ

`requirements.txt` に記載：

```
torch
torchvision
tqdm
opencv-python
matplotlib
numpy
```

インストール：

```bash
pip install -r requirements.txt
```

---

## 問題点・考察

- 過学習の発生
GRUモデルのため学習エポック数が少なくても過学習が発生する傾向がある。
CNN単体モデル（時系列なし）に比べて、早期に検証損失が悪化する。

- 時系列情報の扱い
GTEA Gaze+ の動画は 30fps で、6フレームでは約0.2秒分の情報しか保持できず時系列の恩恵が小さい。
フレーム間の変化が小さいため、GRUが学習するべき有意義な時間的特徴が少ない可能性がある。

- GTEA Gaze+ のデータ量・多様性不足
このプロジェクトはGTEA Gaze+での学習を想定している。
このデータは少人数かつ同じ部屋で作成されているために多様性が不足している可能性がある。

- 現フレームの影響が相対的に弱くなる
入力が6フレーム分すべて同等に扱われており、最新フレーム（現時点）の情報が相対的に埋もれてしまう。
現フレームの重視度低下のデメリットが、時間軸を取り込むことのメリットを上回っている可能性がある。

- モデル構造の複雑化に対する学習効率
モデル容量が大きくなる分、少量データでは学習が安定しない。

- 推論段階での活用に対する過剰設計
実用場面でのリアルタイム推論には、高速で軽量なCNNモデルのみでも充分な精度が期待できるケースも。

---

## ライセンス

MIT License のもとで公開しています。

---

## 謝辞

- GTEA Gaze+ データセット by Georgia Tech
- MobileNetV2 モデル by TorchVision
