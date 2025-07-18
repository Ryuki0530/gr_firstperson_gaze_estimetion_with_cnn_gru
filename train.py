import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.gtea_loader import GTEADataset
from models.gaze_model import GazeEstimationModel
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt  # ← 可視化ライブラリ

# 引数処理
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
args = parser.parse_args()

# ハイパーパラメータ
BATCH_SIZE = 32
EPOCHS = args.epochs
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データローダー
train_dataset = GTEADataset("data/processed/gtea/train_split.npz")
val_dataset   = GTEADataset("data/processed/gtea/val_split.npz")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# モデル構築
model = GazeEstimationModel().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 状態表示
print(f"\n\n使用デバイス: {DEVICE}")
print(f"学習エポック数: {EPOCHS}")
print(f"バッチサイズ: {BATCH_SIZE}")

# 可視化用ロス記録リスト
train_losses = []
val_losses = []

# 描画用関数
def plot_losses(train_losses, val_losses, save_path="loss_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 学習ループ
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
    for batch_x, batch_y in tqdm(train_loader, desc="Training", leave=False):
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / len(train_loader.dataset)

    # 検証
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader, desc="Validation", leave=False):
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            val_loss += loss.item() * batch_x.size(0)

    val_loss /= len(val_loader.dataset)

    train_losses.append(avg_loss)
    val_losses.append(val_loss)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 毎エポックごとにグラフ更新
    plot_losses(train_losses, val_losses)

# モデル保存
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/gaze_estimation_model.pth")
print("✅ モデル保存完了: checkpoints/gaze_estimation_model.pth")

plt.ioff()
plot_losses(train_losses, val_losses)
print("📈 ロス曲線を loss_plot.png に保存しました。")
