import numpy as np

# 元データの読み込み
data = np.load("data/processed/gtea/train.npz")
images = data["images"]  # shape = (N, 6, 224, 224, 3)
gaze = data["gaze"]      # shape = (N, 2)

# データ数
num_samples = images.shape[0]
val_ratio = 0.2
val_size = int(num_samples * val_ratio)

# シャッフルして分割
indices = np.arange(num_samples)
np.random.seed(42)  # 任意の固定シード
np.random.shuffle(indices)

val_indices = indices[:val_size]
train_indices = indices[val_size:]

train_images = images[train_indices]
train_gaze = gaze[train_indices]
val_images = images[val_indices]
val_gaze = gaze[val_indices]

# 保存
np.savez_compressed("data/processed/gtea/train_split.npz", images=train_images, gaze=train_gaze)
np.savez_compressed("data/processed/gtea/val_split.npz", images=val_images, gaze=val_gaze)

print("✅ train_split.npz / val_split.npz を作成しました")
