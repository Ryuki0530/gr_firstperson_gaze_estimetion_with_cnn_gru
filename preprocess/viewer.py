import numpy as np
import matplotlib.pyplot as plt

# データロード
data = np.load("data/processed/gtea/train_split.npz")
images = data["images"]
gazes = data["gaze"]

print(f"Total sequences: {len(images)}")
print(f"Sequence shape: {images.shape}")
print(f"Gaze shape: {gazes.shape}")

def view_sequence(sample_idx):
    """指定されたサンプルの6フレームシーケンスを表示"""
    if sample_idx >= len(images):
        print(f"Error: sample_idx {sample_idx} is out of range (max: {len(images)-1})")
        return
    
    sequence = images[sample_idx]  # (6, 224, 224, 3)
    gaze = gazes[sample_idx]       # (2,)
    
    # 2x3のサブプロットを作成
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Sample {sample_idx}: 6-frame sequence", fontsize=16)
    
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # BGR → RGB変換
        img_rgb = sequence[i][..., ::-1]
        ax.imshow(img_rgb)
        ax.set_title(f"Frame {i+1}")
        ax.axis('off')
        
        # 最後のフレーム（6枚目）に視線座標を表示
        if i == 5:  # 最後のフレーム
            h, w = img_rgb.shape[:2]
            gx = int(gaze[0] * w)
            gy = int(gaze[1] * h)
            ax.scatter([gx], [gy], color='red', s=100, marker='x', linewidths=3)
            ax.set_title(f"Frame {i+1} (Gaze: {gaze[0]:.3f}, {gaze[1]:.3f})", color='red')
    
    plt.tight_layout()
    plt.show()

def interactive_viewer():
    """インタラクティブなビューア"""
    while True:
        try:
            user_input = input(f"\nEnter sample index (0-{len(images)-1}) or 'q' to quit: ")
            if user_input.lower() == 'q':
                break
            
            sample_idx = int(user_input)
            view_sequence(sample_idx)
            
        except ValueError:
            print("Please enter a valid number or 'q'")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

# 最初のサンプルを表示
print("\nShowing first sample:")
view_sequence(0)

# インタラクティブモード開始
print("\n" + "="*50)
print("Interactive mode - you can view any sample")
interactive_viewer()
