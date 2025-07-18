import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import random

SEQ_LEN = 6
IMAGE_SIZE = (224, 224)
MAX_SEQ_PER_VIDEO = 1000

def read_gaze_labels(txt_path):
    gaze_map = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) < 6: continue
            try:
                frame_id = int(parts[5])
                x = float(parts[3])
                y = float(parts[4])
                gaze_map[frame_id] = (x, y)
            except:
                continue
    return gaze_map

def process_video(video_path, label_path):
    cap = cv2.VideoCapture(video_path)
    gaze_map = read_gaze_labels(label_path)

    frames = []
    valid_gaze = []
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id not in gaze_map:
            continue
        resized = cv2.resize(frame, IMAGE_SIZE)
        frames.append(resized)
        gaze_xy = gaze_map[frame_id]
        gaze_norm = (gaze_xy[0] / W, gaze_xy[1] / H)
        valid_gaze.append(gaze_norm)
    cap.release()

    # ランダムな位置から最大 MAX_SEQ_PER_VIDEO 件を抽出
    sequences = []
    labels = []
    total_possible = len(frames) - SEQ_LEN + 1
    if total_possible < 1:
        return np.array([]), np.array([])
    
    indices = list(range(total_possible))
    random.shuffle(indices)
    for i in indices[:min(total_possible, MAX_SEQ_PER_VIDEO)]:
        seq = frames[i:i+SEQ_LEN]
        label = valid_gaze[i+SEQ_LEN-1]
        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)

def main(input_dir, output_path, seed=None):
    if seed is not None:
        print(f"[INFO] Setting random seed: {seed}")
        random.seed(seed)

    all_sequences = []
    all_labels = []

    video_paths = glob(os.path.join(input_dir, "*.avi"))
    for video_path in tqdm(video_paths, desc="Processing videos"):
        base = os.path.splitext(os.path.basename(video_path))[0]
        label_path = os.path.join(input_dir, base + ".txt")
        if not os.path.exists(label_path):
            print(f"[WARN] Missing label file for {video_path}")
            continue
        try:
            seqs, labels = process_video(video_path, label_path)
            if len(seqs) == 0:
                continue
            all_sequences.append(seqs)
            all_labels.append(labels)
        except Exception as e:
            print(f"[ERROR] Failed to process {base}: {e}")
            continue

    all_sequences = np.concatenate(all_sequences, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    np.savez_compressed(output_path, images=all_sequences, gaze=all_labels)
    print(f"[DONE] Saved {len(all_sequences)} sequences to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="data/raw/gtea")
    parser.add_argument('--output', type=str, default="data/processed/gtea/train.npz")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (optional)")
    args = parser.parse_args()
    main(args.input_dir, args.output, seed=args.seed)
