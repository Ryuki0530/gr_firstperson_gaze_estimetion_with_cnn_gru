import os
import cv2
import torch
import numpy as np
from collections import deque
from models.gaze_model import GazeEstimationModel

# モデルの設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 6
IMAGE_SIZE = (224, 224)

# モデルの読み込み
model = GazeEstimationModel()
model.load_state_dict(torch.load("checkpoints/gaze_estimation_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# フレームを保持するキュー
frame_buffer = deque(maxlen=SEQ_LEN)

# 動画ファイルを選択
video_path = "data/raw/gtea/Ahmad_American.avi"  # 任意の動画に差し替え可
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, IMAGE_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor_frame = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    frame_buffer.append(tensor_frame)

    if len(frame_buffer) == SEQ_LEN:
        input_seq = torch.stack(list(frame_buffer)).unsqueeze(0).to(DEVICE)  # (1, 6, 3, 224, 224)
        with torch.no_grad():
            pred = model(input_seq).cpu().numpy()[0]  # [x, y] in normalized [0, 1]

        # 視線位置を元画像サイズに変換
        gx = int(pred[0] * frame_width)
        gy = int(pred[1] * frame_height)

        # 可視化
        vis_frame = frame.copy()
        cv2.circle(vis_frame, (gx, gy), radius=10, color=(0, 0, 255), thickness=-1)
        cv2.imshow("Gaze Prediction", vis_frame)

    # qキーで中断
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
