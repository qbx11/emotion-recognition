import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import mediapipe as mp


# CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"

CAMERA_ID = 0
EMA_ALPHA = 0.7
CONFIDENCE_THRESHOLD = 0.15
TEMPERATURE = 1.5

# OpenCV uses BGR color space (Blue, Green, Red)
EMOTION_COLORS = {
    "Angry": (0, 0, 255),  # Red
    "Fear": (128, 0, 128),  # Purple
    "Happy": (0, 255, 0),  # Green
    "Sad": (255, 0, 0),  # Blue
    "Surprise": (0, 255, 255),  # Yellow
    "Neutral": (200, 200, 200),  # Light Gray
    "Uncertain": (100, 100, 100)  # Dark Gray
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_NAMES = {
    0: "Angry",
    1: "Fear",
    2: "Happy",
    3: "Sad",
    4: "Surprise",
    5: "Neutral"
}


# MODEL ARCHITECTURE
class EmotionCNN(nn.Module):


    def __init__(self, num_classes=6):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
            block(256, 256)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))



# MODEL LOADING

def load_model():
    """
    Loads the trained model weights from the disk.
    """
    if not MODEL_PATH.exists():
        print(f"ERROR: Model file not found at: {MODEL_PATH}")
        print("Ensure 'best_model.pth' is in the same directory as the script.")
        sys.exit(1)

    model = EmotionCNN(num_classes=len(EMOTION_NAMES))
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"ERROR loading model weights: {e}")
        sys.exit(1)

    model.to(DEVICE)
    model.eval()
    return model


def draw_probability_overlay(frame, probs, emotion_names, x=10, y=10, width=220):
    if probs is None:
        return

    overlay = frame.copy()
    line_h = 26
    height = line_h * len(emotion_names) + 10

    # Draw background rectangle (dark)
    cv2.rectangle(
        overlay,
        (x, y),
        (x + width, y + height),
        (0, 0, 0),
        -1
    )

    # Apply alpha blending for transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Render text for each emotion
    for i, (idx, name) in enumerate(emotion_names.items()):
        p = probs[idx]
        text = f"{name:<8} {p * 100:5.0f}%"

        # Highlight the dominant emotion
        text_color = (255, 255, 255)
        if idx == np.argmax(probs):
            text_color = EMOTION_COLORS.get(name, (255, 255, 255))

        cv2.putText(
            frame,
            text,
            (x + 10, y + 25 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
            cv2.LINE_AA
        )


# PREPROCESSING
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Standard transformation pipeline matching training configuration
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


def preprocess_face(face_bgr):
    """
    Prepares the cropped face image for model inference.
    """
    try:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        # Convert back to RGB because the model expects 3 channels
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(rgb)
        return transform(img).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None


# FACE NORMALIZATION
def extract_face(frame, cx, cy, bw, bh, scale=1.3):
    """
    Extracts the Region of Interest (ROI) for the face.
    """
    h, w, _ = frame.shape
    size = int(max(bw, bh) * scale)

    # Calculate coordinates ensuring they stay within image bounds
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w, cx + size // 2)
    y2 = min(h, cy + size // 2)

    face_img = frame[y1:y2, x1:x2]
    return face_img, (x1, y1, x2, y2)


# MAIN APP
def main():
    print("Loading model...")
    model = load_model()
    softmax = nn.Softmax(dim=1)

    # Initialize MediaPipe Face Detection
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera with ID {CAMERA_ID}")

    ema_probs = None
    print("Camera started — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe requires RGB input
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        if result.detections:
            for det in result.detections:
                # Convert relative bounding box to absolute pixel coordinates
                box = det.location_data.relative_bounding_box
                h, w, _ = frame.shape

                cx = int((box.xmin + box.width / 2) * w)
                cy = int((box.ymin + box.height / 2) * h)
                bw = int(box.width * w)
                bh = int(box.height * h)

                face, (x1, y1, x2, y2) = extract_face(frame, cx, cy, bw, bh)

                if face.size == 0:
                    continue

                inp = preprocess_face(face)
                if inp is None:
                    continue

                inp = inp.to(DEVICE)

                # Run inference
                with torch.no_grad():
                    logits = model(inp) / TEMPERATURE
                    probs = softmax(logits).cpu().numpy()[0]

                # Apply Exponential Moving Average (EMA) to reduce jitter in predictions
                ema_probs = probs if ema_probs is None else EMA_ALPHA * ema_probs + (1 - EMA_ALPHA) * probs

                pred_idx = int(np.argmax(ema_probs))
                pred_conf = float(ema_probs[pred_idx])

                # Determine final label based on confidence threshold
                if pred_conf < CONFIDENCE_THRESHOLD:
                    main_label = "Uncertain"
                    color = EMOTION_COLORS["Uncertain"]
                else:
                    label_name = EMOTION_NAMES[pred_idx]
                    main_label = f"{label_name} ({pred_conf * 100:.0f}%)"
                    color = EMOTION_COLORS.get(label_name, (255, 255, 255))

                # Visualisation
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, main_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

                draw_probability_overlay(
                    frame,
                    ema_probs,
                    EMOTION_NAMES,
                    x=10,
                    y=10
                )

                # Process only the first detected face to maintain performance and simplify UI
                break
        else:
            # Reset EMA history if no face is detected to avoid stale predictions
            ema_probs = None

        cv2.imshow("Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
            break

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
if __name__ == "__main__":
    main()