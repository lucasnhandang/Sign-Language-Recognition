import os
import cv2
import argparse
from model import SLR
import mediapipe as mp
import numpy as np

# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SLRApp:
    def __init__(self, model_path, few_shot_path=None):
        self.model_path = model_path
        self.few_shot_path = few_shot_path
        self.model = SLR()

        # Assume the model exists
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model.load(model_path, few_shot_path)
        else:
            raise FileNotFoundError(f"Pre-trained model not found at: {model_path}")

        # A-Z + 'del', 'nothing', 'space'
        self.classes = {i: chr(65 + i) for i in range(26)}
        self.classes[26] = 'del'
        self.classes[27] = 'nothing'
        self.classes[28] = 'space'

        self.is_collecting = False
        self.current_few_shot_label = None
        self.collected_samples = 0
        self.required_samples = 5

    def preprocess_frame(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def run(self):
        cap = cv2.VideoCapture(0)

        with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
        ) as hands:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                h, w = frame.shape[:2]
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                roi = None
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    h, w = frame.shape[:2]

                    # Center of hand
                    cx = int(np.mean([lm.x for lm in hand_landmarks.landmark]) * w)
                    cy = int(np.mean([lm.y for lm in hand_landmarks.landmark]) * h)

                    roi_size = 300
                    x1 = max(cx - roi_size // 2, 0)
                    y1 = max(cy - roi_size // 2, 0)
                    x2 = min(cx + roi_size // 2, w)
                    y2 = min(cy + roi_size // 2, h)

                    # Draw ROI bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    roi = frame[y1:y2, x1:x2]

                if roi is not None and roi.size > 0:
                    processed_roi = self.preprocess_frame(roi)

                    if self.is_collecting:
                        cv2.putText(frame,
                                    f"Collecting for '{self.current_few_shot_label}': {self.collected_samples}/{self.required_samples}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if cv2.waitKey(1) & 0xFF == ord('c'):
                            self.model.add_few_shot_sample(processed_roi, self.current_few_shot_label)
                            self.collected_samples += 1
                            print(f"Collected sample {self.collected_samples}/{self.required_samples}")

                            if self.collected_samples >= self.required_samples:
                                print(f"Few-shot learning complete for '{self.current_few_shot_label}'")
                                self.is_collecting = False
                                self.current_few_shot_label = None
                                self.collected_samples = 0

                                if self.few_shot_path:
                                    self.model.save(self.model_path, self.few_shot_path)
                                    print(f"Few-shot data saved to {self.few_shot_path}")
                    else:
                        result = self.model.predict(processed_roi)

                        label = result['label']
                        confidence = result['confidence']
                        source = result['source']

                        if label is None or confidence < 0.3:
                            cv2.putText(frame, "No gesture detected", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        elif source == 'few-shot':
                            cv2.putText(frame, f"Detected: {label} ({confidence:.2f})", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            letter = self.classes.get(label, f"Class {label}")
                            cv2.putText(frame, f"Detected: {letter} ({confidence:.2f})", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                def draw_text_with_background(img, text, position, font, scale, text_color, bg_color, thickness=1, padding=5):
                    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
                    x, y = position
                    cv2.rectangle(img, (x - padding, y - text_size[1] - padding),
                                  (x + text_size[0] + padding, y + padding), bg_color, -1)
                    cv2.putText(img, text, (x, y), font, scale, text_color, thickness)

                # Texts in console
                draw_text_with_background(frame, "Press 'Q' to quit", (10, h - 60),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), (192, 216, 136))

                draw_text_with_background(frame, "Press 'N' to add a new gesture", (10, h - 35),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), (192, 216, 136))

                if self.is_collecting:
                    draw_text_with_background(frame, "Press 'C' to capture sample", (10, h - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), (0, 128, 128))

                cv2.imshow('SLR App', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n') and not self.is_collecting:
                    label = input("Enter a label for the new gesture: ")
                    if label:
                        self.is_collecting = True
                        self.current_few_shot_label = label
                        self.collected_samples = 0

        cap.release()
        cv2.destroyAllWindows()

        if self.few_shot_path:
            self.model.save(self.model_path, self.few_shot_path)
            print(f"Few-shot data saved to {self.few_shot_path}")

def main():
    parser = argparse.ArgumentParser(description='ASL Recognition with Few-Shot Learning')
    parser.add_argument('--model', type=str, default='models/best_model_asl.pth',
                        help='Path to the pre-trained model')
    parser.add_argument('--few_shot', type=str, default='models/few_shot_data.pkl',
                        help='Path to few-shot learning data')

    args = parser.parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found at: {args.model}")

    app = SLRApp(args.model, args.few_shot)
    app.run()


if __name__ == "__main__":
    main()
