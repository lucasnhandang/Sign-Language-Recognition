import os
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
from model import SLR

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600

MODEL_PATH = 'models/best_model_asl.pth'
FEW_SHOT_PATH = 'models/few_shot_data.pkl'


class SLRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SLR App")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False)

        # MediaPipe Hand
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )

        # Load model
        try:
            self.model = SLR()
            if os.path.exists(MODEL_PATH):
                print(f"Loading model from {MODEL_PATH}")
                self.model.load(MODEL_PATH, FEW_SHOT_PATH)
            else:
                print(f"Warning: Model not found at {MODEL_PATH}")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        # A-Z + other symbol
        self.classes = {i: chr(65 + i) for i in range(26)}
        self.classes.update({26: 'del', 27: 'nothing', 28: 'space'})

        # Few-shot collection state
        self.is_collecting = False
        self.current_label = None
        self.collected = 0
        self.required = 5
        self.latest_roi = None

        self.setup_camera()

        self.setup_ui()

        self.update_frame()

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.video_frame = tk.Frame(main_frame, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        self.video_frame.pack(pady=10)
        self.video_frame.pack_propagate(False)

        self.canvas = tk.Canvas(self.video_frame, bg="black", width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready")
        status_frame = tk.Frame(main_frame, relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill=tk.X, pady=10)

        status_label = tk.Label(status_frame, textvariable=self.status_var, anchor=tk.W, padx=10, pady=5)
        status_label.pack(fill=tk.X)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.btn_add = tk.Button(button_frame, text="Add New Gesture",
                                 command=self.add_new_gesture, padx=20, pady=10)
        self.btn_add.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_capture = tk.Button(button_frame, text="Capture Sample",
                                     command=self.capture_sample, padx=20, pady=10, state=tk.DISABLED)
        self.btn_capture.pack(side=tk.LEFT)

        self.btn_quit = tk.Button(button_frame, text="Quit",
                                  command=self.quit_app, padx=20, pady=10)
        self.btn_quit.pack(side=tk.RIGHT)

    def add_new_gesture(self):
        label = simpledialog.askstring("New Gesture", "Enter label for new gesture:", parent=self.root)
        if label:
            label = label.strip()
            if label:
                self.is_collecting = True
                self.current_label = label
                self.collected = 0
                self.btn_capture.config(state=tk.NORMAL)
                self.status_var.set(f"Ready to collect samples for '{label}'. Press 'Capture Sample' when ready.")

    def capture_sample(self):
        if not self.is_collecting or self.latest_roi is None:
            return

        try:
            self.model.add_few_shot_sample(self.latest_roi, self.current_label)
            self.collected += 1

            self.status_var.set(f"Collected {self.collected}/{self.required} for '{self.current_label}'")

            if self.collected >= self.required:
                if self.model and (os.path.exists(FEW_SHOT_PATH) or FEW_SHOT_PATH):
                    self.model.save(MODEL_PATH, FEW_SHOT_PATH)
                    print(f"Few-shot data saved to {FEW_SHOT_PATH}")

                label = self.current_label
                self.is_collecting = False
                self.current_label = None
                self.collected = 0
                self.btn_capture.config(state=tk.DISABLED)

                messagebox.showinfo("Success. ", f"Added gesture '{label}' successfully!")
                self.status_var.set("Ready")
        except Exception as e:
            print(f"Error capturing sample: {e}")
            messagebox.showerror("Error", f"Failed to capture sample: {str(e)}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror view
            display_frame = frame.copy()

            self.process_hands(frame, display_frame)

            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_frame)
            img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)

            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root.after(15, self.update_frame)

    def process_hands(self, frame, display_frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            hand_landmarks = results.multi_hand_landmarks[0]

            points_x = [lm.x for lm in hand_landmarks.landmark]
            points_y = [lm.y for lm in hand_landmarks.landmark]
            center_x = int(np.mean(points_x) * w)
            center_y = int(np.mean(points_y) * h)

            roi_size = 300
            x1 = max(0, center_x - roi_size // 2)
            y1 = max(0, center_y - roi_size // 2)
            x2 = min(w, center_x + roi_size // 2)
            y2 = min(h, center_y + roi_size // 2)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                self.latest_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                if self.is_collecting:
                    self.display_collection_status(display_frame)
                elif self.model:
                    self.predict_gesture(display_frame)

    def display_collection_status(self, frame):
        text = f"Collecting '{self.current_label}': {self.collected}/{self.required}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    def predict_gesture(self, frame):
        try:
            prediction = self.model.predict(self.latest_roi)

            if prediction:
                label = prediction.get('label')
                confidence = prediction.get('confidence', 0)
                source = prediction.get('source', '')

                if not label or confidence < 0.3:
                    status = "No gesture detected"
                else:
                    display_label = label if source == 'few-shot' else self.classes.get(label, str(label))
                    status = f"{display_label} ({confidence:.2f})"

                if source == 'few-shot':
                    color = (255, 0, 0)
                elif status == 'No gesture detected':
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.putText(frame, f"Detected: {status}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except Exception as e:
            print(f"Error in prediction: {e}")

    def quit_app(self):
        """Clean up and exit application"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.hands.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SLRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()