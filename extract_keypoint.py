import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_keypoints(results):
    # If left hand is recognized
    if results.left_hand_landmarks:
        left_hand = []
        for landmark in results.left_hand_landmarks:
            left_hand.extend([landmark.x, landmark.y, landmark.z])
        left_hand = np.array(left_hand)
    else:
        left_hand = np.zeros(21 * 3)

    # If right hand is recognized
    if results.right_hand_landmarks:
        right_hand = []
        for landmark in results.right_hand_landmarks:
            right_hand.extend([landmark.x, landmark.y, landmark.z])
        right_hand = np.array(right_hand)
    else:
        right_hand = np.zeros(21 * 3)

    # Concatenate to a vector
    keypoints = np.concatenate([left_hand, right_hand])

    return keypoints

def process():
    # Open webcam
    cam = cv2.VideoCapture(0)

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cam.isOpened():
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break

            # Convert to RGB and run MediaPipe
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = hands.process(img)

            # Convert back to BGR
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Draw landmarks if any
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):

                    # Left or right hand?
                    hand_label = hand_info.classification[0].label

                    # Assign colors by left/right hand
                    if hand_label == 'Left':
                        color = (255, 0, 0)  # Blue
                    else:
                        color = (0, 0, 255)  # Red

                    # Draw connections
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Draw each keypoints
                    h, w, _ = img.shape
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 5, color, -1)

            cv2.imshow("Hand Keypoint", img)

            # Press Q to break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Free resources
    cam.release()
    cv2.destroyAllWindows()

def main():
    process()

if __name__ == '__main__':
    main()


