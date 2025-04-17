# SCT_ML_4

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf # Or PyTorch, or Scikit-learn load
# import pyautogui # For control (optional)

# --- Configuration ---
MODEL_PATH = 'path/to/your/trained_gesture_model.h5' # Or .pkl for sklearn
NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
GESTURE_LABELS = ['Fist', 'Open Palm', 'Thumbs Up', 'Pointing', 'Victory'] # Example

# --- Load Model ---
# Example for Keras/TF model
model = tf.keras.models.load_model(MODEL_PATH)
# Example for Scikit-learn model (e.g., SVM, RF)
# import joblib
# model = joblib.load(MODEL_PATH)

# --- Initialize MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)

# --- Real-time Loop ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip image horizontally for a later selfie-view display
    # Convert the BGR image to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False # Performance improvement

    # Process the image and find hands
    results = hands.process(image)

    # Convert back to BGR for OpenCV rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    predicted_gesture = "No Hand Detected"

    # If hand(s) detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # --- Feature Extraction (Example: Normalized Landmark Coordinates) ---
            landmarks_list = []
            # Use wrist (landmark 0) as origin for normalization
            origin = hand_landmarks.landmark[0]
            origin_x, origin_y = origin.x, origin.y # Ignore z for 2D focus if desired

            for landmark in hand_landmarks.landmark:
                 # Example: Normalize relative to wrist, scale appropriately
                 # Note: More robust normalization might be needed
                 norm_x = (landmark.x - origin_x) * image.shape[1] # Scale by image width
                 norm_y = (landmark.y - origin_y) * image.shape[0] # Scale by image height
                 # Add z if using 3D, potentially other features like distances/angles
                 landmarks_list.extend([norm_x, norm_y])

            # Ensure the feature vector has the correct shape expected by the model
            feature_vector = np.array(landmarks_list, dtype=np.float32)
            # Reshape if necessary (e.g., for Keras MLP: (1, num_features))
            feature_vector = np.expand_dims(feature_vector, axis=0)

            # --- Prediction ---
            # For Keras/TF
            prediction = model.predict(feature_vector)
            predicted_class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            # For Scikit-learn
            # prediction = model.predict(feature_vector)
            # predicted_class_index = prediction[0] # predict returns the class index/label directly
            # confidence = model.predict_proba(feature_vector).max() # If proba available

            # --- Map prediction to label (Apply confidence threshold) ---
            CONF_THRESHOLD = 0.8 # Example threshold
            if confidence > CONF_THRESHOLD:
                predicted_gesture = GESTURE_LABELS[predicted_class_index]
            else:
                 predicted_gesture = "Uncertain"

            # --- Display Prediction ---
            cv2.putText(image, f'Gesture: {predicted_gesture} ({confidence:.2f})',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # --- HCI Control (Example) ---
            # if predicted_gesture == 'Fist':
            #     pyautogui.click()
            # elif predicted_gesture == 'Pointing':
            #     # Map pointing finger position to cursor (requires more processing)
            #     pass

            # Only process the first detected hand if NUM_HANDS=1
            break

    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition', image)

    if cv2.waitKey(5) & 0xFF == 27: # Press Esc to exit
        break

# --- Cleanup ---
hands.close()
cap.release()
cv2.destroyAllWindows()
