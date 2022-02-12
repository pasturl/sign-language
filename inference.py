import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import time


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
model = tf.keras.models.load_model('trained_models/signs_model_efficientnetv2-s')
width = model.input_shape[1]
height = model.input_shape[2]
dim = (width, height)
class_names =["Opaque", "Red", "Green"]
frames_to_rolling_mean = 10

# For webcam input:
cap = cv2.VideoCapture(0)
predictions = []
with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        #start_time = time.time()
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = np.zeros(image.shape)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # resize image
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        prediction_scores = model.predict(np.expand_dims(resized, axis=0))
        # predicted_index = np.argmax(prediction_scores)
        # class_predicted = class_names[predicted_index]
        predictions.append(prediction_scores)
        rolling_mean_prediction = np.mean(predictions[-frames_to_rolling_mean:], axis=0)
        predicted_index = np.argmax(rolling_mean_prediction)
        class_predicted = class_names[predicted_index]

        # Get time to print
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"{current_time} Predicted label: {class_predicted} Probabilities: {rolling_mean_prediction}")
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        # To predict each n seconds
        #time.sleep(2.0 - time.time() + start_time)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
