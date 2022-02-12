import cv2
import streamlit as st
import tensorflow as tf
import time
import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model_path = "./trained_models/signs_model_efficientnetv2-s_v0"
class_names =["Sweet milk", "Argentina", "Barbecue", "Thanks"]
frames_to_rolling_mean = 10
predictions = []
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 1
fontColor = (255,255,255)
thickness = 1
lineType = 2


st.title("Webcam Live Feed")
run = st.checkbox('Run')
prediction = ""
st.title(prediction)
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


model = load_model(model_path)
width = model.input_shape[1]
height = model.input_shape[2]
dim = (width, height)

while run:
    with mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        _, frame = camera.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame_predicted, class_predicted, rolling_mean_prediction = inference_video.predict_frame(model, frame, dim, hands)
        results = hands.process(frame)
        frame_prediction = np.zeros(frame.shape)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                mp_drawing.draw_landmarks(
                    frame_prediction,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # resize image
        resized = cv2.resize(frame_prediction, dim, interpolation=cv2.INTER_AREA)
        prediction_scores = model.predict(np.expand_dims(resized, axis=0))
        predicted_index = np.argmax(prediction_scores)
        class_predicted = class_names[predicted_index]
        predictions.append(prediction_scores)
        rolling_mean_prediction = np.mean(predictions[-frames_to_rolling_mean:], axis=0)
        predicted_index = np.argmax(rolling_mean_prediction)
        class_predicted = class_names[predicted_index]
        max_pred = rolling_mean_prediction[0][predicted_index]
        if max_pred>0.1:
            cv2.putText(frame, class_predicted, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(frame)
        # max_pred = rolling_mean_prediction[predicted_index]
        # print(max_pred)
        # if max_pred>0.1:
        # Get time to print
        # now = datetime.now()
        # current_time = now.strftime("%H:%M:%S")
        # text_results = f"{current_time} Predicted label: {class_predicted} Probabilities: {rolling_mean_prediction}"
        # st.write(text_results)
        #time.sleep(0.2)
else:
    st.write('Stopped')