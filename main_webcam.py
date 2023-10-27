# Note: This file is compatible with webcam and works well with webcam of computer. 
import face_recognition
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the emotion classifier model
classifier = load_model('model.h5')

cap = cv2.VideoCapture(0)

cv2.namedWindow('Emotion Detector', cv2.WINDOW_NORMAL)  # Create the window with resizable flag

while True:
    _, frame = cap.read()

    # Find face locations using face_recognition library
    face_locations = face_recognition.face_locations(frame)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Extract the face region from the frame
        roi_frame = frame[top:bottom, left:right]
        roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict emotion using the loaded model
        prediction = classifier.predict(roi_gray)[0]
        label = emotion_labels[np.argmax(prediction)]
        label_position = (left, top)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
