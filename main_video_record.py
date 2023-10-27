# This file is there to be implemented on a video. It creates an ID of the person along with bounding box.
# The output is stored in a new file
import face_recognition
import cv2
import numpy as np
from keras.models import load_model

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the emotion classifier model
classifier = load_model('model.h5')

input_video_path = 'videoplayback.mp4'
output_video_path = 'detected.avi'

input_video = cv2.VideoCapture(input_video_path)

frame_width = int(input_video.get(3))
frame_height = int(input_video.get(4))
fps = int(input_video.get(5))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

face_id_dict = {}  # Dictionary to store unique IDs for detected faces

face_id_counter = 1  # Counter for assigning unique IDs to detected faces

total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame = input_video.read()
    if not ret:
        break

    # Find face locations using face_recognition library
    face_locations = face_recognition.face_locations(frame)

    for face_location in face_locations:
        top, right, bottom, left = face_location

        # Extract the face region from the frame
        face_roi = frame[top:bottom, left:right]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (48, 48))
        face_gray = face_gray.astype('float') / 255.0
        face_gray = np.expand_dims(face_gray, axis=0)
        face_gray = np.expand_dims(face_gray, axis=-1)

        # Predict emotion using the loaded model
        prediction = classifier.predict(face_gray)[0]
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        # Assign a unique ID to the detected face if it's a new face
        face_key = tuple(face_location)
        if face_key not in face_id_dict:
            face_id_dict[face_key] = face_id_counter
            face_id_counter += 1

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

        # Display unique ID and emotion above the box
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"ID: {face_id_dict[face_key]}:\t {predicted_emotion}"
        cv2.putText(frame, text, (left + 6, top - 6), font, 0.8, (255, 255, 255), 1)

    # Write frame to the output video
    output_video.write(frame)

    # Calculate and display the progress percentage
    current_frame = int(input_video.get(cv2.CAP_PROP_POS_FRAMES))
    progress_percentage = (current_frame / total_frames) * 100
    print(f"Progress: {progress_percentage:.2f}%")

# Release video objects
input_video.release()
output_video.release()
cv2.destroyAllWindows()
