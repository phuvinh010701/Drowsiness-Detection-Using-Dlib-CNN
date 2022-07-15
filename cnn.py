
from keras.models import load_model
import numpy as np
from imutils import face_utils
import dlib
import cv2

left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def predict_eye_state(model, ima):
    ima = cv2.resize(ima, (64, 64))
    ima = np.expand_dims(ima, axis=0)
    ima = np.expand_dims(ima, axis=3)
    return model(ima)

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
model = load_model('eye_detect.h5')
cap = cv2.VideoCapture(0)

scale = 0.5
countClose = 0
currState = 0
alarmThreshold = 20

while (True):
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_locations = detector(gray, 0)

    if len(face_locations) >= 1:

        rec = face_locations[0]

        shape = predictor(gray, rec)
        shape = face_utils.shape_to_np(shape)

        face = cv2.convexHull(shape)


        left_eye_indices = shape[left_eye_start_index:left_eye_end_index]

        (x, y, w, h) = cv2.boundingRect(np.array([left_eye_indices]))
        left_eye = gray[y - int(h):y + h + int(h), x - int(0.25 * w):x + w + int(0.25 * w)]
        
        cv2.imshow('left eye', left_eye)

        right_eye_indices = shape[right_eye_start_index:right_eye_end_index]

        (x, y, w, h) = cv2.boundingRect(np.array([right_eye_indices]))
        right_eye = gray[y - int(h):y + h + int(h), x - int(0.25 * w):x + w + int(0.25 * w)]
        
        cv2.imshow('right eye', right_eye)

        if predict_eye_state(model, left_eye) and predict_eye_state(model, right_eye):
            cv2.drawContours(frame, [face], -1, (0, 255, 0), 1)
            currState = 0
            countClose = 0

        else:
            cv2.drawContours(frame, [face], -1, (0, 0, 255), 1)
            currState = 1
            countClose +=1

    if countClose > alarmThreshold:
        cv2.putText(frame, "Sleep detected! Alarm!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)

    cv2.imshow('Sleep Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()
