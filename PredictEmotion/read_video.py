import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import pandas as pd

# Face detection XML load and trained model loading 
face_detection = cv2.CascadeClassifier('haarcascade_frontalface.xml')
emotion_classifier = load_model('emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]
Col = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral","X_pos","Y_pos","scale"]

#비디오경로
video = cv2.VideoCapture('/home/educon/data2/video/kpg/data/201012/201012_ra5.mp4')
preds_record = pd.DataFrame(columns=Col)

while (video.isOpened()):
    ret, frame = video.read()
   
    if ret:
        # convert color to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Face detection in frame
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        # Perform emotion recognition only when face is detected
        if len(faces) > 0:
            # For the largest image
            face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = face
            print(fX,fY,fW,fH)
            # Resize the image to 48x48 for neural network
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # Emotion predict
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            
            cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            
            # record predction value temporary
            print(preds)
            preds_append = {}
            for i in range(len(EMOTIONS)):
                preds_append[preds_record.columns[i]] = preds[i]
            preds_append["X_pos"] = fX
            preds_append["Y_pos"] = fY
            preds_append["scale"] = fW
            preds_record = preds_record.append(preds_append, ignore_index=True)
        
        #fail to detect face
        else:
            preds_append = {}
            for i in range(len(Col)):
                preds_append[preds_record.columns[i]] = 0
            preds_record = preds_record.append(preds_append, ignore_index=True)
        cv2.imshow("Probabilities", canvas)
        cv2.imshow('Emotion Recognition', frame)
        # esc to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

# save data to pickle
print(preds_record)

## 데이터저장되는위치
preds_record.to_pickle("./data/pickle/201012_ra5_0.pkl")
video.release()
cv2.destroyAllWindows()
