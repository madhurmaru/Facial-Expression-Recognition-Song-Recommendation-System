import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
import pandas as pd
import time
from threading import Thread

# Face detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cv2.ocl.setUseOpenCL(False)

# Load trained model
emotion_model = load_model("model.h5")

# Emotion labels and song mappings
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprised"
}

music_dist = {
    0: "songs/angry.csv", 1: "songs/disgusted.csv", 2: "songs/fearful.csv",
    3: "songs/happy.csv", 4: "songs/neutral.csv", 5: "songs/sad.csv", 6: "songs/surprised.csv"
}

# Globals
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
show_text = [0]


''' Class for using another thread for video streaming to boost performance '''
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


''' Class for reading video stream, generating prediction and recommendations '''
class VideoCamera(object):
    def get_frame(self):
        global cap1, df1
        cap1 = WebcamVideoStream(src=0).start()

        while True:
            image = cap1.read()
            if image is not None:
                break
            print("Waiting for camera...")
            time.sleep(0.08)

        image = cv2.resize(image, (600, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        df1 = pd.read_csv(music_dist[show_text[0]])[['Name', 'Album', 'Artist']].head(15)

        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray, (48, 48)).astype("float") / 255.0
            cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

            prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            show_text[0] = maxindex

            emotion_label = emotion_dict[maxindex]
            cv2.putText(image, emotion_label, (x + 20, y - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            df1 = music_rec()

        global last_frame1
        last_frame1 = image.copy()
        img = Image.fromarray(last_frame1)
        img = np.array(img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes(), df1


def music_rec():
    df = pd.read_csv(music_dist[show_text[0]])
    df = df[['Name', 'Album', 'Artist']]
    df = df.head(15)
    return df
