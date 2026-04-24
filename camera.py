import numpy as np
import cv2
from keras.models import load_model
import pandas as pd
import base64

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cv2.ocl.setUseOpenCL(False)

emotion_model = load_model("model.h5")

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprised"
}

music_dist = {
    0: "songs/angry.csv",
    1: "songs/disgusted.csv",
    2: "songs/fearful.csv",
    3: "songs/happy.csv",
    4: "songs/neutral.csv",
    5: "songs/sad.csv",
    6: "songs/surprised.csv"
}


def music_rec(emotion_index):
    df = pd.read_csv(music_dist[emotion_index])
    return df[["Name", "Album", "Artist"]].head(15)


def process_browser_frame(image_data):
    """
    image_data comes from browser as base64 image.
    """

    # Remove base64 header
    if "," in image_data:
        image_data = image_data.split(",")[1]

    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)

    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "emotion": "No image detected",
            "songs": []
        }

    image = cv2.resize(image, (600, 500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion_index = 4
    emotion_label = "Neutral"

    for (x, y, w, h) in face_rects:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = cropped_img.astype("float") / 255.0
        cropped_img = np.expand_dims(cropped_img, axis=-1)
        cropped_img = np.expand_dims(cropped_img, axis=0)

        prediction = emotion_model.predict(cropped_img)
        emotion_index = int(np.argmax(prediction))
        emotion_label = emotion_dict[emotion_index]
        break

    df = music_rec(emotion_index)

    return {
        "emotion": emotion_label,
        "songs": df.to_dict(orient="records")
    }
