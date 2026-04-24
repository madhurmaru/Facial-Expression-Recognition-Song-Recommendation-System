import os
import base64
import numpy as np
import cv2
import pandas as pd
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
model_path = os.path.join(BASE_DIR, "model.h5")

face_cascade = cv2.CascadeClassifier(face_cascade_path)
cv2.ocl.setUseOpenCL(False)

emotion_model = load_model(model_path)

emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
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


def music_rec(emotion_index=4):
    csv_path = os.path.join(BASE_DIR, music_dist[emotion_index])
    df = pd.read_csv(csv_path)
    return df[["Name", "Album", "Artist"]].head(15)


def encode_image_to_base64(image):
    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        return ""

    encoded = base64.b64encode(buffer).decode("utf-8")
    return "data:image/jpeg;base64," + encoded


def process_browser_frame(image_data):
    try:
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return {
                "emotion": "Image decode failed",
                "confidence": 0,
                "faces": 0,
                "processed_image": "",
                "songs": []
            }

        image = cv2.resize(image, (600, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_rects = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40)
        )

        emotion_index = 4
        emotion_label = "Neutral"
        confidence = 0

        if len(face_rects) > 0:
            x, y, w, h = face_rects[0]

            cv2.rectangle(
                image,
                (x, y - 50),
                (x + w, y + h + 10),
                (0, 255, 0),
                2
            )

            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = cropped_img.astype("float32") / 255.0
            cropped_img = np.expand_dims(cropped_img, axis=-1)
            cropped_img = np.expand_dims(cropped_img, axis=0)

            prediction = emotion_model.predict(cropped_img, verbose=0)

            emotion_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100
            emotion_label = emotion_dict[emotion_index]

            cv2.putText(
                image,
                emotion_label,
                (x + 20, y - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        processed_image = encode_image_to_base64(image)
        df = music_rec(emotion_index)

        return {
            "emotion": emotion_label if len(face_rects) > 0 else "No face detected",
            "confidence": round(confidence, 2),
            "faces": int(len(face_rects)),
            "processed_image": processed_image,
            "songs": df.to_dict(orient="records")
        }

    except Exception as e:
        print("Error in process_browser_frame:", str(e))

        return {
            "emotion": "Prediction error",
            "confidence": 0,
            "faces": 0,
            "processed_image": "",
            "songs": music_rec(4).to_dict(orient="records")
        }
