import os
import base64
import numpy as np
import cv2
import pandas as pd
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

if face_cascade.empty():
    print("ERROR: Haar cascade file not loaded.", flush=True)
    print("Expected path:", FACE_CASCADE_PATH, flush=True)

cv2.ocl.setUseOpenCL(False)

print("Loading emotion model...", flush=True)
emotion_model = load_model(MODEL_PATH)
print("Emotion model loaded.", flush=True)

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
    try:
        csv_path = os.path.join(BASE_DIR, music_dist[emotion_index])
        df = pd.read_csv(csv_path)
        return df[["Name", "Album", "Artist"]].head(15)

    except Exception as e:
        print("Music CSV error:", str(e), flush=True)
        return pd.DataFrame(columns=["Name", "Album", "Artist"])


def encode_image_to_base64(image):
    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        return ""

    encoded_image = base64.b64encode(buffer).decode("utf-8")
    return "data:image/jpeg;base64," + encoded_image


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
                "songs": music_rec(4).to_dict(orient="records")
            }

        image = cv2.resize(image, (320, 240))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_rects = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40)
        )

        print("Faces detected:", len(face_rects), flush=True)

        emotion_index = 4
        emotion_label = "No face detected"
        confidence = 0.0

        if len(face_rects) > 0:
            x, y, w, h = face_rects[0]

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            roi_gray = gray[y:y + h, x:x + w]

            if roi_gray.size > 0:
                cropped_img = cv2.resize(roi_gray, (48, 48))
                cropped_img = cropped_img.astype("float32") / 255.0
                cropped_img = np.expand_dims(cropped_img, axis=-1)
                cropped_img = np.expand_dims(cropped_img, axis=0)

                prediction = emotion_model.predict(cropped_img, verbose=0)

                emotion_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction)) * 100
                emotion_label = emotion_dict.get(emotion_index, "Unknown")

                cv2.putText(
                    image,
                    emotion_label,
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                print(
                    "Prediction:",
                    emotion_label,
                    "Confidence:",
                    round(confidence, 2),
                    flush=True
                )

        processed_image = encode_image_to_base64(image)
        df = music_rec(emotion_index)

        return {
            "emotion": emotion_label,
            "confidence": round(confidence, 2),
            "faces": int(len(face_rects)),
            "processed_image": processed_image,
            "songs": df.to_dict(orient="records")
        }

    except Exception as e:
        print("Error in process_browser_frame:", str(e), flush=True)

        return {
            "emotion": "Prediction error",
            "confidence": 0,
            "faces": 0,
            "processed_image": "",
            "songs": music_rec(4).to_dict(orient="records")
        }
