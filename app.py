from flask import Flask, render_template, request, jsonify
from camera import process_browser_frame, music_rec

app = Flask(__name__, static_folder="static")

headings = ("Name", "Album", "Artist")


@app.route("/")
def index():
    df = music_rec(4)  # Default: Neutral
    return render_template("index.html", headings=headings, data=df)


@app.route("/health")
def health():
    return jsonify({"status": "running"})


@app.route("/predict_emotion", methods=["POST"])
def predict_emotion():
    try:
        print("Received prediction request", flush=True)

        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "emotion": "No image received",
                "confidence": 0,
                "faces": 0,
                "processed_image": "",
                "songs": []
            }), 400

        result = process_browser_frame(data["image"])
        return jsonify(result)

    except Exception as e:
        print("SERVER ERROR:", str(e), flush=True)

        return jsonify({
            "emotion": "Server error",
            "confidence": 0,
            "faces": 0,
            "processed_image": "",
            "songs": music_rec(4).to_dict(orient="records")
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
