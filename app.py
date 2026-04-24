from flask import Flask, render_template, request, jsonify
from camera import process_browser_frame, music_rec

app = Flask(__name__, static_folder="static")

headings = ("Name", "Album", "Artist")


@app.route("/")
def index():
    df = music_rec(4)  # default Neutral songs
    return render_template("index.html", headings=headings, data=df)


@app.route("/predict_emotion", methods=["POST"])
def predict_emotion():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({
            "emotion": "No image received",
            "songs": []
        }), 400

    result = process_browser_frame(data["image"])
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
