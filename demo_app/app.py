from flask import Flask, Response, jsonify, render_template
from detection import DetectionState, Detector
import time

app = Flask(__name__)

state = DetectionState()
detector = Detector(state, camera_index=0)


def generate_frames():
    while True:
        frame = detector.get_frame()
        if frame is None:
            time.sleep(0.03)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    snap = state.snapshot()
    last_seen_ago = None
    if snap["last_seen"] is not None:
        last_seen_ago = round(time.time() - snap["last_seen"], 1)
    return jsonify(
        person_in_sight=snap["person_in_sight"],
        last_seen_ago=last_seen_ago,
        frame_count=snap["frame_count"],
    )

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    detector.start()
    try:
        app.run(host="0.0.0.0", port=4000, debug=True, use_reloader=False)
    finally:
        detector.stop()