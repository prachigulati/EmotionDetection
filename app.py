from flask import Flask, render_template, request, jsonify
import base64, re
import numpy as np
import cv2
from tensorflow.keras.models import load_model  # type: ignore

# Initialize Flask app
app = Flask(__name__)

# Load trained emotion detection model
model = load_model("emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def home():
    return render_template("index.html")   # Your HTML frontend

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image data received"}), 400

        image_data = data['image']

        # Decode base64 image
        img_match = re.search(r'base64,(.*)', image_data)
        if not img_match:
            return jsonify({"success": False, "error": "Invalid image format"}), 400

        img_str = img_match.group(1)
        img_bytes = base64.b64decode(img_str)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return jsonify({"success": False, "error": "No face detected"}), 200

        # Take the first detected face
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]

        # Preprocess
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # add channel
        roi_gray = np.expand_dims(roi_gray, axis=0)   # add batch

        # Predict emotion
        preds = model.predict(roi_gray)
        preds = preds[0]  # get first sample

        emotions = [
            {"emotion": emotion_labels[i], "confidence": float(preds[i])}
            for i in range(len(preds))
        ]

        # Get top prediction
        top_idx = int(np.argmax(preds))
        result = {
            "success": True,
            "predicted_emotion": emotion_labels[top_idx],
            "emotions": emotions
        }

        return jsonify(result)

    except Exception as e:
        print("Error in /analyze:", str(e))  # Will show in Render logs
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    # On Render, PORT is provided by environment variable
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
