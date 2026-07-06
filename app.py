from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

print("Loading model...")
model = load_model("amblyopia_model_new.keras")
print("✅ Model loaded successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():

    if request.method == "POST":

        file = request.files["image"]

        # Save uploaded image
        UPLOAD_FOLDER = os.path.join("static", "uploads")
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Load image
        img = image.load_img(filepath, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)

        classes = ["Amblyopia", "Normal"]

        predicted_index = np.argmax(prediction)
        result = classes[predicted_index]
        confidence = float(np.max(prediction) * 100)

        return render_template(
            "detect.html",
            result=result,
            confidence=round(confidence, 2),
            filename=file.filename
        )

    return render_template("detect.html")


if __name__ == "__main__":
    app.run(debug=True)