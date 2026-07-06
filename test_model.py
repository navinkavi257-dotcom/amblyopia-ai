from tensorflow.keras.models import load_model

print("Loading model...")

model = load_model("amblyopia_model.keras")

print("✅ Model loaded successfully!")

model.summary()