from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import mlflow
import tensorflow as tf

MODEL_ID = "7e8210778802463d8425c601f8af7331"

IMG_SIZE = 28

app = FastAPI()

# Load model from MLflow
model = mlflow.tensorflow.load_model(f"runs:/{MODEL_ID}/model")

@app.get("/")
async def home():
    return {"message": "Hello World!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load image from upload
    image = Image.open(file.file).convert('L').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))

    return {"prediction": predicted_class}
