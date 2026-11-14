from fastapi import FastAPI, File, UploadFile
from enum import Enum
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/1")

Class_Names = ['Early_blight', 'Late_blight', 'Healthy']


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image





@app.get("/ping")
async def ping():
    return "Hello I am alive"


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    tf.keras.backend.clear_session()
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch,verbose=0)
    index = np.argmax(predictions[0])
    prediction_class = Class_Names[index]
    confidence = np.max(predictions[0])
    return {
        'class' : prediction_class,
        'confidence' : float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
