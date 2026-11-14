from fastapi import FastAPI, File, UploadFile
from enum import Enum
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potato_model:predict"

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
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    json_data={
        "instances": img_batch.tolist()
    }
    respone = requests.post(endpoint , json = json_data)
    prediction = np.array(respone.json()["predictions"][0])
    predicted_class = Class_Names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        'class': predicted_class,
        'confidence': confidence
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
