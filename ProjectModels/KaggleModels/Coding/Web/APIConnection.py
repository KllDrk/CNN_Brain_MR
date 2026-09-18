import uvicorn as uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DicProjectUrl = "C:/Users/ahmet/Documents/Desktop/FinalProjesi"
modelsUrl = f"{DicProjectUrl}/ProjectModels/KaggleModels/Models/"
webUrl = f"{DicProjectUrl}/ProjectModels/KaggleModels/Coding/Web/"
app = FastAPI()

ModelTumor = tf.keras.models.load_model(f"{modelsUrl}ModelDL/TumorModel/TumorModel1")
Class_Names_Tumor = ["MormalImg", "TumorImg"]

ModelAlzheimer = tf.keras.models.load_model(f"{modelsUrl}ModelDL/AlzheimerModel/AlzheimerModel1")
Class_Names_Alzheimer = ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]

templates = Jinja2Templates(directory="templates")


@app.get("/Deneme")
async def ping():
    return "Hello World"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/predictTumor", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("predictTumor.html", {'request': request})


@app.post("/predictTumor")
async def predictTumor(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    print(image.shape)
    img_batch = np.expand_dims(image, 0)
    print(img_batch)
    predictions = ModelTumor.predict(img_batch)

    print(predictions)
    index = np.max(predictions[0])
    predicted_class = Class_Names_Tumor[np.argmax(predictions[0])]

    return {'class': predicted_class, 'index': str(index)}


@app.get("/predictAlzheimer", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("predictAlzheimer.html", {'request': request})


@app.post("/predictAlzheimer")
async def predictAlzheimer(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    print(image.shape)
    image = cv2.merge((image, image, image))
    print(image.shape)
    img_batch = np.expand_dims(image, 0)
    print(img_batch)
    predictions = ModelAlzheimer.predict(img_batch)

    print(predictions)
    index = np.max(predictions[0])
    predicted_class = Class_Names_Alzheimer[np.argmax(predictions[0])]

    return {'class': predicted_class, 'index': str(index)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
