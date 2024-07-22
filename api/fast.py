# TODO: Import your package, replace this by explicit imports of what you need

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.responses import Response

from packagename.registry import load_my_model, load_my_yolo_model
from packagename.main import preprocess, my_predict, my_yolo_mask, calculate_severity

from PIL import Image
from pydantic import BaseModel
from typing import List
from io import BytesIO
import cv2
# from packagename.registry import load_model
# from packagename.main import predict


app = FastAPI()
app.state.model = load_my_model()
app.state.yolo_model = load_my_yolo_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.post("/predict")
def post_predict(file: UploadFile = File(...)):
    # TODO: Do something with your input
    # i.e. feed it to your model.predict, and return the output
    # For a dummy version, just return the sum of the two inputs and the original inputs
    # prediction = float(input_one) + float(input_two)

    # Read the uploaded image
    contents = file.file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    image_processed=preprocess(image)

    prediction = app.state.model.predict(image_processed)

    return {'prediction' : float(prediction[0][0])}

@app.post("/yolo_predict", responses={
        204: {"description": "No cracks were detected"},
        })
def yolo_predict(file: UploadFile = File(...)):
    contents = file.file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    results = app.state.yolo_model(image)

    # Initialize a list to store mask coordinates
    xy_array_list = []

    # Check if masks were detected
    if results:
        for r in results:
            if r.masks is not None:
                xy_array_list.append(r.masks.xy)

    if len(xy_array_list)!=0:
        cracks=1
        my_mask=my_yolo_mask(xy_array_list,image)

    else :
        cracks=0
        return JSONResponse(status_code=204,\
                        content={"message": "No crack detected"})

    im = cv2.imencode('.png', my_mask)[1] # extension depends on which format is sent from Streamlit
    return Response(content=im.tobytes(), media_type="image/jpg")

@app.post('/predict_severity')
def yolo_severity(file: UploadFile = File(...)):
    contents = file.file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    results = app.state.yolo_model(image)

    my_severity = 0

    # Check if masks were detected
    if results:
        for r in results:
            if r.masks is not None:
                # print(r.masks.xy[0])
                my_severity += round(calculate_severity(r.masks.xy[0],image),4)


    else :
        cracks=0
        return JSONResponse(status_code=204,\
                        content={"message": "No crack detected"})

    return {'severity' : float(my_severity)}
