# TODO: Import your package, replace this by explicit imports of what you need

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from packagename.registry import load_my_model
from packagename.main import preprocess, my_predict

from PIL import Image
from pydantic import BaseModel
from typing import List
from io import BytesIO
# from packagename.registry import load_model
# from packagename.main import predict


app = FastAPI()
app.state.model = load_my_model()

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
