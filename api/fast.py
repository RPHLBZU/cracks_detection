# TODO: Import your package, replace this by explicit imports of what you need

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from packagename.registry import load_model
# from packagename.main import predict


app = FastAPI()
app.state.model = load_model()

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
@app.get("/predict")
def get_predict(image):
    # TODO: Do something with your input
    # i.e. feed it to your model.predict, and return the output
    # For a dummy version, just return the sum of the two inputs and the original inputs
    # prediction = float(input_one) + float(input_two)

    image_processed=preprocess_image(image)

    prediction = app.state.model.predict(image_processed) #(index??)
    if prediction == 0 :
        prediction_sentence = "Photograp does not contain cracks"
    elif prediction == 1 :
        prediction_sentence = "Photograp contains cracks"


    return {
        'prediction': prediction_sentence,
            }
