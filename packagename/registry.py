import glob
import os
from tensorflow import keras

def load_model() -> keras.Model:

    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    model = keras.models.load_model(local_model_paths)

    return model
