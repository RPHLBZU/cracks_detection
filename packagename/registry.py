import glob
import os
from tensorflow import keras
from tensorflow.keras.models import load_model
from pathlib import Path

# Load the model
def load_my_model() -> keras.Model:

    #local_model_directory = os.path.join("/models/my_model.keras")

    base_path = Path(__file__).parent.parent   #/models
    local_model_directory = base_path / "models/my_model.keras"

    print(local_model_directory)

    model = load_model(local_model_directory)

    return model
