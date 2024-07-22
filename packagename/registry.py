import glob
import os
from tensorflow import keras
from tensorflow.keras.models import load_model
from pathlib import Path
from ultralytics import YOLO

# Load the model
def load_my_model() -> keras.Model:

    #local_model_directory = os.path.join("/models/my_model.keras")

    base_path = Path(__file__).parent.parent   #/models
    local_model_directory = base_path / "models/model_2.keras"

    model = load_model(local_model_directory)

    return model

def load_my_yolo_model():
    base_path = Path(__file__).parent.parent   #/models
    local_model_directory = base_path / "models/yolo_crack_segm_weights_30train.pt"

    model = YOLO(local_model_directory)

    return model
