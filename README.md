**Cracks Detector**


Cracks Detector is a Python package designed to predict cracks on infrastructures (roads, bridges, buildings) using deep learning models on photographs.
This tool helps in identifying and assessing the severity of cracks, making it a valuable resource for maintenance and safety inspections.

**Features**

Predicts the presence of cracks in images with 2 different models
Assesses the severity and dimensions of detected cracks.
       

**Models**

Model2.keras is a custom made Convolutional Neural Network, built, trained and evaluated by the team who created this package
Yolov8m-seg was used as pre-trained model. It was then trained on 30 epochs with roboflow crack-detection dataset.
    
![image](https://github.com/user-attachments/assets/64fb217a-3cf4-4a3e-b58f-4a5d34d11065)

**Datasets**

*Roboflow dataset*
https://universe.roboflow.com/university-bswxt/crack-bphdr/dataset/2

*Kaggle Surcace Crack Detection*
https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

*SDNet 2018*
https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images


**Usage**

This package is the back-end part of a web application.
https://cracksdetectionui-a8f28peutiafdfxnrmdqzw.streamlit.app/
    


**Examples**

Here are some example images and their predicted results:
![image](https://github.com/user-attachments/assets/9c411aeb-6c23-484f-8361-e519b52dd75d)


**Acknowledgements**

*Yolo model*
  
We acknowledge the use of the Ultralytics YOLOv8 model in this application. Proper credit is given to the authors of this model, as detailed below:
@software{yolov8_ultralytics,
author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
title = {Ultralytics YOLOv8},
version = {8.0.0},
year = {2023},
url = {https://github.com/ultralytics/ultralytics},
orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
license = {AGPL-3.0}
}
The use of this model is in accordance with the AGPL-3.0 license, and we ensure that our application complies with the terms of this license.



