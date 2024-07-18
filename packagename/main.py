from packagename.registry import load_my_model, load_my_yolo_model
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
#from tkinter import *
from PIL import Image
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import cv2

def preprocess(image):
    x=img_to_array(image)
    x_scaled=x/255
    x_scaled.resize((120, 120,3))
    reshaped_image = np.expand_dims(x_scaled, axis=0)
    return reshaped_image

def my_predict(image):
    model = load_my_model()

    prediction = model.predict(image)

    print(f'Probabilities of new cracks {prediction[0][0]}')

    return prediction[0][0]

def my_yolo_predict(image):
    model = load_my_yolo_model()

    results=model(image)

    # Initialize a list to store mask coordinates
    xy_array_list = []

    # Check if masks were detected
    if results:
        for r in results:
            if r.masks is not None:
                xy_array_list.append(r.masks.xy)

    if len(xy_array_list)!=0:
        return xy_array_list
    else :
        return 0

def my_yolo_mask(xy_array_list,image):
    # Read the original image
    image = cv2.imread(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Create a figure with two subplots (side by side)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image on the first subplot
    axs[0].imshow(image_rgb)
    axs[0].set_title(f"Original Image")
    axs[0].axis('off')

    # Display the image with masks on the second subplot
    axs[1].imshow(image_rgb)

    if xy_array_list:
        # Plot the masks
        for xy_array in xy_array_list:
            for xy in xy_array:
                polygon = plt.Polygon(xy, closed=True, fill=None, edgecolor='r')
                axs[1].add_patch(polygon)

                # Calculate and display mask metrics
                area, length, average_width = calculate_mask_metrics(xy)
                axs[1].text(10, 10, f"Area: {area:.2f}\nLength: {length:.2f}\nAvg Width: {average_width:.2f}",
                            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        axs[1].set_title(f"Image with Mask")
    else:
        axs[1].set_title(f"No crack detected")

    axs[1].axis('off')

    # Save the plot
    plt.tight_layout()

    plt.savefig('my_mask.jpeg',format='jpeg')

    my_mask=cv2.imread('my_mask.jpeg')

    return my_mask



def calculate_mask_metrics(xy_array):
    # Convert xy array to a polygon
    polygon = Polygon(xy_array)

    # Calculate surface area
    area = polygon.area

    # Calculate length (perimeter of the polygon)
    length = polygon.length

    # Calculate the average width (area divided by length)
    average_width = area / length if length != 0 else 0

    return area, length, average_width







if __name__ == "__main__":
    my_path = Path(__file__).parent.parent/"models/00001.jpg"
    image = load_img(my_path)
    reshaped_image = preprocess(image)

    my_predict(reshaped_image)
