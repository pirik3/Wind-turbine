from roboflow import Roboflow
from PIL import Image
import os
from os import listdir
import time

rf = Roboflow(api_key="HTUVYlfkYY3uioUyEkhO")
project = rf.workspace().project("surface-damage-wind-turbine-40qrp")
model = project.version(1).model

folder_dir = "C:/Users/pirik3/Pictures/damaged blade/"
for images in os.listdir(folder_dir):
     
    # check if the image ends with png
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
        img_path = folder_dir + images
        model.predict(img_path, confidence=1, overlap=1).save('result_{}.png'.format(int(time.time())))
        im = Image.open(r'result_{}.png'.format(int(time.time())))
        im.show()

# infer on a local imageclear



# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
