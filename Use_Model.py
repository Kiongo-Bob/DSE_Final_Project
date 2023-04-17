import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Define the path to the test data directory
test_dir = './Test'

# Define the target image size
target_size = (224, 224)

# Define the class labels
class_labels = ['Cattle', 'Chicken', 'Goat', 'Sheep']

# Load the trained model
model_path = 'DSE_Final_Project/Domestic_Farmer.h5'
model = load_model(model_path)

# Loop over each class directory
for class_name in class_labels:
    class_dir = os.path.join(test_dir, class_name)
    
    # Loop over each image file in the class directory
    for filename in os.listdir(class_dir):
        if filename.endswith('.jpg'):
            # Load the image and preprocess it
            image_path = os.path.join(class_dir, filename)
            image = load_img(image_path, target_size=target_size)
            image_array = img_to_array(image)
            image_array = image_array / 255.0
            image_batch = np.expand_dims(image_array, axis=0)
            
            # Make a prediction for the image
            prediction = model.predict(image_batch)
            predicted_label = class_labels[np.argmax(prediction)]
            
            # Print the predicted label and file name
            print(f'{filename} - predicted: {predicted_label}')

