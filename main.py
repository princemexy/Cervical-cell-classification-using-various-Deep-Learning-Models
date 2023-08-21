from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

def classify_image(image_path):
    # Loading the model
    model = load_model("models/ensemble_model_v2.h5")
    
    # Preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply the same preprocessing used during training
    img_array = preprocess_input(img_array)
    
    # Make prediction
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])  # Get the index of the highest predicted probability
    
    class_labels = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']

    predicted_class = class_labels[class_index]
    return predicted_class

#result = classify_image(r"C:\Users\Mexy\Documents\Hull Workshop\Research\data for research\im_Koilocytotic\CROPPED\002_01.bmp")
#print(result)