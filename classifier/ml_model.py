import tensorflow as tf
import os
from django.conf import settings
import numpy as np



'''MODEL = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'ml', 'animalClassier.keras'))
CLASS_NAMES = ['Cat', 'Dog', 'Snake']

def predict(img_file):
    img = tf.keras.preprocessing.image.load_img(img_file, target_size=(224, 224))
    img = tf.keras.preprocessing.image.imge_to_array(img)
    img = np.expand_dims(img, axis=0)

    pred = MODEL.predict(img)[0]
    class_index = np.argmax(pred)
    pred_class = CLASS_NAMES[class_index]
    confidence = float(class_index) * 100

    return pred_class, confidence'''

