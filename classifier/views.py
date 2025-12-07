
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .forms import UploadImageForm
# from .ml_model import predict

import tensorflow as tf
import numpy as np
import os


# load model onces
model = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'animalClassier.keras'))
class_names = ['Cat', 'Dog', 'Snake']


def classify_image(request):
    result = None
    uploaded_image = None
    confidence = None

    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            img = request.FILES['image']
            uploaded_image = img.name
            img_path = os.path.join(settings.MEDIA_ROOT, uploaded_image)

            '''with open(img_path, 'wb+') as f:
                for chunk in img.chunks():
                    f.write(chunk)'''

            # make url for templtae
            # image_path = settings.MEDIA_URL + img.name

            # saving the image
            fs = FileSystemStorage()
            filename = fs.save(img.name, img)
            image_url = fs.url(filename)
            file_path = fs.path(filename)

            # load and preprocess the image
            img_obj = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_obj)
            img_array= np.expand_dims(img_array, axis=0) / 255.0
            
            # inference
            pred = model.predict(img_array)[0]
            index = np.argmax(pred)
            result = class_names[index]
            confidence = float(pred[index]) * 100
    else:
        form = UploadImageForm()
    context = {'form': form, 'result': result, 'uploaded_image': image_url, 'confidence': confidence}
    return render(request, 'classifier/index.html', context)