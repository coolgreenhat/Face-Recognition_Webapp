from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import numpy as np
import urllib
import json
import cv2
import os
from keras.models import load_model
from keras import backend as K
import numpy as np
from keras.applications import VGG16
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def predict(request):
        K.clear_session()#Solution:TypeError: Cannot interpret feed_dict key as Tensor:
        if request.method == 'POST':
                image_read = request.FILES['image']
                # print(image_read.name)
                # print(image_read.size)
                vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
                celeb = ['Chris Evans', 'Chris Hemsworth', 'Mark Ruffalo', 'Robert Downey Jr.']

                def classify(testImageFile):
                        classifier = load_model('path/to/model')
                        from keras.preprocessing import image
                        test_image = image.load_img(testImageFile, target_size=(128, 128))
                        train_features = np.zeros(shape=(1, 4, 4, 512))
                        test_image = image.img_to_array(test_image)
                        test_image = np.expand_dims(test_image, axis=0)
                        train_features = vgg_conv.predict(test_image)
                        test_image = np.reshape(train_features, (1, 4 * 4 * 512))
                        result = classifier.predict(test_image)
                        result = celeb[np.argmax(result)]
                        # print('{}'.format(celeb[np.argmax(result)]))
                        return result

                p = classify(image_read)
                # print('{}'.format(celeb[np.argmax(p)]))
                return render(request,'face/home.html',{"person": p})


def home(request):
        return render(request, 'face/home.html')
