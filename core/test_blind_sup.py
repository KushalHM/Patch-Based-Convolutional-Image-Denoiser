from sklearn.metrics import mean_squared_error
from .models import make_model
import numpy as np
import math, gc

from keras.optimizers import Adam
from skimage import measure

import scipy.io as sio
from keras import backend as K

def fine_tuning_loss(y_true,y_pred):    #
    return K.mean(K.square(y_true[:,:,:,1]-(y_pred[:,:,:,0]*y_true[:,:,:,1]+y_pred[:,:,:,1])) + 2*y_pred[:,:,:,0]*K.square(y_true[:,:,:,2]) - K.square(y_true[:,:,:,2]))


def preprocessing(noisy_img):
    noisy_img /= 255.
    X_data = (noisy_img - 0.5) / 0.2
    # print (X_data.shape)
    X_data = X_data.reshape(1,noisy_img.shape[0],noisy_img.shape[1], 1)
    return X_data

def denoising(noisy_image, model):
    noisy_img = np.float32(noisy_image)
    img_x = noisy_image.shape[0]
    img_y = noisy_image.shape[1]
    
    X_data = preprocessing(noisy_img)
    
    returned_score = model.predict(X_data,batch_size=1, verbose=0)
    returned_score = np.array(returned_score)
    returned_score = returned_score.reshape(1,img_x,img_y,2)

    denoised_test_image = returned_score[0,:,:,0] * (noisy_img) + returned_score[0,:,:,1]

    return denoised_test_image