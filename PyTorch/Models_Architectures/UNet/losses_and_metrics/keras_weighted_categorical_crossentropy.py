# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

"""
A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
"""
#from keras import backend as K
#from tensorflow.keras.backend import backend as K

#from tensorflow.keras.backend import sum, clip, epsilon, log, variable

from torch import sum, log

#Fulfill: clip, sum???, epsilon


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    #weights = V.variable(weights)
    weights = variable(weights)

    def loss(y_true, y_pred):
        # y_true = K.print_tensor(y_true, message='y_true = ')
        # y_pred = K.print_tensor(y_pred, message='y_pred = ')
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = clip(y_pred, epsilon(), 1 - epsilon())
        # calc
        loss = y_true * log(y_pred) * weights
        loss = -sum(loss, -1)
        return loss

    return loss
