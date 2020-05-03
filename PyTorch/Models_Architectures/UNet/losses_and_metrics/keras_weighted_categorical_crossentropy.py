# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

from tensorflow.keras.backend import sum, clip, epsilon, log, variable

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

    weights = variable(weights)

    def loss(y_true, y_pred):
        y_pred /= sum(y_pred, axis=-1, keepdims=True)
        y_pred = clip(y_pred, epsilon(), 1 - epsilon())
        loss = y_true * log(y_pred) * weights
        loss = -sum(loss, -1)
        return loss
    return loss
