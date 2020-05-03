# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------
#from tensorflow.keras.backend import sum, abs, square
# Careul with mul <- inherited from tensorflow.keras.backend square

from torch import sum, abs, pow


def dice_coef_label(label):
    def dice_coef(y_true, y_pred, smooth=1):
        """Calculate Dice Coefficient

        Notes: https://arxiv.org/pdf/1606.04797v1.pdf

        """

        y_true_label = y_true[:, :, :, :, label]
        y_pred_label = y_pred[:, :, :, :, label]
        intersection = sum(abs(y_true_label * y_pred_label), axis=-1)
        return (2. * intersection + smooth) / (
                    sum(pow(y_true_label, 2), -1) + sum(pow(y_pred_label, 2), -1) + smooth)
    return dice_coef
