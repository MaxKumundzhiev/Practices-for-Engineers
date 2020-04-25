# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import os
import pandas as pd





if __name__ == '__main__':
    excel_path = '/Users/macbook/PycharmProjects/Python_Practices/PyTorch/Data/ImperialCollegeLondon/UWSpineCT-meta-data.xlsx'
    data = pd.read_excel(excel_path, header=1)


    data_shape = data.shape
    data_columns = data.columns

    nan_values = dict{{for key,value in }}

