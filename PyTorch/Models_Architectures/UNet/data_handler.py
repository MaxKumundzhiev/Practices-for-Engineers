# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import pandas as pd
import nibabel as ni
import matplotlib.pyplot as plt

import os
import zipfile
import logging


class DataHandler:
    def __init__(self, input_train_data_path: str = '', input_test_data_path: str = ''):
        self.input_train_data_path = input_train_data_path
        self.input_test_data_path = input_test_data_path

    @staticmethod
    def unzip_folder(path: str = ''):
        """Unzip input folder and remove zipped folder:

        Returns:
              True if done else False

        """
        try:
            if path != '':
                path_to_main_folder = path
                folders = [folder for folder in os.listdir(path_to_main_folder) if folder.endswith('.zip')]
                for folder in folders:
                    path_to_zip_folder = os.path.join(path_to_main_folder, folder)
                    logging.info(f'Started unzipping {path_to_zip_folder.split("/")[-1]}')
                    with zipfile.ZipFile(f'{path_to_zip_folder}', 'r') as zip_ref:
                        path_to_save_folder = os.path.join(path_to_main_folder, path_to_zip_folder.split("/")[-1].split(".")[0])
                        zip_ref.extractall(f'{path_to_save_folder}')
                        logging.info(f'Finished unzipping {path_to_zip_folder.split ("/")[-1]} with status True.')
                        os.remove(path_to_zip_folder)
                        logging.info(f'Removed {path_to_zip_folder} with status True.')
                return True
            else:
                logging.info('Provided path does not exists OR empty. Please provide correct path.')
        except:
            return logging.info(f'Error occurred during unzipping {path_to_zip_folder.split("/")[-1]} folder.')

    def run(self):
        logging.getLogger().setLevel(logging.INFO)
        DataHandler.unzip_folder(self.input_train_data_path)


if __name__ == '__main__':
    DataHandler('/Users/macbook/Documents/University/UniversityELTE/Mediso_SpineCT_Data/Train').run()
