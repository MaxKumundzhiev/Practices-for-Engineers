# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import os
import sys
import zipfile
import logging
import numpy as np

import nibabel as nib

from typing import List, Tuple


class DataHandler:
    def __init__(self, input_train_data_path: str = '', input_test_data_path: str = ''):
        self.input_train_data_path = input_train_data_path
        self.input_test_data_path = input_test_data_path

    @staticmethod
    def unzip_input_folders(path: str = ''):
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

    @staticmethod
    def transform_train_data(path_to_train_folder: str):
        """Apply affain transformation for train data.
        Save transformed images to the dedicated folder with .nii extension.

        Returns:
             paths_to_images: List[str]:
                list of paths of unzipped train images

        """
        try:
            path_to_gz_files = []
            train_images_paths = []
            for root, dirs, files in os.walk(path_to_train_folder, topdown=False):
                for name in files:
                    buf_path = os.path.join(root, name)
                    if buf_path.endswith('.gz'):
                        path_to_gz_file = buf_path
                        path_to_gz_files.append(path_to_gz_file)
            logging.info(f'Handled all paths. Starting transforming.')
            files_len = len(path_to_gz_files)
            for file_index, gz_file in enumerate(path_to_gz_files):
                if gz_file.endswith('.gz'):
                    file_name = gz_file.split('/')[-1].replace('.gz', '')
                    path_to_save = (
                        '/Users/macbook/Documents/University/UniversityELTE/Mediso_SpineCT_Data/TrainInput/Transformed')
                    path_save_transformed_image = os.path.join(path_to_save, file_name)
                    logging.info(f'Started transforming {file_index}/{files_len}')
                    array_data = nib.load(gz_file).get_fdata()
                    affine = nib.load(gz_file).affine
                    array_img = nib.Nifti1Image(array_data.astype(np.float64), affine)
                    nib.save(array_img, path_save_transformed_image)
                    logging.info(f'Finished transforming {file_index}/{files_len}')
                    train_images_paths.append(path_save_transformed_image)
            logging.info('Transformation finished')
            return train_images_paths
        except:
            logging.info("Unexpected error:", sys.exc_info()[0])
            raise


    def run(self):
        logging.getLogger().setLevel(logging.INFO)
        #DataHandler.unzip_input_folders(self.input_test_data_path)
        #train_images_paths = DataHandler.transform_train_data(self.input_train_data_path)
        #return train_images_paths

if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train', '-tr',
    #                     help='Path to train data folder', required=True, type=str)
    # parser.add_argument('--test', '-te',
    #                     help='Path to test data folder', required=True, type=str)
    #
    # args = vars(parser.parse_args())
    # DataHandler().run(**args)

    DataHandler(
        input_test_data_path='/Users/macbook/Documents/University/UniversityELTE/Mediso_SpineCT_Data/Test',
        input_train_data_path='/Users/macbook/Documents/University/UniversityELTE/Mediso_SpineCT_Data/Train').run()

