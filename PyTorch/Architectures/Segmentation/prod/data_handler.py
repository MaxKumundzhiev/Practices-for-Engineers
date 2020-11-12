import os
from typing import List, Dict


import pandas as pd
from tqdm import tqdm

import cv2
import nrrd  
import nibabel as nib
from nilearn import plotting

import matplotlib.pyplot as plt

ROOT_DATASET_FOLDER = './data'

def create_patients_df(paths: List[str]):
    """Iterating over each folder:
        - rename original and segmented images
        - collect original and segmented images paths
        - count faild folders paths, where:
            faild folders <- missing one of [original | segmented] images

    Notes:
        list of patients_df columns:
            [original_path, segmented_path]

    Returns:
        patients_df (pd.DataFrame): summarizing patients dataframe 
    """

    rows, failed_folders = [], []
    for folder_index, folder in enumerate(tqdm(paths)):
        patient_folder_path = os.path.join(annotated_data_path, folder)
        content = os.listdir(patient_folder_path)

        if len(content) < 2:
            failed_folders.append(patient_folder_path)
            continue

        old_image_name, old_segmentation_name = content[0], content[1] # old images names
        new_image_name, new_segmentation_name = f'original_image_{folder_index}.nii', f'segmented_image_{folder_index}.nrrd'

        os.rename(f'{patient_folder_path}/{old_image_name}', f'{patient_folder_path}/{new_image_name}')
        os.rename (f'{patient_folder_path}/{old_segmentation_name}', f'{patient_folder_path}/{new_segmentation_name}')

        rows.append(
            {
                'original_path': f'{patient_folder_path}/{new_image_name}',
                'segmented_path': f'{patient_folder_path}/{new_segmentation_name}'
            }
        )
    patients_df = pd.DataFrame(rows)
    patients_df.to_csv('patients_df.csv')

    return patients_df, failed_folders



def create_nii_df(patients_df: pd.DataFrame):
    original_images = patients_df['original_path'].values
    images_amount = len(original_images) 
    print(f'Read Patients DataFrame. Amount of images to process: {images_amount}')
    nii_df = pd.DataFrame(columns=["image_id", "image_path", "image_context", "image_resolution", "slices_number"])

    for image_id, image_path in enumerate(tqdm(original_images)):
        print(f'Processing image {image_id}/{images_amount}')

        try:
            image = nib.load(image_path)
            image_data = image.get_fdata() # (128, 128, 54), where 54 denotes number of slices reiled to the image
            image_channels = len(image_data.shape)
        except Exception as e:
            print(f'[SILENCED] Error: {e} occured for {image_id}')
            continue
        
        slices_number = image_data.shape[2] # e.g.: 54
        width, height = image_data.shape[0], image_data.shape[1] # e.g.: 128, 128
        resolution = f'{width}*{height}' 
    
        rows = []
        if image_channels == 3:
            for slice_number in range(slices_number):
                image_context = image_data[:, : ,slice_number]
                buffer_row = {
                    'image_id': image_id,
                    'image_path': image_path,
                    'image_context': image_context,
                    'image_resolution': resolution,
                    'slices_number': slices_number
                }
                rows.append(buffer_row)
        else:
            continue

    nii_df = pd.DataFrame(rows)
    nii_df.to_csv('./nii_images.csv')

    return nii_df


def plot_nii_image(image_path: str):
    Nifti_img  = nib.load(image_path)
    nii_data = Nifti_img.get_fdata()
    nii_aff  = Nifti_img.affine
    nii_hdr  = Nifti_img.header
    if(len(nii_data.shape)==3):
        for slice_Number in range(nii_data.shape[2]):
            plt.imshow(nii_data[:,:,slice_Number ])
            plt.show()
    if(len(nii_data.shape)==4):
        for frame in range(nii_data.shape[3]):
            for slice_Number in range(nii_data.shape[2]):
                plt.imshow(nii_data[:,:,slice_Number,frame])
                plt.show()


def create_nrrd_df(patients_df: pd.DataFrame):
    segmented_images = patients_df['segmented_path'].values
    images_amount = len(segmented_images) 
    print(f'Read Patients DataFrame. Amount of images to process: {images_amount}')
    nrrd_df = pd.DataFrame(columns=["image_id", "image_path", "image_context", "image_resolution", "slices_number"])

    for image_id, image_path in enumerate(tqdm(segmented_images)):
        print(f'Processing image {image_id}/{images_amount}')

        try:
            image_data, _ = nrrd.read(image_path)
            image_channels = len(image_data.shape)
        except Exception as e:
            # skipping unexisting segmented images
            print(f'[SILENCED] Error: {e} occured for {image_id}')
            continue
        
        slices_number = image_data.shape[2] # e.g.: 54
        width, height = image_data.shape[0], image_data.shape[1] # e.g.: 128, 128
        resolution = f'{width}*{height}' 
    
        rows = []
        if image_channels == 3:
            for slice_number in range(slices_number):
                image_context = image_data[:, : ,slice_number]
                buffer_row = {
                    'image_id': image_id,
                    'image_path': image_path,
                    'image_context': int(image_context),
                    'image_resolution': resolution,
                    'slices_number': slices_number
                }
                rows.append(buffer_row)
        else:
            continue

    nrrd_df = pd.DataFrame(rows)
    nrrd_df.to_csv('./nrrd_images.csv')

    return nrrd_df

if __name__ == "__main__":
    annotated_data_path = f'{ROOT_DATASET_FOLDER}/annotated_data/slicer'
    paths = sorted(os.listdir(annotated_data_path))

    patients_df, failed_folders = create_patients_df(paths=paths)
    nii_df = create_nii_df(patients_df)
    nrrd_df = create_nrrd_df(patients_df)
    
    
    
    

        
    

