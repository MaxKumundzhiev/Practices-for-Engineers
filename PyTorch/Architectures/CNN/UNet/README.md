# Detection and localization of vertebrae centroids from CT scans. 

# Data: 
The database consists of spine-focused (i.e. tightly cropped) CT scans of 125 patients with varying types of pathologies. For most patients, multiple scans from longitudinal examinations are available, resulting in overall 242 scans in the database. For each scan, manual annotations of vertebrae centroids are provided. The data has been acquired at the Department of Radiology at the University of Washington.
[Data Link](https://imperialcollegelondon.app.box.com/s/erhcm28aablpy1725lt93xh6pk31ply1)

# Scenario:
### Clone Repository
```bash
git clone git@github.com:KumundzhievMaxim/Practices-for-Engineers.git
```
```bash
cd PyTorch/Models_Architectures/UNet
```
### Setup Environment
```bash
conda create -n tf python=3.6
conda activate tf
```

```bash
pip install -r requirements.txt
```
### Setup Folder Structure
#### Manually
 - Create `training_dataset` and `testing_dataset` folders
    - After data preprocessing training and testing data will appear under paths
 - Create `plots` folder
    - Resulting image will appear under path
 - Create `samples` folder with dedicated structure as in `Project tree`
#### Automatically
 - `saved_models`, `checkpoints`, `logs` folders will be created automatically
 
### Project tree
```
.
├── README.md
├── generate_detection_samples.py
├── generate_identification_samples.py
├── keras_models
│   ├── detection.py
│   └── identification.py
├── learning_functions
│   ├── create_partition.py
│   ├── data_generator.py
│   └── perform_learning.py
├── losses_and_metrics
│   ├── dsc.py
│   └── keras_weighted_categorical_crossentropy.py
├── measure.py
├── plots
├── requirements.txt
├── samples
│   ├── detection
│   │   ├── testing
│   │   └── training
│   └── identification
│       ├── testing
│       └── training
├── saved_models
│   ├── detection.h5
│   └── identification.h5
├── testing_dataset
├── train_detection_model.py
├── train_identification_model.py
├── training_dataset
└── utility_functions
    ├── __init__.py
    ├── labels.py
    ├── opening_files.py
    ├── processing.py
    └── sampling_helper_functions.py
```

### Download data
-  First you must download the data from BioMedia: https://biomedia.doc.ic.ac.uk/data/spine/. In the dropbox package there are collections of spine scans called 'spine-1', 'spine-2', 'spine-3', 'spine-4' and 'spine-5', download and unzip these files and move all these scans into a directory called 'training_dataset'. You will also see a zip file called 'spine-test-data', download and unzip this file and rename it 'testing_dataset'.
    ### Preprocess data
    - Generate samples to train and test the `detection` model: 
    ```bash
    python generate_detection_samples.py 'training_dataset' 'samples/detection/training'
    python generate_detection_samples.py 'testing_dataset' 'samples/detection/testing' 
    ``` 
    - Generate samples to train and test the `identification` model:
    ```bash
   python generate_identification_samples.py 'training_dataset' 'samples/identification/training' 
   python generate_identification_samples.py 'testing_dataset' 'samples/identification/testing'
    ```
    ### Train models
    - Train the detection network:
    ```bash
    python train_detection_model.py 'samples/detection/training' 'samples/detection/testing' 'saved_models/detection.h5'
    ```
    - Train the identification network:
    ```bash
   python train_identification_model.py 'samples/identification/training' 'samples/identification/testing' 'saved_models/identification.h5'
    ```
   ### Predict
   - Run the full algorithm on the test data:
   ```bash
   python measure.py 'testing_dataset' 'saved_models/detection.h5' 'saved_models/identification.h5'
   ```
   
# Observed results:
 - During executing measure.py script outputs will look like:
```
6 testing_dataset/4778644.nii.gz

apply detection

[64 64 80] [32 32 40] (140, 140, 312) (201, 201, 389)

finished detection

apply identification

finished identification

finished multiplying

start aggregating

finish aggregating

start averages

C7 7959

T1 10731

T2 15923

T3 17780

T4 14246

T5 21202

T6 23281

T7 23431

T8 24238

T9 22769

T10 27983

T11 32709

T12 33462

L1 37416

finish averages

T1 C7

T2 T2

T3 T3

T4 T4

T5 T5

T6 T6

T7 T7

T8 T8

T9 T9

T10 T10

T11 T12

T12 L1

T1 [ 81.9662  36.4989 297.5   ] [81.0, 39.0, 288.0]

T2 [ 79.3494  45.0036 275.    ] [77.0, 46.0, 271.0]

T3 [ 73.4615  50.8914 257.5   ] [74.0, 52.0, 251.0]

T4 [ 69.5363  55.4709 240.    ] [73.0, 57.0, 235.0]

T5 [ 66.2653  59.069  215.    ] [68.0, 60.0, 215.0]

T6 [ 64.3027  60.3774 192.5   ] [63.0, 62.0, 193.0]

T7 [ 62.6671  59.3961 165.    ] [61.0, 61.0, 167.0]

T8 [ 62.9942  59.7232 140.    ] [62.0, 60.0, 143.0]

T9 [ 65.9382  57.1064 115.    ] [63.0, 59.0, 119.0]

T10 [70.5176 57.1064 85.    ] [69.0, 58.0, 97.0]

T11 [74.77   56.4522 57.5   ] [71.0, 57.0, 72.0]

T12 [78.6952 54.4896 27.5   ] [76.0, 57.0, 50.0]

average 7.756259886627397
```   
   
## IAM
**[Malsim Kumundzhiev](https://github.com/KumundzhievMaxim)**

[<img src="http://i.imgur.com/0o48UoR.png" width="35">](https://github.com/KumundzhievMaxim)             [<img src="https://i.imgur.com/0IdggSZ.png" width="35">](https://www.linkedin.com/in/maksim-kumundzhiev/)             [<img src="https://loading.io/s/icon/vzeour.svg" width="35">](https://www.kaggle.com/maximkumundzhiev)               
