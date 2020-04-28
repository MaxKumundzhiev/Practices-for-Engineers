# Implementation of UNet Neural Net with PyTorch. 

# Data: 
The database consists of spine-focused (i.e. tightly cropped) CT scans of 125 patients with varying types of pathologies. For most patients, multiple scans from longitudinal examinations are available, resulting in overall 242 scans in the database. For each scan, manual annotations of vertebrae centroids are provided. The data has been acquired at the Department of Radiology at the University of Washington.
[Data Link](https://imperialcollegelondon.app.box.com/s/erhcm28aablpy1725lt93xh6pk31ply1)

# Scenario:
### Download data
-  First you must download the data from BioMedia: https://biomedia.doc.ic.ac.uk/data/spine/. In the dropbox package there are collections of spine scans called 'spine-1', 'spine-2', 'spine-3', 'spine-4' and 'spine-5', download and unzip these files and move all these scans into a directory called 'training_dataset'. You will also see a zip file called 'spine-test-data', download and unzip this file and rename it 'testing_dataset'.
    ### Transform data
    - You must then generate samples to train and test the detection network: 
    ```bash
    python generate_detection_samples.py 'training_dataset' 'samples/detection/training' python generate_detection_samples.py 'testing_dataset' 'samples/detection/testing'
    ``` 
    - You must then generate samples to train and test the identification network:
    ```bash
     python generate_identification_samples.py 'training_dataset' 'samples/identification/training' python generate_identification_samples.py 'testing_dataset' 'samples/identification/testing'
    ```
    ### Train models
    - Now train the detection network:
    ```bash
    python train_detection_model.py 'samples/detection/training' 'samples/detection/testing' 'saved_models/detection.h5'
    ```
    - Now train the identification network:
    ```bash
   python train_identification_model.py 'samples/identification/training' 'samples/identification/testing' 'saved_models/identification.h5'
    ```
   ### Predict on test data
   - You can now run the full algorithm on the test data:
   ```bash
   python measure.py 'testing_dataset' 'saved_models/detection.h5' 'saved_models/identification.h5'
   ```
   
   
## IAM
**[Malsim Kumundzhiev](https://github.com/KumundzhievMaxim)**

[<img src="http://i.imgur.com/0o48UoR.png" width="35">](https://github.com/KumundzhievMaxim)             [<img src="https://i.imgur.com/0IdggSZ.png" width="35">](https://www.linkedin.com/in/maksim-kumundzhiev/)             [<img src="https://loading.io/s/icon/vzeour.svg" width="35">](https://www.kaggle.com/maximkumundzhiev)               