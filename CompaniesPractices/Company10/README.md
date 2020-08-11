# Forward Deployed Engineer challenge
In this exercise, we'll execute a data extraction from an imaginary data partner.
## Overview
The partner stores the data in the following three systems:
* PACS (Picture Archiving and Communication System): DICOM images
* RIS (Radiology Information System): Radiologist reports
* LIMS (Laboratory Information Management System): Pathology reports

The task is to merge all of this information to a single format which is usable for machine learning.

## Source files
* `ris.csv` all the data from the RIS. This contains the radiologist opinions on a case
* `pacs.json.csv` all the imaging data from the PACS, one line per image
* `lims.txt` all the pathology data from LIMS in HL7 format

### DICOM image metadata
Relevant DICOM tags (keys) are:
- Patient ID: 00100020
- Accession number: 00080050
- Date of Study: 00080020

### HL7
For reading in HL7 messages, a python tool `hl7read.py` is provided

## Target format
The machine learning team expects the data to be delivered in the following format. Every line should be a valid json.

`imaginary_partner_patients.txt`:
```json
{"patient_uid": "1.2.3.4", "sex": "F", "date_of_birth": "1955.05.14",  "studies": [...], "rad": [...], "patho": [...]}
{"patient_uid": "2.3.4.5", "sex": "F", "date_of_birth": "1964.03.05",  "studies": [...], "rad": [...], "patho": [...]}
...
```
* `patient_uid` is the Patient identifier
* `studies` are all the DICOM studies
* `rad` is an array where each element is a radiologist report. The structure of the elements is arbitrary, the only requiremnt is that it should contain the following information: `side`, `date`, `opinion`
* `patho` is an array where each element is a patholgy report. The structure of the elements is arbitrary, the only requiremnt is that it should contain the following information: `date`, `opinion`

## Tasks
### Create a target dataset from the source files and point out consistency issues
Design and code a pipeline that gathers all the data and merges them one file (see target format). There are some inconsistencies in the dataset - point them out and resolve them whenever possible.
### De-identify data
* Replace all identifiers with new ones
* Keep the mappings for future use
* Jitter dates - shift dates with a random number of days

## What we review
* Clarity of code, answers and presentation
* Correctness of the information extracted and presented.
## Delivery
Please send all code snippets, visualization results, schemas, reports and queries in a zipped file via e-mail.
