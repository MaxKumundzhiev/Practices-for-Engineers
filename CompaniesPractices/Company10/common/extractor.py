

import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from pydicom.dataset import Dataset


class Extractor:
    def __init__(self, lim_path, pac_path, ris_path):
        self.lim_file = Path(lim_path)
        self.pac_file = Path(pac_path)
        self.ris_file = Path(ris_path)

    def process_pac(self):
        with open(self.pac_file, 'r') as f:
            pac_data = f.readlines()  # List[nested dicts], where each item of list (904 items) is dedicated examination

        rows = []
        pac_patients_ids = []
        pac_accession_number_ids = []

        # Iterate per each patient
        for patient_index, patient in enumerate(tqdm(pac_data)):
            buf_dicom = Dataset.from_json(patient)

            # Patient ID
            patient_id = buf_dicom.PatientID  # can be wrong, need check with assert later
            pac_patients_ids.append(patient_id)  # for insights comparison

            # Patient Gender
            patient_gender = buf_dicom.PatientSex

            # Date of birth
            patient_date_of_birth = buf_dicom.PatientBirthDate

            # @TODO Patient's date of birth is same for all PATIENTS
            # @TODO It means we can become more confident if take d_o_b from "RIS"
            patient_date_of_birth = datetime.strptime(str(patient_date_of_birth), '%Y%M%d')
            patient_date_of_birth = f'{patient_date_of_birth.year}.{patient_date_of_birth.month}.{patient_date_of_birth.day}'

            # Access number
            patient_access_number = buf_dicom.AccessionNumber
            pac_accession_number_ids.append(patient_access_number)  # for mapping with RIS data

            # PAC Studies
            study_id = buf_dicom.StudyID
            study_date = buf_dicom.StudyDate
            study_time = buf_dicom.StudyTime
            study_description = buf_dicom.StudyDescription
            study_instance_uid = buf_dicom.StudyInstanceUID
            study_status_id = buf_dicom.StudyStatusID
            study_comments = buf_dicom.StudyComments

            buffer_row = {
                'patient_uid': patient_id,
                'sex': patient_gender,
                # @TODO Patient's date of birth is same for all PATIENTS
                'date_of_birth': patient_date_of_birth,
                'patient_accession_number_id': patient_access_number,
                'studies': [{'StudyID': study_id if not len(study_id) == 0 else np.nan,
                             'StudyDate': study_date if not len(study_date) == 0 else np.nan,
                             'StudyTime': study_time if not len(study_time) == 0 else np.nan,
                             'StudyDescription': study_description if not len(study_description) == 0 else np.nan,
                             'StudyInstanceUID': study_instance_uid if not len(study_instance_uid) == 0 else np.nan,
                             'StudyStatusID': study_status_id if not len(study_status_id) == 0 else np.nan,
                             'StudyComments': study_comments if not len(study_comments) == 0 else np.nan
                             }]
            }

            rows.append(buffer_row)

        pac_df = pd.DataFrame.from_dict(rows)
        pac_df.drop_duplicates(subset='patient_uid', ignore_index=True, inplace=True)  # drop duplicated records {290 unique records}


        return pac_df

    def process_ris(self, pac_accession_number_ids):
        rows = []

        ris_df = pd.read_csv(self.ris_file)
        columns = ['outcome_l', 'outcome_r']

        ris_patients_ids = []

        for patient_access_number in pac_accession_number_ids:
            # patient_id
            patient_id = ris_df[ris_df['id'] == patient_access_number]['pat_id'].item()
            ris_patients_ids.append(patient_id)

            # retrieve sides
            opinions = ris_df[ris_df['id'] == patient_access_number][columns]
            voice_1 = opinions.iloc[0]['outcome_l']
            voice_2 = opinions.iloc[0]['outcome_r']

            # Date
            ris_date = ris_df[ris_df['id'] == patient_access_number]['date'].item()
            ris_date = datetime.strptime (str (ris_date), '%Y%M%d')
            ris_date = f'{ris_date.year}.{ris_date.month}.{ris_date.day}'

            buffer_row = {
                'patient_uid': patient_id,
                'patient_accession_number_id': patient_access_number,
                'rad': [{
                    'side': (voice_1, voice_2),
                    'date': ris_date,
                    'opinion': voice_1 if voice_1 == voice_2 else (voice_1, voice_2)
                    }]
                }

            rows.append(buffer_row)


        return ris_patients_ids, rows


    def retrieve_lim(self):
        rows = []
        count = 0
        separator = '-'
        lims_patients_external_ids = []

        for line in open(self.lim_file):
            retrieved_line = re.sub ('[\W_]+', ' ', line).split ()
            # assert cleaned_line[0] in ['MSH', 'EVN', 'PID', 'OBR', 'OBX'], 'Wrong key detected.'
            print (retrieved_line)
            if len (retrieved_line) == 0:
                print ('FOund blank line')
                continue
            # work with PID {Personal Patient Information} key
            if retrieved_line[0] == 'PID':
                print ('PROCESSED PID')
                assert len (retrieved_line) == 8, 'Wrong len of PID record.'

                lims_patient_external_id = (separator.join (retrieved_line[1:6]))[3:]
                lims_patient_internal_id = separator.join (retrieved_line[1:6])

                lims_patients_external_ids.append (lims_patient_external_id)

                print (lims_patient_external_id)
                print (lims_patient_internal_id)

            if retrieved_line[0] == 'OBR':  # work with OBR key
                print ('PROCESSED OBR')
                if len (retrieved_line) != 10:
                    print ('Found OBR record with missing ID.')
                    # ID
                    lims_observation_request_id = np.nan

                    # requested_date
                    lims_requested_date = retrieved_line[1]
                    lims_requested_date = datetime.strptime (lims_requested_date, '%Y%M%d')
                    lims_requested_date = f'{lims_requested_date.year}.{lims_requested_date.month}.{lims_requested_date.day}'

                    # observation_date
                    lims_observation_date = retrieved_line[2]
                    lims_observation_date = datetime.strptime (lims_observation_date, '%Y%M%d')
                    lims_observation_date = f'{lims_observation_date.year}.{lims_observation_date.month}.{lims_observation_date.day}'

                    # observation_status
                    lims_observation_status = retrieved_line[3]

                else:
                    # ID
                    lims_observation_request_id = np.nan

                    # requested_date
                    lims_requested_date = retrieved_line[6]
                    lims_requested_date = datetime.strptime (lims_requested_date, '%Y%M%d')
                    lims_requested_date = f'{lims_requested_date.year}.{lims_requested_date.month}.{lims_requested_date.day}'

                    # observation_date
                    lims_observation_date = retrieved_line[7]
                    lims_observation_date = datetime.strptime (lims_observation_date, '%Y%M%d')
                    lims_observation_date = f'{lims_observation_date.year}.{lims_observation_date.month}.{lims_observation_date.day}'

                    # observation_status
                    lims_observation_status = retrieved_line[8]

                print(lims_observation_request_id)
                print(lims_requested_date)
                print(lims_observation_date)
                print(lims_observation_status)

            if retrieved_line[0] == 'OBX':  # work with OBX key
                print ('PROCESSED OBX')
                assert len (retrieved_line) == 6, 'Wrong len of OBR record.'

                #         lims_observation_sub_id = # if empty --> same as lims_observation_request_id
                #         lims_observation_value = # RES DIAG

                lims_observation_result_status = str (retrieved_line[5][0])
                lims_observation_probability = int (retrieved_line[5][1:]) / 100
                lims_result_status_repr = 'Record coming over is a correction and thus replaces a final result'


                print (lims_observation_result_status)
                print (lims_observation_probability)
                print (lims_result_status_repr)

            count += 1

            if count == 5:
                print ('Adding to buff dict to main dict.')
                buffer_rows = {
                    'MSH': np.nan,
                    'ENV': np.nan,
                    'PID': [{
                        'external_pat_id': lims_patient_external_id,
                        'internal_pat_id': lims_patient_internal_id
                    }],

                    'OBR': [{
                        'observation_request_id': lims_observation_request_id,
                        'observation_requested_date': lims_requested_date,
                        'observation_date': lims_observation_date,
                        'observation_status': lims_observation_status
                    }],

                    'OBX': [{
                        'observation_probability': lims_observation_probability,
                        'observation_result_status': lims_observation_result_status,
                        'observation_result_status_representation': lims_result_status_repr if lims_observation_result_status == 'C' else 'other'
                    }]
                }
                rows.append (buffer_rows)
                count = 0

            # write results to file
            with open("/Users/macbook/Desktop/lims_json.txt", "w") as file:
                file.write(json.dumps(rows))





