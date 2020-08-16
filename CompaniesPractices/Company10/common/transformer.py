

import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from pydicom.dataset import Dataset

from common.external.utils import salty_encode


class Transformer:
    def __init__(self, lim_path, pac_path, ris_path):
        self.lim_file = Path(lim_path)
        self.pac_file = Path(pac_path)
        self.ris_file = Path(ris_path)

    def transform_pac(self):
        # Read as PAC file
        with open(self.pac_file, 'r') as f:
            pac_data = f.readlines()  # List[nested dicts],
            # where each item of list (904 items)
            # is dedicated examination

        rows = []

        for patient_index, patient in enumerate(tqdm(pac_data)):
            buf_dicom = Dataset.from_json(patient)

            # Patient ID
            patient_id = buf_dicom.PatientID  # can be wrong, need check with assert later

            # Patient Gender
            patient_gender = buf_dicom.PatientSex

            # Date of birth
            patient_date_of_birth = buf_dicom.PatientBirthDate
            patient_date_of_birth = datetime.strptime(str(patient_date_of_birth), '%Y%M%d')
            # Patient date of birth is same for all PATIENTS(BUG IN DATA)
            ## IT MEANS BETTER TO GREP DATE OF BIRTH FROM "RIS" FILE
            patient_date_of_birth = f'{patient_date_of_birth.year}.{patient_date_of_birth.month}.{patient_date_of_birth.day}'

            # Access number
            patient_access_number = buf_dicom.AccessionNumber

            # PAC Studies {DICOM STUDIES}
            StudyID = buf_dicom.StudyID
            StudyDate = buf_dicom.StudyDate
            StudyTime = buf_dicom.StudyTime
            StudyDescription = buf_dicom.StudyDescription
            StudyInstanceUID = buf_dicom.StudyInstanceUID
            StudyStatusID = buf_dicom.StudyStatusID
            StudyComments = buf_dicom.StudyComments

            buffer_row = {
                'patient_uid': patient_id,
                'sex': patient_gender,
                'date_of_birth': patient_date_of_birth,  # should be obtained from RIS file
                'patient_access_number_id': patient_access_number,
                'studies': {'StudyID': salty_encode(str(StudyID)),
                            'StudyDate': StudyDate if not len(StudyDate) == 0 else np.nan, # check condition
                            'StudyTime': StudyTime if not len(StudyTime) == 0 else np.nan, # check condition
                            'StudyDescription': StudyDescription if not len(StudyDescription) == 0 else np.nan, # check condition
                            'StudyInstanceUID': StudyInstanceUID if not len(StudyInstanceUID) == 0 else np.nan, # check condition
                            'StudyStatusID': salty_encode(str(StudyStatusID)),
                            'StudyComments': StudyComments if not len(StudyComments) == 0 else np.nan, # check condition
                            }
            }

            rows.append(buffer_row)

        # pass dict with all records into pandas DataFrame

        return pd.DataFrame.from_dict(rows)

    def transform_ris(self):
        # read RIS data
        ris_df = pd.read_csv(self.ris_file)
        ris_df['rad'] = np.nan * len(ris_df)

        rows = []

        for row_index, row in enumerate (ris_df.itertuples ()):
            voice_1 = row.outcome_l
            voice_2 = row.outcome_r

            sides = (row.outcome_l, row.outcome_r)
            date = datetime.strptime (str (row.date), '%Y%M%d')

            buff_row = {
                'side': sides,
                'date': f'{date.year}.{date.month}.{date.day}',
                'opinion': voice_1 if voice_1 == voice_2 else (voice_1, voice_2)
            }
            rows.append(buff_row)

        ris_df['rad'] = rows

        return ris_df

    def transform_lim(self):
        count = 0
        separator = '-'

        rows = []

        for line in open(self.lim_file):

            cleaned_line = re.sub('[\W_]+', ' ', line).split()
            if len(cleaned_line) == 0:
                continue

            assert cleaned_line[0] in ['MSH', 'EVN', 'PID', 'OBR', 'OBX'], 'Wrong key detected.'

            if cleaned_line[0] == 'MSH':
                print ('PROCESSED MSH')
                assert len (cleaned_line) == 13, 'Wrong len of MSH record.'
                msh = separator.join (cleaned_line[1:])

            if cleaned_line[0] == 'EVN':
                print ('PROCESSED EVN')
                assert len (cleaned_line) == 7, 'Wrong len of EVN record.'
                evn = separator.join (cleaned_line[1:])

            if cleaned_line[0] == 'PID':
                print('PROCESSED PID')
                assert len (cleaned_line) == 8, 'Wrong len of PID record.'

                lims_patient_external_id = (separator.join (cleaned_line[1:6]))[3:]
                lims_patient_internal_id = separator.join (cleaned_line[1:6])

            if cleaned_line[0] == 'OBR':
                print ('PROCESSED OBR')
                if len (cleaned_line) != 10:
                    print ('Found OBR record with missing observation_id.')
                    # ID
                    lims_observation_request_id = np.nan

                    # requested_date
                    lims_requested_date = cleaned_line[1]
                    lims_requested_date = datetime.strptime (lims_requested_date, '%Y%M%d')
                    lims_requested_date = f'{lims_requested_date.year}.{lims_requested_date.month}.{lims_requested_date.day}'

                    # observation_date
                    lims_observation_date = cleaned_line[2]
                    lims_observation_date = datetime.strptime (lims_observation_date, '%Y%M%d')
                    lims_observation_date = f'{lims_observation_date.year}.{lims_observation_date.month}.{lims_observation_date.day}'

                    # observation_status
                    lims_observation_status = cleaned_line[3]

                else:
                    # ID
                    lims_observation_request_id = np.nan

                    # requested_date
                    lims_requested_date = cleaned_line[6]
                    lims_requested_date = datetime.strptime (lims_requested_date, '%Y%M%d')
                    lims_requested_date = f'{lims_requested_date.year}.{lims_requested_date.month}.{lims_requested_date.day}'

                    # observation_date
                    lims_observation_date = cleaned_line[7]
                    lims_observation_date = datetime.strptime (lims_observation_date, '%Y%M%d')

                    lims_observation_date = f'{lims_observation_date.year}.{lims_observation_date.month}.{lims_observation_date.day}'

                    # observation_status
                    lims_observation_status = cleaned_line[8]

            if cleaned_line[0] == 'OBX':  # work with OBX key
                print('PROCESSED OBX')
                assert len(cleaned_line) == 6, 'Wrong len of OBR record.'

                lims_observation_result_status = str(cleaned_line[5][0])
                lims_observation_probability = int(cleaned_line[5][1:]) / 100
                lims_result_status_repr = 'Record coming over is a correction and thus replaces a final result'
                # if C --> Record coming over is a correction and thus replaces a final result
                # if D --> Deletes the OBX record
                # if F --> Final results; Can only be changed with a corrected result.
                # if I --> Specimen in lab; results pending
                # if P --> Preliminary results
                # if R --> Results entered -- not verified
                # if S --> Partial results
                # if U --> Results status change to final without retransmitting results already sent as ‘preliminary.’  E.g., radiology changes status from preliminary to final
                # if W --> Post original as wrong, e.g., transmitted for wrong patient
                # if X --> Results cannot be obtained for this observation

            count += 1

            if count == 5:
                print ('Adding to buff dict to main dict.')
                buffer_rows = {
                    'id': lims_patient_external_id,
                    'patho': {
                        'observation_date': lims_observation_date,
                        'observation_result_status': lims_observation_result_status,
                        'observation_result_status_representation': lims_result_status_repr if lims_observation_result_status == 'C' else 'other',
                        'MSH': msh,
                        'EVN': evn,
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
                            'observation_date': lims_observation_date,
                            'observation_probability': lims_observation_probability,
                            'observation_result_status': lims_observation_result_status,
                            'observation_result_status_representation': lims_result_status_repr if lims_observation_result_status == 'C' else 'other'
                        }]
                    }
                }
                rows.append(buffer_rows)
                count = 0

        return pd.DataFrame.from_dict(rows)

    @staticmethod
    def process_treatment(
            ris_df: pd.DataFrame,
            pac_df: pd.DataFrame,
            lims_df: pd.DataFrame
    ) -> pd.DataFrame:

        rows = []

        for treatment_index, treatment_id in enumerate (tqdm (ris_df['id'])):
            patient_uid = pac_df[pac_df['patient_access_number_id'] == treatment_id] \
                ['patient_uid'].values[0]  # zeroed item, cause usually get array of equal values

            gender = pac_df[pac_df['patient_access_number_id'] == treatment_id] \
                ['sex'].values[0]

            date_of_birth = ris_df[ris_df['id'] == treatment_id] \
                ['pat_dob'].values[0]  # 19690722
            date_of_birth = datetime.strptime (str (date_of_birth), '%Y%M%d')  # 1969.07.22

            dicom_studies = list (pac_df[pac_df['patient_access_number_id'] == treatment_id] \
                                      ['studies'].values)  # LIST[DictS]

            radiology_studies = ris_df['rad'].iloc[treatment_index]

            pathology = lims_df[lims_df['id'] == treatment_id]

            if pathology.shape[0] == 0:
                pathology_studies = {
                    'date': np.nan,
                    'opinion': np.nan
                }

            else:
                pathology_studies = pathology['patho'].values  # List[Dicts]

            buff_row = {
                'patient_uid': salty_encode (patient_uid),
                'sex': gender,
                'date_of_birth': f'{date_of_birth.year}.{date_of_birth.month}.{date_of_birth.day}',
                'studies': dicom_studies,
                'rad': radiology_studies,
                'patho': pathology_studies
            }

            rows.append (buff_row)

        # return pd.DataFrame.from_dict(rows)
        return rows
