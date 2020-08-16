

import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from pydicom.dataset import Dataset

from common.external.utils import salty_encode, jitter_date
from settings import LOGGER


class Transformer:
    def __init__(self, lim_path, pac_path, ris_path):
        self.lim_file = Path(lim_path)
        self.pac_file = Path(pac_path)
        self.ris_file = Path(ris_path)

    def transform_pac(self):
        LOGGER.info('Transforming PAC file.')
        with open(self.pac_file, 'r') as f:
            pac_data = f.readlines()  # List[nested dicts],

        rows = []

        for patient_index, patient in enumerate(tqdm(pac_data)):
            buf_dicom = Dataset.from_json(patient)

            # PAC Patient
            patient_id = buf_dicom.PatientID
            patient_gender = buf_dicom.PatientSex
            patient_date_of_birth = jitter_date(buf_dicom.PatientBirthDate)
            patient_access_number = buf_dicom.AccessionNumber

            # PAC Study
            study_id = buf_dicom.StudyID
            study_date = jitter_date(buf_dicom.StudyDate)
            study_time = buf_dicom.StudyTime
            study_description = buf_dicom.StudyDescription
            study_instance_uid = buf_dicom.StudyInstanceUID
            study_status_id = buf_dicom.StudyStatusID
            study_comments = buf_dicom.StudyComments

            buffer_row = {
                'patient_uid': patient_id,
                'sex': patient_gender,
                'date_of_birth': patient_date_of_birth,  # @TODO obtaining from RIS file
                'patient_access_number_id': patient_access_number,
                'studies': {'StudyID': salty_encode(str(study_id)),
                            'StudyDate': study_date,
                            'StudyTime': study_time,
                            'StudyDescription': study_description if not len(study_description) == 0 else np.nan,  # check condition
                            'StudyInstanceUID': salty_encode(str(study_instance_uid)),
                            'StudyStatusID': salty_encode(str(study_status_id)),
                            'StudyComments': study_comments if not len(study_comments) == 0 else np.nan,  # check condition
                            }
            }
            rows.append(buffer_row)
        return pd.DataFrame.from_dict(rows)

    def transform_ris(self):
        LOGGER.info('Transforming RIS file.')

        ris_df = pd.read_csv(self.ris_file)
        ris_df['rad'] = np.nan * len(ris_df)

        rows = []

        for row_index, row in enumerate(tqdm(ris_df.itertuples())):
            voice_1 = row.outcome_l
            voice_2 = row.outcome_r

            sides = (row.outcome_l, row.outcome_r)
            date = jitter_date(str(row.date))

            buff_row = {
                'side': sides,
                'date': date,
                'opinion': voice_1 if voice_1 == voice_2 else (voice_1, voice_2)
            }
            rows.append(buff_row)
        ris_df['rad'] = rows
        return ris_df

    def transform_lim(self):
        LOGGER.info('Transforming LIM file.')

        count = 0
        separator = '-'

        rows = []

        for _, line in enumerate(tqdm(open(self.lim_file))):

            cleaned_line = re.sub('[\W_]+', ' ', line).split()

            if len(cleaned_line) == 0:  # skip blank line
                continue

            key = cleaned_line[0]

            assert key in ['MSH', 'EVN', 'PID', 'OBR', 'OBX'], 'Wrong key detected.'

            if key == 'MSH':
                assert len (cleaned_line) == 13, 'Wrong len of MSH record.'
                msh = separator.join (cleaned_line[1:])

            if key == 'EVN':
                assert len (cleaned_line) == 7, 'Wrong len of EVN record.'
                evn = separator.join (cleaned_line[1:])

            if key == 'PID':
                assert len(cleaned_line) == 8, 'Wrong len of PID record.'
                lims_patient_external_id = (separator.join(cleaned_line[1:6]))[3:]
                lims_patient_internal_id = separator.join(cleaned_line[1:6])

            if key == 'OBR':
                if len(cleaned_line) != 10:
                    LOGGER.warning('Found OBR record with missing observation_id. Filled with NaN.')
                    lims_observation_request_id = np.nan
                else:
                    lims_observation_request_id = separator.join(cleaned_line[1:6])
                    lims_requested_date = jitter_date(cleaned_line[6])
                    lims_observation_date = jitter_date(cleaned_line[7])
                    lims_observation_status = cleaned_line[8]

            if key == 'OBX':
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
                # if U --> Results status change to final without retransmitting results already sent as ‘preliminary.’
                # E.g., radiology changes status from preliminary to final
                # if W --> Post original as wrong, e.g., transmitted for wrong patient
                # if X --> Results cannot be obtained for this observation
            count += 1

            if count == 5:
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
                            'observation_request_id': salty_encode(lims_observation_request_id),
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

        LOGGER.info('-Creating final file.')

        rows = []

        for treatment_index, treatment_id in enumerate(tqdm(ris_df['id'])):
            LOGGER.info(f'Processing {treatment_index+1}/{len(ris_df)} treatments.')

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
            rows.append(buff_row)
        return rows

