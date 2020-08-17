

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict

from pydicom.dataset import Dataset

from common.external.utils import salty_encode, jitter_date
from settings import LOGGER


class Transformer:
    def __init__(self, lim_path, pac_path, ris_path):
        self.lim_file = Path(lim_path)
        self.pac_file = Path(pac_path)
        self.ris_file = Path(ris_path)

    def transform_pac(self) -> pd.DataFrame:
        """Per each patient collect and write target data into result dict.

        Notes:
            applied inplace changes for:
                - "patient_date_of_birth": shifted date
                - "study_date": shifted date
                - "study_id": encoded id
                - "study_instance_uid": encoded id
                - "study_status_id": encoded id

        Returns:
            pd.DataFrame: resulting PAC DataFrame
        """

        with open(self.pac_file, 'r') as f:
            pac_data = f.readlines()

        rows = []

        for patient_index, patient in enumerate(tqdm(pac_data)):
            buf_dicom = Dataset.from_json(patient)

            # PAC Patient
            patient_id = buf_dicom.PatientID
            patient_gender = buf_dicom.PatientSex
            patient_date_of_birth = jitter_date(buf_dicom.PatientBirthDate)
            patient_access_number = buf_dicom.AccessionNumber

            # PAC Study
            study_id = salty_encode(str(buf_dicom.StudyID))
            study_date = jitter_date(buf_dicom.StudyDate)
            study_time = buf_dicom.StudyTime
            study_description = buf_dicom.StudyDescription
            study_instance_uid = salty_encode(str(buf_dicom.StudyInstanceUID))
            study_status_id = salty_encode(str(buf_dicom.StudyStatusID))
            study_comments = buf_dicom.StudyComments

            buffer_row = {
                'patient_uid': patient_id,
                'sex': patient_gender,
                'date_of_birth': patient_date_of_birth,  # @TODO obtaining from RIS file
                'patient_access_number_id': patient_access_number,
                'studies': {'StudyID': study_id,
                            'StudyDate': study_date,
                            'StudyTime': study_time,
                            'StudyDescription': study_description if not len(study_description) == 0 else np.nan,  # check condition
                            'StudyInstanceUID': study_instance_uid,
                            'StudyStatusID': study_status_id,
                            'StudyComments': study_comments if not len(study_comments) == 0 else np.nan,  # check condition
                            }
            }
            rows.append(buffer_row)
        return pd.DataFrame.from_dict(rows)

    def transform_ris(self) -> pd.DataFrame:
        """Per each row collect and write target data into result dict.

        Notes:
            - "side" value formed as combination
             of outcome_l and outcome_r values

            - "opinion" value formed as voting
             between 2 values.

        Returns:
            pd.DataFrame: resulting RIS DataFrame
        """

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

    def transform_lim(self) -> pd.DataFrame:
        """Per each row transform and write target data into result dict.

        Notes:
            - found one record with missing {OBR: observation_id}
            - formed 2 types of patient_id:
                - internal_id: e.g.: a45931c9-b895-4b49-bf25-a7ac5cc81754
                - external_id e.g.: MVZa45931c9-b895-4b49-bf25-a7ac5cc81754

            - dict of transcriptions for "result_status":
                C --> Record coming over is a correction and thus replaces a final result
                D --> Deletes the OBX record
                F --> Final results; Can only be changed with a corrected result.
                I --> Specimen in lab; results pending
                P --> Preliminary results
                R --> Results entered -- not verified
                S --> Partial results
                U --> Results status change to final without retransmitting results already sent as ‘preliminary.’
                E.g., radiology changes status from preliminary to final
                W --> Post original as wrong, e.g., transmitted for wrong patient
                X --> Results cannot be obtained for this observation

            - in current ETL transcription dict applied just for "C"

        Returns:
            pd.DataFrame: resulting LIM DataFrame
        """

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
                assert len(cleaned_line) == 13, 'Wrong len of MSH record.'
                msh = separator.join(cleaned_line[1:])

            if key == 'EVN':
                assert len(cleaned_line) == 7, 'Wrong len of EVN record.'
                evn = separator.join(cleaned_line[1:])

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
                            'observation_result_status': lims_observation_result_status
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
    ) -> List[Dict]:
        """Per each treatment (RIS id) gather information
        from resulting RIS, PAC, LIM dataframes into resulting dict

        Args:
            ris_df (pd.DataFrame): resulting RIS dataframe
            pac_df (pd.DataFrame): resulting RIS dataframe
            lims_df (pd.DataFrame): resulting RIS dataframe

        Returns:
            rows (List[Dict]): resulting dict

        """

        LOGGER.info('Creating final file.')
        rows = []

        for treatment_index, treatment_id in enumerate(tqdm(ris_df['id'])):
            LOGGER.info(f'Processing {treatment_index+1}/{len(ris_df)} treatments.')

            ris_patient_id = ris_df[ris_df['id'] == treatment_id]['pat_id'].item()  # for mapping with LIM

            patient_uid = pac_df[pac_df['patient_access_number_id'] == treatment_id]['patient_uid'].values[0]
            gender = pac_df[pac_df['patient_access_number_id'] == treatment_id]['sex'].values[0]

            date_of_birth = ris_df[ris_df['id'] == treatment_id]['pat_dob'].values[0]  # 19690722
            date_of_birth = jitter_date(str(date_of_birth))  # 1969.08.02

            # studies
            dicom_studies = list(pac_df[pac_df['patient_access_number_id'] == treatment_id]['studies'].values)
            radiology_studies = ris_df['rad'].iloc[treatment_index]
            pathology = lims_df[lims_df['id'] == ris_patient_id]

            # check whether there is pathology record data
            if pathology.empty:
                pathology_studies = {
                    'date': np.nan,
                    'opinion': np.nan
                }
            else:
                pathology_studies = list(pathology['patho'].values)

            buff_row = {
                'patient_uid': salty_encode(patient_uid),
                'sex': gender,
                'date_of_birth': date_of_birth,
                'studies': dicom_studies,
                'rad': radiology_studies,
                'patho': pathology_studies
            }
            rows.append(buff_row)
        return rows

