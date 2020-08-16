# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


import json
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


from settings import LOGGER
from common.transformer import Transformer
from common.external.utils import salty_encode

# change UPPERCASE M to LOWERCASE within datetime


def run(lim_path: str,
        pac_path: str,
        ris_path: str):

    LOGGER.info('Started Kheiron test ETL job.')

    lim_file = Path(lim_path)
    pac_file = Path(pac_path)
    ris_file = Path(ris_path)

    assert lim_file.exists() and lim_file.is_file(), f'File: {lim_file} does not exist.'
    assert pac_file.exists() and pac_file.is_file(), f'File: {pac_file} does not exist.'
    assert ris_file.exists() and ris_file.is_file(), f'File: {ris_file} does not exist.'

    transformer = Transformer(lim_file, pac_file, ris_file)

    pac_df = transformer.transform_pac()
    LOGGER.info('Retrieved and Transformed PAC file.')

    ris_df = transformer.transform_ris()
    LOGGER.info('Retrieved and Transformed RIS file.')

    lim_df = transformer.transform_lim()
    LOGGER.info('Retrieved and Transformed LIM file.')

    result_rows = Transformer.process_treatment(ris_df, pac_df, lim_df)
    LOGGER.info('Finished processing all treatments.')

    with open("/Users/macbook/Desktop/imaginary_partner_patients.txt", "w") as file:
        file.write(json.dumps(result_rows))
    LOGGER.info(f'Write result file to {"/Users/macbook/Desktop/imaginary_partner_patients.txt"}.')
    LOGGER.info('Kheiron test ETL job finished.')


if __name__ == '__main__':
    import argparse

    args = argparse.Namespace(
        lim_path='/Users/macbook/Documents/GitRep/PracticesForEngineers/Practices-for-Engineers/CompaniesPractices/Company10/samples/lims.txt',
        pac_path='/Users/macbook/Documents/GitRep/PracticesForEngineers/Practices-for-Engineers/CompaniesPractices/Company10/samples/pacs.json.csv',
        ris_path='/Users/macbook/Documents/GitRep/PracticesForEngineers/Practices-for-Engineers/CompaniesPractices/Company10/samples/ris.csv'
    )

    run(**vars(args))


