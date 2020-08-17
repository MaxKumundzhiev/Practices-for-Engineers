# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


import json
from pathlib import Path

from settings import LOGGER
from common.transformer import Transformer


def run(
        lim_path: str,
        pac_path: str,
        ris_path: str,
        path_to_save: str
):
    """Main function which:
        - retrieve and transform PAC file.
        - retrieve and transform RIS file.
        - retrieve and transform LIM file.
        - create result dictionary within target format processing each treatment.
        - write result dictionary into imaginary_partner_patients.txt file.

    Args:
        lim_path (str): path to LIM.txt file
        pac_path (str): path to PAC.json.csv file
        ris_path (str): path to RIS.csv file
        path_to_save (str): path to save result file
    """

    LOGGER.info('Started Kheiron test ETL job.')

    lim_file = Path(lim_path)
    pac_file = Path(pac_path)
    ris_file = Path(ris_path)

    assert lim_file.exists() and lim_file.is_file(), f'File: {lim_file} does not exist.'
    assert pac_file.exists() and pac_file.is_file(), f'File: {pac_file} does not exist.'
    assert ris_file.exists() and ris_file.is_file(), f'File: {ris_file} does not exist.'

    transformer = Transformer(lim_file, pac_file, ris_file)

    pac_df = transformer.transform_pac()
    LOGGER.info('Retrieved and transformed PAC file.')

    ris_df = transformer.transform_ris()
    LOGGER.info('Retrieved and transformed RIS file.')

    lim_df = transformer.transform_lim()
    LOGGER.info('Retrieved and transformed LIM file.')

    result_dict = transformer.process_treatment(ris_df, pac_df, lim_df)
    LOGGER.info('Processed all treatments.')

    LOGGER.info(f'Write result file to {path_to_save}')
    output_file = open(path_to_save, 'w')
    for dic in result_dict:
        json.dump(dic, output_file)
        output_file.write("\n")

    LOGGER.info('Kheiron test ETL job finished.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Kheiron test ETL job transforms and write the Partner's"
                    "dedicated data into imaginary_partner_patients.txt file."
    )
    parser.add_argument('--lim', type=str, required=True, help='Local path to LIM file')
    parser.add_argument('--ris', type=str, required=True, help='Local path to RIS file')
    parser.add_argument('--pac', type=str, required=True, help='Local path to PAC file')
    parser.add_argument('--save', type=str, required=True, help='Local path to save result file')

    arguments = parser.parse_args()

    run(lim_path=arguments.lim,
        ris_path=arguments.ris,
        pac_path=arguments.pac,
        path_to_save=arguments.save)



