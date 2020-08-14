# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import os
import json
import logging

import pandas as pd
from tqdm import tqdm

from common.hl7read import process_file



if __name__ == '__main__':
    import argparse
    import pprint

    # Setup logging
    handlers = [logging.StreamHandler()]
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s',
        handlers=handlers)

    # Setup argument parsing
    # parser = argparse.ArgumentParser(description='Process HL7 file')

    # parser.add_argument(
    #     'fpath',
    #     metavar='hl7_file_path',
    #     type=str,
    #     help='path to the input HL7 file')
    #
    # args = parser.parse_args()
    # fpath = args.fpath

    parser = argparse.Namespace(
        fpath='/Users/macbook/Documents/GitRep/PracticesForEngineers/Practices-for-Engineers/CompaniesPractices/Company10/samples/lims.txt'
    )

    parser = vars(parser)

    logging.info('Start processing file {}'.format(parser['fpath']))
    results = process_file(parser['fpath'], logging)
    for index, result in enumerate(results):
        print(f'Result # {index}')
        for item in result:
            print(item)

    # pprint.pprint([res for res in result])
    logging.info('Finished processing file {}'.format(parser['fpath']))








