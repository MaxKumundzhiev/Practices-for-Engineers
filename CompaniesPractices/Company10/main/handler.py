# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import argparse
from common.extractor import Extractor







def run(**paths):
    # # check whether file exists
    # assert paths['lim_path'].exists() and paths['lim_path'].is_file (), f'File: {paths["lim_path"]} does not exist.'
    # assert paths.pac_path.exists () and paths.pac_path.is_file (), f'File: {paths.pac_path} does not exist.'
    # assert paths.ris_path.exists () and paths.ris_path.is_file (), f'File: {paths.ris_path} does not exist.'

    extractor = Extractor(**paths)

    pac_accession_number_ids, result_dict = extractor.process_pac()

    ris_patients_ids, result_dict = extractor.process_ris(pac_accession_number_ids, result_dict)




if __name__ == '__main__':
    # get input arguments
    args = argparse.Namespace(
        lim_path='/Users/macbook/Documents/GitRep/PracticesForEngineers/Practices-for-Engineers/CompaniesPractices/Company10/samples/lims.txt',
        pac_path='/Users/macbook/Documents/GitRep/PracticesForEngineers/Practices-for-Engineers/CompaniesPractices/Company10/samples/pacs.json.csv',
        ris_path='/Users/macbook/Documents/GitRep/PracticesForEngineers/Practices-for-Engineers/CompaniesPractices/Company10/samples/ris.csv'
    )

    run(**vars(args))

