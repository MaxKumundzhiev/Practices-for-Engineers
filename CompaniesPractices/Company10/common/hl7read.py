def process_file(fpath, logger):
    """Process an HL7 file with the given fpath.

    Returns: an array with all the messages
    """

    message = []
    for line in open(fpath):
        line = line.rstrip('\n')
        line = line.strip()
        if line[:3] in ['FHS', 'BHS', 'FTS', 'BTS']:
            continue
        if line[:3] == 'MSH':
            if message:
                yield message
            message = [line]
        else:
            if len(message) == 0:
                logger.error(
                    'Segment received before message header [%s]',
                    line)
                continue
            if line:
                message.append(line)



# if __name__ == '__main__':
#     import argparse
#     import logging
#     import pprint
#
#     # Setup logging
#     handlers = []
#     handlers.append(logging.StreamHandler())
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format='%(asctime)s:%(levelname)s:%(message)s',
#         handlers=handlers)
#
#     # Setup argument parsing
#     parser = argparse.ArgumentParser(description='Process HL7 file')
#
#     parser.add_argument(
#         'fpath',
#         metavar='hl7_file_path',
#         type=str,
#         help='path to the input HL7 file')
#
#     args = parser.parse_args()
#     fpath = args.fpath
#
#     logging.info('Start processing file {}'.format(fpath))
#     result = process_file(fpath, logging)
#     pprint.pprint([res for res in result])
#     logging.info('Finished processing file {}'.format(fpath))
