import json
import time
import logging

import boto3

# Setup our logger to work with both AWS CloudWatch and locally
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Setup our boto3 clients
s3 = boto3.client('s3')


def capture_post(event, context):
    logger.info(f"Hi from Lambda: {event}")

    response = s3.list_buckets()
    logger.info(f"S3 Buckets: {response}")
    return 'Batch works'
