

import boto3
import argparse 



parser = argparse.ArgumentParser(description='EC2 arguments parser.')

parser.add_argument('--InstanceType', required=True, type=str)
parser.add_argument('--MaxCount', required=True, type=str)
parser.add_argument('--MinCount', required=True, type=str)
parser.add_argument('--ImageId', required=True, type=str)

args = parser.parse_args()
print(args)

