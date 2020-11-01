

import os
import boto3
import argparse 

ImageId = 'ami-0bb3fad3c0286ebd5'


class EC2Instance:
    def __init__(self, InstanceType, MaxCount, MinCount):
        self.ImageId = ImageId
        self.InstanceType = InstanceType
        self.MaxCount = MaxCount
        self.MinCount = MinCount

    @staticmethod
    def create_ec2_key():
        try: 
            ec2 = boto3.resource('ec2')
            aws_ec2_keypair_file = open('ec2-keypair.pem', 'w')  
            key_pair = ec2.create_key_pair(KeyName='EC2-AWSProject1')
            KeyPairOut = str(key_pair.key_material)
            aws_ec2_keypair_file.write(KeyPairOut)
        except Exception as e:
            return e
        return KeyPairOut

    @staticmethod
    def chmod_key():
        try:
            cmd = 'chmod 400 ec2-keypair.pem'
            os.system(cmd)
        except Exception as e:
            return e
        return True

    def launch_instance(self):
        try:
            ec2 = boto3.resource('ec2')
            ec2.create_instances(ImageId=self.ImageId, InstanceType=self.InstanceType, MinCount=self.MinCount, MaxCount=self.MaxCount)
        except Exception as e:
            return e
        return True
    
    def __call__(self):
        KeyPairOut = self.create_ec2_key()
        self.chmod_key()
        return 'Created and Chmod EC2 Key Pair.'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC2 arguments parser.')

    parser.add_argument('--InstanceType', required=True, type=str)
    parser.add_argument('--MaxCount', required=True, type=str)
    parser.add_argument('--MinCount', required=True, type=str)

    args = parser.parse_args()

    ec2_insatnce = EC2Instance(args.InstanceType, args.MaxCount, args.MinCount)
    print(ec2_insatnce())

    


