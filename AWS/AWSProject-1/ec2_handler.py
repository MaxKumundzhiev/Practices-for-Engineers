

import os
import glob
import argparse 
from time import sleep

import boto3

ImageId = 'ami-0bb3fad3c0286ebd5'


class EC2Instance:
    def __init__(self, InstanceType, MaxCount, MinCount, EC2KeyName, Action):
        self.ImageId = ImageId
        self.InstanceType = InstanceType
        self.MaxCount = MaxCount
        self.MinCount = MinCount
        self.EC2KeyName = EC2KeyName
        self.Action = Action

    @staticmethod
    def create_ec2_key():
        try: 
            ec2 = boto3.resource('ec2')
            aws_ec2_keypair_file = open('ec2-keypair.pem', 'w')  
            key_pair = ec2.create_key_pair(KeyName=self.EC2KeyName)
            KeyPairOut = str(key_pair.key_material)
            aws_ec2_keypair_file.write(KeyPairOut)
        except Exception as e:
            return e
        return self.EC2KeyName

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
            instance = ec2.create_instances(
                ImageId=self.ImageId,
                InstanceType=self.InstanceType,
                MinCount=self.MinCount,
                MaxCount=self.MaxCount,
                KeyName=self.EC2KeyName
                )
            
            # ASSUMED WE GOT FILE WITH NECESSARY INSTANCE INFORAMTION
            # @TODO resolve issue with creating in file. 
            # {File "launch_ec2.py", line 100, in __call__
            # instance_name, instance_id, instance_public_ip = self.launch_instance()
            # TypeError: 'TypeError' object is not iterable}


            sleep(60) # give a time to be triggered and launched EC2 sleep for 60 seconds

            # get ID and IP address of the created instance 
            instance_name = instance[u'State'][u'Name']
            instance_id = instance.get(u'ip')
            instance_public_ip = instance.get(u'PublicIpAddress')

            ec2_instance_info = open(f'{instance_id}.txt', 'w')  
            context = str(f'Instance Name: {instance_name}, Insatnce ID: {instance_id}, Instance Public IP: {instance_public_ip}')
            ec2_instance_info.write(context)    

        except Exception as e:
            return e

        return instance_name, instance_id, instance_public_ip
    
    # @TODO finish terminate func
    @staticmethod
    def terminate_instance(insatnce_id: None):
        if insatnce_id == None:
            root_dir = os.getcwdb()
            file = glob.glob(f'{root_dir}/*.txt')
            with open(file, 'r') as f:
                context = f.readlines()
            return root_dir

        try:
            ec2 = boto3.resource('ec2')
            ec2.instances.filter(InstanceIds=insatnce_id).terminate()
        except Exception as e:
            return e
        return 'Terminated EC2 Instance {} with ID: {}. Public IP address: {}'.format(instance_name, instance_id, instance_public_ip)
    
    # @TODO finish pause func
    @staticmethod
    def pause_instance():
        try:
            ec2 = boto3.resource('ec2')
            ec2.instances.filter(InstanceIds=insatnce_id).stop()
        except Exception as e:
            return e
        return 'Pasued EC2 Instance {} with ID: {}. Public IP address: {}'.format(instance_name, instance_id, instance_public_ip)


    def __call__(self):
        if not self.EC2KeyName:
            EC2KeyName = self.create_ec2_key()
            self.chmod_key()
            return 'Created EC2 Key Pair with name {}.'.format(EC2KeyName)

        elif self.EC2KeyName and self.Action == 'create':    
            instance_name, instance_id, instance_public_ip = self.launch_instance()
            return 'Created EC2 Instance {} with ID: {}. Public IP address: {}. Instance Information stored under: {}-{}.txt'.format(instance_name, instance_id, instance_public_ip, instance_name, instance_id)
        
        elif self.Action == 'pause':
            self.pause_instance()

        elif self.Action == 'terminate':
            context = self.terminate_instance(insatnce_id=None)

        return context

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC2 arguments parser.')

    parser.add_argument('--MaxCount', required=True, type=int, default=1)
    parser.add_argument('--MinCount', required=True, type=int, default=1)
    parser.add_argument('--Action', required=True, type=str, default='create')
    parser.add_argument('--EC2KeyName', required=True, type=str, default='EC2-AWSProject1')

    parser.add_argument('--InstanceType', type=str, default='t2.micro')
    

    args = parser.parse_args()

    ec2_insatnce = EC2Instance(args.InstanceType, args.MaxCount, args.MinCount, args.EC2KeyName, args.Action)
    ec2_insatnce()

    


