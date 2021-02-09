import io
import os
import logging

import botocore
import boto3

import settings


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/fetch/resource.log')
logger.addHandler(handler)


DOWNLOAD_DIR = 'tmp'


class S3ObjectReadError(Exception):
    """Fail to read S3 object error"""


class S3AccessDeniedError(Exception):
    """Fail to access S3 resource without aws access key and secret access key"""


class S3Resource(object):
    """Manage S3 resource class to get S3 basic information"""
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None):
        """Instantiate S3 resource class
        
        Parameters
        ----------
        aws_access_key_id: str
            AWS access key which has access to S3 resource
        aws_secret_access_key: str
            AWS secret access key which has access to S3 resource
        """
        if aws_access_key_id is None or aws_secret_access_key is None:
            logger.error({
                'action': '__init__',
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key,
                'message': 'S3 resource access is denied.'
            })
            raise S3AccessDeniedError
        self.resource = boto3.resource('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        self.bucket = self.resource.Bucket(settings.bucket)

    def get_all_buckets(self) -> list:
        buckets = []
        for bucket in self.resource.buckets.all():
            logger.debug({'action': 'get_all_buckets', 'bucket': bucket.name})
            buckets.append(bucket.name)
        return buckets

    def get_all_objects(self) -> list:
        return [obj.key for obj in self.bucket.objects.all()]

    def get_filtered_by_prefix(self, prefix=settings.prefix_key) -> list:
        """Filter S3 objects by using the prefix of object key
        
        Parameters
        ----------
        prefix: str
            Prefix key to filter S3 object
        """
        return [obj.key for obj in self.bucket.objects.filter(Prefix=prefix)]


class S3Object(object):
    """Manage S3 file object class"""
    def __init__(self, file_path: str, aws_access_key_id=None, aws_secret_access_key=None):
        """Instantiate S3Object class
        
        Parameters
        ----------
        file_path: str
            The specified path to handle image file
            Set s3 object key excluding bucket name
        aws_access_key_id: str
            AWS access key which has access to S3 resource
        aws_secret_access_key: str
            AWS secret access key which has access to S3 resource
        """
        self.file_path = file_path
        self.client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        self.bucket = settings.bucket
    
    def get_class_label(self) -> int:
        """Obtain class label for training data
        Returns: int
            class label based on file path
        """
        return int(self.file_path.split('/')[-2])

    def download_image(self) -> None:
        """Download image file to a designated directory"""
        file_name = os.path.join(DOWNLOAD_DIR, self.file_path.split('/')[-1])
        try:
            self.client.download_file(self.bucket, self.file_path, file_name)
        except botocore.exceptions.ClientError as e:
            logger.error({'action': 'download_image', 'status': 'fail', 'file_path': self.file_path, 'message': e})
            raise
        logger.debug({'action': 'download_image', 'status': 'success', 'file_path': self.file_path, 'message': 'success to download file'})

    def get_bytes_image_on_memory(self):
        """Get image binary image data and load on memory
        Returns: str
            Image data in byte format
        """
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.file_path)
            if int(response['ResponseMetadata']['HTTPStatusCode']) >= 400:
                raise S3ObjectReadError
        except botocore.exceptions.ClientError as e:
            logger.error({'action': 'get_binary_image', 'status': 'fail', 'file_path': self.file_path, 'message': e})
            raise
        except S3ObjectReadError as e:
            logger.error({'action': 'get_binary_image', 'status': 'fail', 'file_path': self.file_path, 'message': e})
            raise
        logger.debug({'action': 'get_binary_image', 'status': 'success', 'file_path': self.file_path, 'message': 'success to get image'})
        bytes_image = response['Body'].read()
        io_image = io.BytesIO(bytes_image)
        return io_image
