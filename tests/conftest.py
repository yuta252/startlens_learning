import csv
import os
import pathlib

import boto3
from moto import mock_s3
import pytest

from fetch.resource import S3Resource, S3Object
import settings


def pytest_addoption(parser):
    parser.addoption('--env', default='development', help='environment')


@pytest.fixture
def csv_data():
    return [{'class': 1, 'vector': [-1.5, 2.0, 1.0, 0.0]}]


@pytest.fixture
def csv_file(csv_data):
    csv_file_name = os.path.join(settings.base_dir, 'tmp', 'csv', '0.csv')
    with open(csv_file_name, 'w+') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['CLASS', 'VECTOR'])
        writer.writeheader()
        writer.writerow({
            'CLASS': csv_data[0]['class'],
            'VECTOR': csv_data[0]['vector'],
        })
    yield csv_file_name
    pathlib.Path(csv_file_name).unlink()


@pytest.fixture
def hdf5_dir():
    return os.path.join(settings.base_dir, 'tmp', 'hdf5')


@pytest.fixture
@mock_s3
def create_s3():
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket('startlens-media-storage')
    bucket.create()
    return s3_resource


@pytest.fixture
@mock_s3
def s3_resource():
    s3_resource = S3Resource(aws_access_key_id=settings.aws_access_key_id, aws_secret_access_key=settings.aws_secret_access_key)
    return s3_resource


@pytest.fixture
@mock_s3
def s3_object():
    s3_object = S3Object('uploads/picture/1/2/sample.jpg',
                         aws_access_key_id=settings.aws_access_key_id,
                         aws_secret_access_key=settings.aws_secret_access_key)
    s3_object.client.create_bucket(Bucket='startlens-media-storage')
    s3_object.client.put_object(Bucket='startlens-media-storage', Key='uploads/picture/1/2/sample.jpg')
    return s3_object


@pytest.fixture
def create_file():
    file_path = os.path.join(settings.base_dir, 'tmp', 'image.txt')
    path = pathlib.Path(file_path)
    path.touch()
    yield file_path
    path.unlink()


@pytest.fixture
@mock_s3
def s3_object_uploaded():
    s3_object = S3Object('uploads/picture/1/2/sample.jpg',
                         aws_access_key_id=settings.aws_access_key_id,
                         aws_secret_access_key=settings.aws_secret_access_key)
    s3_object.client.create_bucket(Bucket='startlens-media-storage')
    return s3_object
