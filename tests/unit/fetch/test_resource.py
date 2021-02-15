import io
import os

import botocore
from moto import mock_s3
import pytest

from fetch.resource import S3Resource, S3AccessDeniedError, S3Object
import settings


@mock_s3
class TestS3Resource(object):
    bucket = 'startlens-media-storage'

    def test_initialize_class(self):
        with pytest.raises(S3AccessDeniedError):
            S3Resource()

    def test_get_all_buckets(self, s3_resource):
        buckets = ['storage', 'picture']
        for bucket in buckets:
            bucket = s3_resource.resource.Bucket(str(bucket))
            bucket.create()
        actual_buckets = s3_resource.get_all_buckets()
        assert actual_buckets[0] == buckets[0] and actual_buckets[1] == buckets[1]
    
    def test_get_all_objects(self, s3_resource):
        bucket = s3_resource.resource.Bucket(TestS3Resource.bucket)
        bucket.create()
        s3_keys = ['tmp/sample1.txt', 'tmp/sample2.txt']
        for s3_key in s3_keys:
            s3_resource.resource.Object(TestS3Resource.bucket, s3_key).put()
        actual_objects = s3_resource.get_all_objects()
        assert actual_objects[0] == s3_keys[0] and actual_objects[1] == s3_keys[1]

    def test_get_filtered_by_prefix(self, s3_resource):
        bucket = s3_resource.resource.Bucket(TestS3Resource.bucket)
        bucket.create()
        s3_keys = ['tmp/2020/sample1.txt', 'tmp/2020/sample2.txt']
        other_keys = ['tmp/2021/sample4.txt', 'tmp/2021/sample4.txt']
        for s3_key in s3_keys:
            s3_resource.resource.Object(TestS3Resource.bucket, s3_key).put()
        for s3_key in other_keys:
            s3_resource.resource.Object(TestS3Resource.bucket, s3_key).put()
        actual_objects = s3_resource.get_filtered_by_prefix('tmp/2020')
        assert actual_objects[0] == s3_keys[0] and actual_objects[1] == s3_keys[1]


@mock_s3
class TestS3Object(object):
    def test_initialize_class(self):
        with pytest.raises(S3AccessDeniedError):
            S3Object('uploads/picture/1/2/sample.jpg')
    
    def test_get_class_label(self, s3_object):
        assert s3_object.get_class_label() == 2

    def test_downlaod_file(self, s3_object):
        download_dir = os.path.join(settings.base_dir, 'tmp')
        s3_object.download_file(download_dir)
        assert os.path.exists(os.path.join(download_dir, 'sample.jpg'))
    
    def test_bytes_image_on_memory(self, s3_object):
        io_image = s3_object.get_bytes_image_on_memory()
        assert isinstance(io_image, io.BytesIO)

    def test_bytes_image_on_memory_raises(self, s3_object):
        s3_object.file_path = 'uploads/picture/1/2/sample2.jpg'
        with pytest.raises(botocore.exceptions.ClientError):
            s3_object.get_bytes_image_on_memory()

    def test_upload_file(self, s3_object, create_file):
        is_uploaded = s3_object.upload_file(create_file)
        assert is_uploaded is True

