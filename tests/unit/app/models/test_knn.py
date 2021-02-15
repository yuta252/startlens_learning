import csv
import os
import pathlib

import botocore
from moto import mock_s3
import pytest

from app.model.knn import Hdf5Model, NoFileFoundError
from app.model.knn import CsvModel, NoSpotIdError
from app.model.knn import KnnModel, NoDirectoryFoundError
import settings


@mock_s3
class TestHdf5Model(object):
    bucket = 'startlens-media-storage'

    # @classmethod
    # def setup_class(cls):
    #     cls.hdf5Model = Hdf5Model()

    # @classmethod
    # def teardown_class(cls):
    #     del cls.hdf5Model

    def setup_method(self, method):
        print(f'method={method.__name__}')
        self.hdf5_file_name = "embedding_20201212_043210.hdf5"
        self.file_path = os.path.join(settings.base_dir, 'tmp', 'hdf5', "embedding_20201212_043210.hdf5")
        self.path = pathlib.Path(self.file_path)
        self.path.touch()

    def teardown_method(self, method):
        print(f'method={method.__name__}')
        self.path.unlink()

    def test_get_hdf5_file_path(self, hdf5_dir):
        hdf5_model = Hdf5Model(hdf5_dir=hdf5_dir, hdf5_file_name="embedding_20201212_043210.hdf5")
        assert hdf5_model.file_path == self.file_path

    def test_get_class_label_from_path_raise(self):
        with pytest.raises(NoFileFoundError):
            Hdf5Model()

    def test_get_file_from_s3(self, hdf5_dir, create_s3):
        s3_key = os.path.join('hdf5', self.hdf5_file_name)
        create_s3.Object(TestHdf5Model.bucket, s3_key).put()
        
        hdf5_model = Hdf5Model(hdf5_dir=hdf5_dir, hdf5_file_name="embedding_20201212_043210.hdf5")
        is_downlaod = hdf5_model.get_file_from_s3()
        assert is_downlaod is True

    def test_get_file_from_s3_raises(self, hdf5_dir, create_s3):
        with pytest.raises(botocore.exceptions.ClientError):
            hdf5_model = Hdf5Model(hdf5_dir=hdf5_dir, hdf5_file_name="embedding_20201212_043210.hdf5")
            hdf5_model.get_file_from_s3()


@mock_s3
class TestCsvModel(object):
    bucket = 'startlens-media-storage'

    def setup_method(self, method):
        self.spot_id = 0
        self.csv_file_name = str(self.spot_id) + '.csv'
        self.csv_dir = os.path.join(settings.base_dir, 'tmp', 'csv')
        self.csv_file_path = os.path.join(self.csv_dir, self.csv_file_name)

    def teardown_method(self, method):
        if os.path.exists(self.csv_file_path):
            pathlib.Path(self.csv_file_path).unlink()

    def test_initialize_class_raises(self):
        with pytest.raises(NoSpotIdError):
            CsvModel()

    def test_get_file_from_s3(self, create_s3):
        s3_key = os.path.join('csv', self.csv_file_name)
        create_s3.Object(TestCsvModel.bucket, s3_key).put()

        csv_model = CsvModel(spot_id=self.spot_id, is_write=False, csv_dir=self.csv_dir)
        is_csv = csv_model.get_file_from_s3()
        assert is_csv is True
    
    def test_get_file_from_s3_raises(self, create_s3):
        s3_key = os.path.join('csv', '5.csv')
        create_s3.Object(TestCsvModel.bucket, s3_key).put()
        
        csv_model = CsvModel(spot_id=self.spot_id, is_write=False, csv_dir=self.csv_dir)
        csv_model.get_file_from_s3()
        assert os.path.exists(os.path.join(self.csv_dir, '0.csv'))

    def test_load_data(self, csv_file, csv_data):
        csv_model = CsvModel(spot_id=self.spot_id, is_write=False, csv_dir=self.csv_dir)
        classes, vector_images = csv_model.load_data()
        actual_classes = csv_data[0]['class']
        actual_vector_images = csv_data[0]['vector']
        assert classes[0] == actual_classes and vector_images[0] == actual_vector_images

    def test_save_all_data(self):
        classes = [1, 2, 4, 1]
        images = [
            [-1.5, 2.0, 1.1, 2.3],
            [2.5, 1.2, -0.1, 0.3],
            [3.2, -1.0, 1.1, 2.3],
            [2.3, 1.1, 0.6, 0.6],
        ]
        csv_model = CsvModel(spot_id=self.spot_id, is_write=True, csv_dir=self.csv_dir)
        csv_model.save_all_data(classes, images)

        assert os.path.exists(self.csv_file_path)

        actual_classes = []
        actual_images = []
        with open(self.csv_file_path, 'r+') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                imgs = [float(s) for s in eval(row['VECTOR'])]
                actual_images.append(imgs)
                actual_classes.append(int(row['CLASS']))
        assert classes == actual_classes and images == actual_images
    
    def test_add_new_data(self, csv_file, csv_data):
        csv_model = CsvModel(spot_id=self.spot_id, is_write=True, csv_dir=self.csv_dir)
        added_classes = [1, 2, 4, 1]
        added_images = [
            [-1.5, 2.0, 1.1, 2.3],
            [2.5, 1.2, -0.1, 0.3],
            [3.2, -1.0, 1.1, 2.3],
            [2.3, 1.1, 0.6, 0.6],
        ]
        csv_model.add_new_data(added_classes, added_images)
        actual_classes, actual_images = csv_model.load_data()
        added_classes.insert(0, csv_data[0]['class'])
        added_images.insert(0, csv_data[0]['vector'])
        assert actual_classes == added_classes and actual_images == added_images


@mock_s3
class TestKnnModel(object):
    bucket = 'startlens-media-storage'

    def setup_method(self, method):
        self.spot_id = 0
        self.knn_file_name = str(self.spot_id) + '.pkl'
        self.knn_dir = os.path.join(settings.base_dir, 'tmp', 'knn')
        self.knn_file_path = os.path.join(self.knn_dir, self.knn_file_name)

    def teardown_method(self, method):
        if os.path.exists(self.knn_file_path):
            pathlib.Path(self.knn_file_path).unlink()

    def test_initialize_class_raises(self):
        with pytest.raises(NoSpotIdError):
            KnnModel(spot_id=None, is_write=False, knn_dir='tmp')
    
    def test_initialize_class_raises_no_directory(self):
        with pytest.raises(NoDirectoryFoundError):
            KnnModel(spot_id=1, is_write=False)

    def test_get_knn_file_from_s3(self, create_s3):
        s3_key = os.path.join('knn', self.knn_file_name)
        create_s3.Object(TestKnnModel.bucket, s3_key).put()

        knn_model = KnnModel(spot_id=self.spot_id, is_write=False, knn_dir=self.knn_dir)
        is_knn = knn_model.get_knn_file_from_s3()
        assert is_knn is True

    def test_get_knn_file_from_s3_create_file(self):
        knn_model = KnnModel(spot_id=self.spot_id, is_write=False, knn_dir=self.knn_dir)
        knn_model.get_knn_file_from_s3()
        assert os.path.exists(self.knn_file_path)

    def test_save_trained_model(self):
        obj = {'model': 'knn', 'status': 'start'}
        knn_model = KnnModel(spot_id=self.spot_id, is_write=True, knn_dir=self.knn_dir)
        knn_model.save_trained_model(obj)
        assert os.path.exists(self.knn_file_path)

    def test_load_trained_model(self):
        obj = {'model': 'knn', 'status': 'start'}
        knn_model = KnnModel(spot_id=self.spot_id, is_write=False, knn_dir=self.knn_dir)
        knn_model.save_trained_model(obj)

        actual_obj = knn_model.load_trained_model()
        assert actual_obj == obj
