import csv
import logging
import os
import pathlib
import pickle

import botocore

from fetch.resource import S3Object
import settings


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/app/model/knn.log')
logger.addHandler(handler)


CSV_COLUMN_CLASS = 'CLASS'
CSV_COLUMN_VECTOR = 'VECTOR'


class NoSpotIdError(Exception):
    """Spot Id Not Found"""


class NoFileFoundError(Exception):
    """File Not Found"""


class NoDirectoryFoundError(Exception):
    """Directory Not Found"""


class FileModel(object):
    """Base file model"""
    def __init__(self, file_path):
        self.file_path = file_path


class Hdf5Model(FileModel):
    """Hdf5Model class to manage trained hdf5 file"""
    def __init__(self, hdf5_dir=None, hdf5_file_name=None, *args, **kwargs):
        """Instanciate Hdf5Model

        Parameters
        ----------
        hdf5_dir: str
            file directory
        hdf5_file_name: str
            file path
        """
        if hdf5_dir is None or hdf5_file_name is None:
            raise NoFileFoundError
        self.hdf5_dir = hdf5_dir
        self.hdf5_file_name = hdf5_file_name
        self.file_path = os.path.join(self.hdf5_dir, self.hdf5_file_name)
        if not os.path.exists(self.file_path):
            self.get_file_from_s3()
        super().__init__(self.file_path, *args, **kwargs)

    def get_file_from_s3(self) -> bool:
        """Get hdf5 file from S3 storage

        Return: bool
            If success to download file, returns True
        """
        logger.debug({'action': 'get_file_from_s3', 'status': 'start', 'file_path': self.file_path, 'message': 'start to downlaod hdf5 file from S3'})
        s3_key = os.path.join('hdf5', self.hdf5_file_name)
        s3_client = S3Object(s3_key,
                             aws_access_key_id=settings.aws_access_key_id,
                             aws_secret_access_key=settings.aws_secret_access_key)
        download_dir = os.path.join(settings.base_dir, 'tmp', 'hdf5')
        try:
            s3_client.download_file(download_dir)
        except botocore.exceptions.ClientError as e:
            logger.error({'action': 'get_file_from_s3', 'status': 'fail', 'message': e})
            raise
        logger.debug({'action': 'get_file_from_s3', 'status': 'end', 'file_path': self.file_path, 'message': 'finish to downlaod hdf5 file from S3'})
        return True


class CsvModel(FileModel):
    """CsvModel class to manage csv file that triplet prediction result is saved"""
    def __init__(self, spot_id=None, is_write=False, csv_dir=None, *args, **kwargs):
        """Instantiate CsvModel class

        Parameters
        ----------
        spot_id: int
            sightseeing spot id. Each csv file managed by each spot
        is_write: bool
            the flag if csv file is writable
        csv_dir: str
            file directory, /tmp/csv path as default
        """
        if spot_id is None:
            raise NoSpotIdError
        self.spot_id = spot_id
        self.file_name = str(self.spot_id) + '.csv'
        if not csv_dir:
            raise NoDirectoryFoundError
        self.csv_dir = csv_dir
        self.file_path = os.path.join(self.csv_dir, self.file_name)
        self.is_write = is_write
        if not os.path.exists(self.file_path) and not self.is_write:
            self.get_file_from_s3()
        super().__init__(self.file_path, *args, **kwargs)
        self.columns_class = CSV_COLUMN_CLASS
        self.columns_vector = CSV_COLUMN_VECTOR
        self.columns = [CSV_COLUMN_CLASS, CSV_COLUMN_VECTOR]
        self.classes = []
        self.vector_images = []

    def get_file_from_s3(self) -> bool:
        """Get csv file form S3 resource
        If not get s3 object, create new csv file to read

        Returns: bool
            If success to download csv file or create new file, returns True
        """
        logger.info({'action': 'get_file_from_s3', 'status': 'start', 'message': 'start to downlaod csv file from S3', 'file_path': self.file_path})
        s3_key = os.path.join('csv', self.file_name)
        s3_client = S3Object(s3_key,
                             aws_access_key_id=settings.aws_access_key_id,
                             aws_secret_access_key=settings.aws_secret_access_key)
        try:
            s3_client.download_file(self.csv_dir)
        except botocore.exceptions.ClientError as e:
            logger.error({
                'action': 'get_file_from_s3',
                'status': 'fail',
                'message': e,
                'file_path': self.file_path
            })
            pathlib.Path(self.file_path).touch()
        return True
    
    def load_data(self) -> tuple:
        """Load csv data

        Returns: tuple(list[int], list[list[float]])
            2 lists, the list of classification class and the one of 50 dimentional vector data that image is conveted
        """
        with open(self.file_path, 'r+') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                images = [float(s) for s in eval(row[self.columns_vector])]
                self.classes.append(int(row[self.columns_class]))
                self.vector_images.append(images)
        return self.classes, self.vector_images

    def save_all_data(self, classes: list, images: list) -> None:
        """Save class and converted vector data to csv file

        Parameters
        ----------
        classes: list
            the list of classification class
        images: list
            the list of 50 dimentional vector data of image
        """
        logger.info({'action': 'save_all_data', 'status': 'start', 'spot_id': self.spot_id, 'message': 'start to write csv file'})
        with open(self.file_path, 'w+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.columns)
            writer.writeheader()

            for class_, image in zip(classes, images):
                writer.writerow({
                    self.columns_class: class_,
                    self.columns_vector: image
                })
                logger.info({'action': 'save_all_data', 'status': 'writing', 'class': class_, 'message': 'writing class, image vector as csv row'})
        logger.info({'action': 'save_all_data', 'status': 'end', 'spot_id': self.spot_id, 'message': 'finish to write csv file'})

        # Save csv to S3 as backup
        logger.info({'action': 'save_all_data', 'status': 'start', 'spot_id': self.spot_id, 'message': 'start to upload csv to S3'})
        s3_key = os.path.join('csv', self.file_name)
        s3_client = S3Object(s3_key,
                             aws_access_key_id=settings.aws_access_key_id,
                             aws_secret_access_key=settings.aws_secret_access_key)
        s3_client.upload_file(self.file_path)
        logger.info({'action': 'save_all_data', 'status': 'end', 'spot_id': self.spot_id, 'message': 'finish to uplaod csv to S3'})

    def add_new_data(self, classes: list, images: list) -> None:
        """Save class and converted vector data to csv file with each exhibit update

        Parameters
        ----------
        classes: list
            the list of classification class
        images: list
            the list of 50 dimentional vector data of image
        """
        logger.info({'action': 'save_new_data', 'status': 'start', 'spot_id': self.spot_id, 'message': 'start to write csv file'})
        with open(self.file_path, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.columns)
            # writer.writeheader()

            for class_, image in zip(classes, images):
                writer.writerow({
                    self.columns_class: class_,
                    self.columns_vector: image
                })
                logger.info({'action': 'save_new_data', 'status': 'writing', 'class': class_, 'message': 'writing class, image vector as csv row'})
        logger.info({'action': 'save_new_data', 'status': 'end', 'spot_id': self.spot_id, 'message': 'finish to write csv file'})

        # Save csv to S3 as backup
        logger.info({'action': 'add_new_data', 'status': 'start', 'spot_id': self.spot_id, 'message': 'start to upload csv to S3'})
        s3_key = os.path.join('csv', self.file_name)
        s3_client = S3Object(s3_key,
                             aws_access_key_id=settings.aws_access_key_id,
                             aws_secret_access_key=settings.aws_secret_access_key)
        s3_client.upload_file(self.file_path)
        logger.info({'action': 'add_new_data', 'status': 'end', 'spot_id': self.spot_id, 'message': 'finish to upload csv to S3'})


class KnnModel(FileModel):
    """KnnModel class to manage trained knn pickle file"""
    def __init__(self, spot_id=None, is_write=False, knn_dir=None, *args, **kwargs):
        """Instantiate KnnModel class

        Parameters
        ----------
        spot_id: int
            sightseeing spot id is required
        is_write: bool
            the flag if knn file is writable
        knn_file: str
            file path
        """
        if spot_id is None:
            raise NoSpotIdError
        self.spot_id = spot_id
        self.file_name = str(self.spot_id) + '.pkl'
        if not knn_dir:
            raise NoDirectoryFoundError
        self.knn_dir = knn_dir
        self.file_path = os.path.join(self.knn_dir, self.file_name)
        self.csv_dir = os.path.join(settings.base_dir, 'tmp', 'csv')
        self.is_write = is_write
        print('test1')
        if not os.path.exists(self.file_path) and not self.is_write:
            print('test2')
            self.get_knn_file_from_s3()
        super().__init__(self.file_path, *args, **kwargs)

    def get_knn_file_from_s3(self) -> bool:
        """Get knn file from S3 resource
        If not get s3 object, create new pickle file to read

        Returns: bool
            If success to download csv file or create new file, returns True
        """
        logger.info({'action': 'get_knn_file_from_s3', 'status': 'start', 'message': 'start to downlaod knn file from S3'})
        s3_key = os.path.join('knn', self.file_name)
        s3_client = S3Object(s3_key,
                             aws_access_key_id=settings.aws_access_key_id,
                             aws_secret_access_key=settings.aws_secret_access_key)
        print('test')
        try:
            s3_client.download_file(self.knn_dir)
        except botocore.exceptions.ClientError as e:
            logger.error({
                'action': 'get_knn_file_path',
                'status': 'fail',
                'message': e,
                'knn_file_path': self.file_path
            })
            pathlib.Path(self.file_path).touch()
        logger.info({'action': 'get_knn_file_from_s3', 'status': 'end', 'message': 'finish to downlaod knn file from S3'})
        return True

    def save_trained_model(self, model_obj) -> None:
        """Save model to pickle
        
        Parameters
        ----------
        model_obj:
            trained model to save such as knn
        """
        logger.debug({'action': 'save_train_data', 'status': 'start', 'knn_file_path': self.file_path})
        with open(self.file_path, 'wb') as pkl_file:
            pickle.dump(model_obj, pkl_file)
        logger.debug({'action': 'save_train_data', 'status': 'end', 'knn_file_path': self.file_path})

        # Save pkl to S3 as backup
        logger.info({'action': 'save_trained_model', 'status': 'start', 'spot_id': self.spot_id, 'message': 'start to upload pkl to S3'})
        s3_key = os.path.join('knn', self.file_name)
        s3_client = S3Object(s3_key,
                             aws_access_key_id=settings.aws_access_key_id,
                             aws_secret_access_key=settings.aws_secret_access_key)
        s3_client.upload_file(self.file_path)
        logger.info({'action': 'save_trained_model', 'status': 'end', 'spot_id': self.spot_id, 'message': 'finished to upload pkl to S3'})

    def load_trained_model(self):
        """Get trained model by each spot
        
        Returns:
            knn model
        """
        logger.debug({'action': 'load_train_data', 'status': 'start', 'knn_file_path': self.file_path})
        with open(self.file_path, 'rb') as pkl_file:
            neighbor = pickle.load(pkl_file)
        logger.debug({'action': 'load_train_data', 'status': 'end', 'knn_file_path': self.file_path})
        return neighbor

    def inference(self, input_data: list):
        """Inference knn model

        Parameters
        ----------
        input_data: list
            the list of 50 dimentional vector data derived from embedding model
            ex. [[0.1, 0.3, ..., -0.3]]
        Returns: list
            the list of tuple of inferenced class and distance
        """
        neighbor = self.load_trained_model()
        distances, neighbors = neighbor.kneighbors(input_data)
        distances, neighbors = distances.tolist(), neighbors.tolist()

        csv_file = CsvModel(self.spot_id, is_write=False, csv_dir=self.csv_dir)
        classes, vector_images = csv_file.load_data()

        result_list = []
        for distance, neighbor in zip(distances, neighbors):
            for d, n in zip(distance, neighbor):
                result_list.append((int(classes[n]), d))
        result_list.sort(key=lambda x: x[1])
        return result_list
