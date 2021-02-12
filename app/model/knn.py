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
    """Spot Id is Not Found"""


class NoFileFoundError(Exception):
    """File Not Found"""


class FileModel(object):
    """Base trained hdf5 model"""
    def __init__(self, file_path):
        self.file_path = file_path


class Hdf5Model(FileModel):
    def __init__(self, hdf5_file=None, *args, **kwargs):
        if not hdf5_file:
            hdf5_file = self.get_hdf5_file_path()
        super().__init__(hdf5_file, *args, **kwargs)

    def get_hdf5_file_path(self):
        """Set hdf5 file path
        If the trained hdf5 file exists in a local tmp directory, returns its file path.
        If not exists, download hdf5 file from s3 resource.

        Return: str
            hdf5 file path
        """
        hdf_file_path = os.path.join(settings.base_dir, 'tmp', 'hdf5', settings.hdf5_file_name)
        logger.debug({'action': 'get_hdf5_file_path', 'hdf_file_path': hdf_file_path})
        if not os.path.exists(hdf_file_path):
            logger.info({'action': 'get_hdf5_file_path', 'status': 'start', 'message': 'downlaod hdf5 file from S3'})
            s3_key = os.path.join('hdf5', settings.hdf5_file_name)
            s3_client = S3Object(s3_key,
                                 aws_access_key_id=settings.aws_access_key_id,
                                 aws_secret_access_key=settings.aws_secret_access_key)
            download_dir = os.path.join(settings.base_dir, 'tmp', 'hdf5')
            try:
                s3_client.download_file(download_dir)
            except botocore.exceptions.ClientError as e:
                logger.error({'action': 'get_hdf5_file_path', 'status': 'fail', 'message': e})
                raise
            except Exception as e:
                logger.error({'action': 'get_hdf5_file_path', 'status': 'fail', 'message': e})
                raise
        return hdf_file_path


class CsvModel(FileModel):
    def __init__(self, spot_id: int, csv_file=None, *args, **kwargs):
        """Instantiate CsvModel class

        Parameters
        ----------
        spot_id: int
            sightseeing spot id. Each csv file managed by each spot
        """
        if not spot_id:
            raise NoSpotIdError
        self.spot_id = spot_id
        self.file_name = str(self.spot_id) + '.csv'
        if not csv_file:
            csv_file = self.get_csv_file_path()
        super().__init__(csv_file, *args, **kwargs)
        self.columns_class = CSV_COLUMN_CLASS
        self.columns_vector = CSV_COLUMN_VECTOR
        self.columns = [CSV_COLUMN_CLASS, CSV_COLUMN_VECTOR]
        self.classes = []
        self.vector_images = []

    def get_csv_file_path(self):
        """Set csv file path
        If the file in format csv managed by spot_id exists in a local tmp directory, returns its file path.
        If not exists, download csv file from S3. If also fail to downlaod file, create a new csv file.
        """
        csv_file_path = os.path.join(settings.base_dir, 'tmp', 'csv', self.file_name)
        logger.debug({'action': 'get_csv_file_path', 'csv_file_path': csv_file_path})
        if not os.path.exists(csv_file_path):
            logger.info({'action': 'get_csv_file_path', 'status': 'start', 'message': 'downlaod csv file from S3'})
            s3_key = os.path.join('csv', self.file_name)
            s3_client = S3Object(s3_key,
                                 aws_access_key_id=settings.aws_access_key_id,
                                 aws_secret_access_key=settings.aws_secret_access_key)
            download_dir = os.path.join(settings.base_dir, 'tmp', 'csv')
            try:
                s3_client.download_file(download_dir)
            except botocore.exceptions.ClientError as e:
                logger.error({
                    'action': 'get_csv_file_path',
                    'status': 'fail',
                    'message': e,
                    'csv_file_path': csv_file_path
                })
                pathlib.Path(csv_file_path).touch()
        return csv_file_path
    
    def laod_data(self):
        """Load csv data

        Returns: 
        """
        with open(self.file_path, 'r+') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                images = [float(s) for s in eval(row[self.columns_vector])]
                self.classes.append(row[self.columns_class])
                self.vector_images.append(images)
        return self.classes, self.vector_images

    def save_all_data(self, classes: list, images: list):
        """Save class and converted vector data to csv file

        Parameters
        ----------

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
        logger.info({'action': 'save_all_data', 'status': 'start', 'spot_id': self.spot_id, 'message': 'finish to upload csv to S3'})
        s3_key = os.path.join('csv', self.file_name)
        s3_client = S3Object(s3_key,
                             aws_access_key_id=settings.aws_access_key_id,
                             aws_secret_access_key=settings.aws_secret_access_key)
        s3_client.upload_file(self.file_path)
        logger.info({'action': 'save_all_data', 'status': 'end', 'spot_id': self.spot_id, 'message': 'finish to uplaod csv to S3'})

    def add_new_data(self, classes: list, images: list):
        """Save class and converted vector data to csv file with each exhibit update
        """
        logger.info({'action': 'save_new_data', 'status': 'start', 'spot_id': self.spot_id, 'message': 'start to write csv file'})
        with open(self.file_path, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.columns)
            writer.writeheader()

            for class_, image in zip(classes, images):
                writer.writerow({
                    self.columns_class: class_,
                    self.columns_vector: image
                })
                logger.info({'action': 'save_new_data', 'status': 'writing', 'class': class_, 'message': 'writing class, image vector as csv row'})
        logger.info({'action': 'save_new_data', 'status': 'end', 'spot_id': self.spot_id, 'message': 'finish to write csv file'})
        return None


class KnnModel(FileModel):
    def __init__(self, spot_id: int, knn_file=None, *args, **kwargs):
        if not spot_id:
            raise NoSpotIdError
        self.spot_id = spot_id
        self.file_name = str(self.spot_id) + '.pkl'
        if not knn_file:
            knn_file = self.get_knn_file_path()
        super().__init__(knn_file, *args, **kwargs)

    def get_knn_file_path(self):
        knn_file_path = os.path.join(settings.base_dir, 'tmp', 'knn', self.file_name)
        logger.debug({'action': 'get_knn_file_path', 'knn_file_path': knn_file_path})
        if not os.path.exists(knn_file_path):
            logger.info({'action': 'get_knn_file_path', 'status': 'start', 'message': 'downlaod knn file from S3'})
            s3_key = os.path.join('csv', self.file_name)
            s3_client = S3Object(s3_key,
                                 aws_access_key_id=settings.aws_access_key_id,
                                 aws_secret_access_key=settings.aws_secret_access_key)
            download_dir = os.path.join(settings.base_dir, 'tmp', 'knn')
            try:
                s3_client.download_file(download_dir)
            except botocore.exceptions.ClientError as e:
                logger.error({
                    'action': 'get_knn_file_path',
                    'status': 'fail',
                    'message': e,
                    'knn_file_path': knn_file_path
                })
                pathlib.Path(knn_file_path).touch()
        return knn_file_path

    def save_trained_model(self, model_obj):
        logger.debug({'action': 'save_train_data', 'status': 'start', 'knn_file_path': self.file_path})
        with open(self.file_path, 'wb') as pkl_file:
            pickle.dump(model_obj, pkl_file)
        logger.debug({'action': 'save_train_data', 'status': 'end', 'knn_file_path': self.file_path})

    def load_trained_model(self):
        logger.debug({'action': 'load_train_data', 'status': 'start', 'knn_file_path': self.file_path})
        with open(self.file_path, 'rb') as pkl_file:
            neighbor = pickle.load(pkl_file)
        logger.debug({'action': 'load_train_data', 'status': 'end', 'knn_file_path': self.file_path})
        return neighbor

    def inference(self, input_data: list):
        neighbor = self.load_trained_model()
        distances, neighbors = neighbor.kneighbors(input_data)
        distances, neighbors = distances.tolist(), neighbors.tolist()

        csv_file = CsvModel(self.spot_id)
        classes, vector_images = csv_file.laod_data()

        result_list = []
        for distance, neighbor in zip(distances, neighbors):
            for d, n in zip(distance, neighbor):
                result_list.append((int(classes[n]), d))
        result_list.sort(key=lambda x: x[1])
        return result_list


        

