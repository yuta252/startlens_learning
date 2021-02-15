import os
import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

from app.model.knn import CsvModel, KnnModel
from fetch.resource import S3Resource
import settings
from train.triplet_loss import TripletLoss


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/app/controller/training.log')
logger.addHandler(handler)


PATH_CSV_DIR = os.path.join(settings.base_dir, 'tmp', 'csv')
PATH_KNN_DIR = os.path.join(settings.base_dir, 'tmp', 'knn')


class TrainController(object):

    def __init__(self, spot_id: int):
        self.resource = S3Resource(aws_access_key_id=settings.aws_access_key_id,
                                   aws_secret_access_key=settings.aws_secret_access_key)
        self.spot_id = spot_id

    def train_triplet_model(self):
        file_paths = self.resource.get_filtered_by_prefix()
        triplet_model = TripletLoss()
        triplet_model.train(file_paths)

    def save_image_predict(self):
        prefix = os.path.join(settings.prefix_key, str(self.spot_id)) + '/'
        file_paths = self.resource.get_filtered_by_prefix(prefix=prefix)
        triplet_model = TripletLoss()
        triplet_model.predict(file_paths, self.spot_id, is_exhibit=False)

    def save_image_predict_by_exhibit(self, exhibit_id: int):
        prefix = os.path.join(settings.prefix_key, str(self.spot_id), str(exhibit_id)) + '/'
        file_paths = self.resource.get_filtered_by_prefix(prefix=prefix)
        triplet_model = TripletLoss()
        triplet_model.predict(file_paths, self.spot_id, is_exhibit=True)

    def train_knn_model(self):
        # load training data from csv file
        csv_model = CsvModel(spot_id=self.spot_id, is_write=False, csv_dir=PATH_CSV_DIR)
        classes, vector_images = csv_model.load_data()
        # train
        vector_images = np.array(vector_images)
        neighbor = NearestNeighbors(n_neighbors=4)
        neighbor.fit(vector_images)
        # save binary data as pickle
        knn_model = KnnModel(spot_id=self.spot_id, is_write=True, knn_dir=PATH_KNN_DIR)
        knn_model.save_trained_model(neighbor)
