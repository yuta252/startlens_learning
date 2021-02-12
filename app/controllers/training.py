import os

import numpy as np
from sklearn.neighbors import NearestNeighbors

from app.model.knn import CsvModel, KnnModel
from fetch.resource import S3Resource
import settings
from train.input_generator import GenerateSample
from train.triplet_loss import TripletLoss
from utils.utils import get_class_label_from_path


class TrainImage(object):

    def __init__(self):
        self.resource = S3Resource(aws_access_key_id=settings.aws_access_key_id,
                                   aws_secret_access_key=settings.aws_secret_access_key)

    def train_model(self):
        file_paths = self.resource.get_filtered_by_prefix()
        # s3_objects = [S3Object(file_path) for file_path in file_paths]
        # file_class_mapping = {file_path: get_class_label_from_path(file_path) for file_path in file_paths}
        # samples = GenerateSample(file_class_mapping)
        # g = samples.generate()
        # print(next(g))

        triplet_model = TripletLoss()
        print('start')
        triplet_model.train(file_paths)
        # print(constants.EMBEDDING_DIM)

    def save_image_predict(self, spot_id: int):
        prefix = os.path.join(settings.prefix_key, str(spot_id))
        file_paths = self.resource.get_filtered_by_prefix(prefix=prefix)
        triplet_model = TripletLoss()
        triplet_model.predict(file_paths, spot_id)

    def train_knn_model(self, spot_id: int):
        # load training data from csv file
        csv_model = CsvModel(spot_id=spot_id)
        classes, vector_images = csv_model.laod_data()
        # train
        vector_images = np.array(vector_images)
        neighbor = NearestNeighbors(n_neighbors=4)
        neighbor.fit(vector_images)
        # save binary data as pickle
        knn_model = KnnModel(spot_id=spot_id)
        knn_model.save_trained_model(neighbor)
