import constants
from fetch.resource import S3Resource
import settings
from train.triplet_loss import TripletLoss


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
