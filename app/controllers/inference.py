import os
import logging

import numpy as np

from app.model.knn import KnnModel
import settings


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/app/controller/inference.log')
logger.addHandler(handler)


PATH_KNN_DIR = os.path.join(settings.base_dir, 'tmp', 'knn')


class InferenceController(object):

    def __init__(self, spot_id: int):
        self.spot_id = spot_id

    def inference_knn_model(self, data: list) -> list:
        # preprocess vector data
        data = np.array(data)
        logger.info({'action': 'inference_knn_model', 'data': data})
        data = data.reshape(1, -1)
        logger.info({'action': 'inference_knn_model', 'data': data})
        knn_model = KnnModel(spot_id=self.spot_id, is_write=False, knn_dir=PATH_KNN_DIR)
        result_list = knn_model.inference(input_data=data)
        # TODO: To be considered implementing function that exclude the prediction results by the point-to-point distance
        result_classes = []
        for result in result_list:
            result_classes.append(result[0])
        result_classes = list(set(result_classes))
        logger.info({
            'action': 'inference_knn_model',
            'status': 'end',
            'spot_id': self.spot_id,
            'result_classes': result_classes
        })
        return result_classes
