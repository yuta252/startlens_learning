import logging

import numpy as np

from app.model.knn import KnnModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/app/controller/inference.log')
logger.addHandler(handler)


class InferenceController(object):

    def __init__(self, spot_id: int):
        self.spot_id = spot_id

    def inference_knn_model(self, data: list) -> list:
        # preprocess vector data
        data = np.array(data)
        data = data.reshape(1, -1)
        knn_model = KnnModel(spot_id=self.spot_id)
        result_list = knn_model.inference(input_data=data)
        # TODO: 推論結果の距離により予測結果から除外する処理を検討
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
