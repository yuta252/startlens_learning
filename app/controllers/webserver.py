from flask import Flask
from flask import jsonify
from flask import request

from app.controllers.inference import InferenceController
from app.controllers.training import TrainController
import settings


app = Flask(__name__)


@app.route('/api/v1/health')
def hello_world():
    return jsonify({'error': False, 'message': 'okay'})


@app.route('/api/v1/train/spot', methods=['POST'])
def train_spot():
    spot_id = request.json['spotId']
    print(spot_id)
    if spot_id is None:
        return jsonify({'error': True, 'message': 'spot_id is not found'}), 400
    train_controller = TrainController(spot_id=spot_id)
    train_controller.save_image_predict()
    train_controller.train_knn_model()
    return jsonify({'error': False, 'message': 'success to train knn model'}), 200


@app.route('/api/v1/train/exhibit', methods=['POST'])
def train_exhibit():
    spot_id = request.json['spotId']
    exhibit_id = request.json['exhibitId']
    if not spot_id or not exhibit_id:
        return jsonify({'error': True, 'message': 'spot_id or exhibit_id is no found'}), 400
    train_controller = TrainController(spot_id=spot_id)
    train_controller.save_image_predict_by_exhibit(exhibit_id=exhibit_id)
    train_controller.train_knn_model()
    return jsonify({'error': False, 'message': 'success to train knn model by exhibit_id'}), 200


@app.route('/api/v1/inference/knn', methods=['POST'])
def inference_knn():
    spot_id = request.json['spotId']
    data = request.json['data']
    if not spot_id:
        return jsonify({'error': True, 'message': 'spot_id is no found'}), 400
    inference_controller = InferenceController(spot_id=spot_id)
    result = inference_controller.inference_knn_model(data=data)
    return jsonify({'error': False, 'result': result}), 200


def start():
    app.debug = True
    app.run(host='0.0.0.0', port=settings.port)
