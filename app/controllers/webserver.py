from flask import Flask
from flask import jsonify
from flask import request

from app.controllers.inference import InferenceController
from app.controllers.training import TrainController
import settings


app = Flask(__name__)


@app.route('/api/v1/health')
def hello_world():
    return 'hello world'


@app.route('/api/v1/train/spot', methods=['POST'])
def train_spot():
    spot_id = request.args.get('spot_id')
    if not spot_id:
        return jsonify({'error': True, 'message': 'spot_id is no found'}), 400
    train_controller = TrainController(spot_id=spot_id)
    train_controller.save_image_predict()
    train_controller.train_knn_model()
    return jsonify({'error': False, 'message': 'success to train knn model'}), 200


@app.route('/api/v1/train/exhibit', methods=['POST'])
def train_exhibit():
    spot_id = request.args.get('spot_id')
    exhibit_id = request.args.get('exhibit_id')
    if not spot_id or not exhibit_id:
        return jsonify({'error': True, 'message': 'spot_id or exhibit_id is no found'}), 400
    train_controller = TrainController(spot_id=spot_id)
    train_controller.save_image_predict(exhibit_id=exhibit_id)
    train_controller.train_knn_model()
    return jsonify({'error': False, 'message': 'success to train knn model by exhibit_id'}), 200


@app.route('/api/v1/inference/knn', methods=['POST'])
def inference_knn():
    spot_id = request.args.get('spot_id')
    data = request.args.get('data')
    if not spot_id:
        return jsonify({'error': True, 'message': 'spot_id is no found'}), 400
    inference_controller = InferenceController(spot_id=spot_id)
    result = inference_controller.inference_knn_model(data=data)
    return jsonify({'error': False, 'result': result}), 200


def start():
    app.debug = True
    app.run(host='0.0.0.0', port=settings.port)
