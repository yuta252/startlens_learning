from flask import Flask
from flask import jsonify
from flask import request

import settings


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'hello world'


@app.route('/train/knn', methods=['POST'])
def train_knn():
    spot_code = request.args.get('spot_code')
    if not spot_code:
        return jsonify({'error': 'No spot_code params'}), 400
    return str(request.values)


@app.route('/inference/knn', methods=['POST'])
def inference_knn():
    return str(request.values)


def start():
    app.debug = True
    app.run(host='0.0.0.0', port=settings.port)
