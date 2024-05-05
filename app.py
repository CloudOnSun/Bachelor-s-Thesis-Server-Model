from flask import Flask, request, jsonify
from pso import PSOModel

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    pso_model = PSOModel(data['rfs'])
    cost, position = pso_model.predict()
    result = {"cost": cost, "position": list(position)}
    return jsonify(result)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
