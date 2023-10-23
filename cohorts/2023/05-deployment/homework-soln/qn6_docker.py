from flask import Flask
from flask import request
from flask import jsonify

import pickle

def load(input_file):
    with open(input_file, 'rb') as f_in: 
        return pickle.load(f_in)

dv = load(f'dv.bin')
model = load(f'model2.bin')

app = Flask('bank')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    approved = y_pred >= 0.5

    result = {
        'approval_probability': float(y_pred),
        'loan approved': bool(approved)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

