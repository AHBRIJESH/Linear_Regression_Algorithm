from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # <== Allow requests from frontend

model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    x_val = float(data["x"])
    x_arr = np.array([[x_val]])
    y_pred = model.predict(x_arr)[0]
    return jsonify({"prediction": float(y_pred)})

if __name__ == "__main__":
    app.run(debug=True)
