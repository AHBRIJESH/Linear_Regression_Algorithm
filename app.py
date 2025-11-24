from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

df = pd.read_csv("data.csv")
X = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values

model = LinearRegression()
model.fit(X, y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    x_val = float(data["x"])
    pred = model.predict([[x_val]])[0]
    return jsonify({"prediction": float(pred)})
