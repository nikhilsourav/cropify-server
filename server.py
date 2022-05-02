# *******************************************************************************************
# ********************************** Backend specific ***************************************
# *******************************************************************************************
import data.mapped_values as maps
import pickle
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin


# *******************************************************************************************
# ************************************ ML specific ******************************************
# *******************************************************************************************
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# *******************************************************************************************
# ************************************ Algorithms *******************************************
# *******************************************************************************************
from sklearn.ensemble import RandomForestClassifier


# *******************************************************************************************
# ********************************* Create Flask App ****************************************
# *******************************************************************************************
app = Flask(__name__)
CORS(app, support_credentials=True)


# *******************************************************************************************
# *********************************** Pickle Models *****************************************
# *******************************************************************************************
crop_recommendation_model = pickle.load(
    open("./model/crop_recommendation_model", "rb"))
yield_prediction_model = pickle.load(
    open("./model/yield_prediction_model", "rb"))


# *******************************************************************************************
# ************************************ API routes ******************************************
# *******************************************************************************************

#
# Home Route
#
@app.route("/", methods=['GET'])
@cross_origin(supports_credentials=True)
def home():
    return_data = "Welcome to cropify api"
    return Response(json.dumps(return_data),  mimetype='application/json')

#
# Crop Recommendation Route
#
@app.route('/crop-recommendation', methods=['POST'])
@cross_origin(supports_credentials=True)
def crop_recommendation():
    data = request.json

    recommendation = crop_recommendation_model.predict(
        [
            [
                data["N"],
                data["P"],
                data["K"],
                data["temperature"],
                data["humidity"],
                data["ph"],
                data["rainfall"]
            ]
        ])

    return_data = [
        {
            "recommended crop": maps.crop_map[recommendation[0]],
        }
    ]

    return Response(json.dumps(return_data),  mimetype='application/json')

#
# Yield Prediction Route
#
@app.route('/yield-prediction', methods=['POST'])
@cross_origin(supports_credentials=True)
def yield_prediction():
    data = request.json

    prediction = yield_prediction_model.predict(
        [
            [
                data["Area"],
                data["State_Name"],
                data["Season"],
                data["Crop"],
                data["Soil_Type"],
            ]
        ])

    return_data = [
        {
            "predicted yield": prediction[0]
        }
    ]

    return Response(json.dumps(return_data),  mimetype='application/json')


# *******************************************************************************************
# ************************************ Main Section *****************************************
# *******************************************************************************************
if __name__ == "__main__":
    app.run()
