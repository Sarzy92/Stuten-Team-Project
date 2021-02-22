import joblib 
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import catboost 


#load model 
app = Flask(__name__)
api = Api(app)

house_model = joblib.load("house_model.sav")

class RunPred(Resource):
    @staticmethod
    def post():
        data = request.form.to_dict()
        type_ = data["house-type"]
        loc = data["locations"]
        bed = data["bedroom"]
        bath = data["bathroom"]
        toilet = data["toilet"]

        pred = house_model.predict([type_, loc, bed, bath, toilet])

        pred = int(pred)

        return jsonify({
            "House Price" : pred
        })

api.add_resource(RunPred, "/predict")

if __name__ == '__main__':
    app.run(debug=True)
