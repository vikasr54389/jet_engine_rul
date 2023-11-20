
import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle
ridge_model=pickle.load(open('models/ridge (1).pkl','rb'))
standard_scaler=pickle.load(open('models/scaler (1).pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        unit_number=float(request.form.get('unit_number'))
        time = float(request.form.get('time'))
        operational_setting_1 = float(request.form.get('operational_setting_1'))
        operational_setting_2 = float(request.form.get('operational_setting_2'))
        sensor_measurement_3 = float(request.form.get('sensor_measurement_3'))
        sensor_measurement_4 = float(request.form.get('sensor_measurement_4'))
        sensor_measurement_6 = float(request.form.get('sensor_measurement_6'))
        sensor_measurement_7 = float(request.form.get('sensor_measurement_7'))
        sensor_measurement_8 = float(request.form.get('sensor_measurement_8'))
        sensor_measurement_9 = float(request.form.get('sensor_measurement_9'))
        sensor_measurement_11 = float(request.form.get('sensor_measurement_11'))
        sensor_measurement_12 = float(request.form.get('sensor_measurement_12'))
        sensor_measurement_13 = float(request.form.get('sensor_measurement_13'))
        sensor_measurement_15 = float(request.form.get('sensor_measurement_15'))
        sensor_measurement_17 = float(request.form.get('sensor_measurement_17'))
        sensor_measurement_20 = float(request.form.get('sensor_measurement_20'))
        sensor_measurement_21 = float(request.form.get('sensor_measurement_21'))

        new_data_scaled=standard_scaler.transform([[unit_number,time,operational_setting_1,operational_setting_2,sensor_measurement_3,sensor_measurement_4,sensor_measurement_6,sensor_measurement_7,sensor_measurement_8,sensor_measurement_9,sensor_measurement_11,sensor_measurement_12,sensor_measurement_13,sensor_measurement_15,sensor_measurement_17,sensor_measurement_20,sensor_measurement_21]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")