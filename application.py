from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

##import ridge regression and standard scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
Standard_Scaler = pickle.load(open('models/scaler.pkl', 'rb'))




@app.route('/')
def index():
    return render_template('index.html')  # ✅ matches your actual file

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        ISI = float(request.form.get('ISI'))
        Classes = int(request.form.get('Classes'))
        Region = int(request.form.get('Region'))

        new_scaled_data=Standard_Scaler.transform([[Temperature, RH, Ws, Rain, FFMC, ISI, Classes, Region]])
        result=ridge_model.predict(new_scaled_data)

        return render_template('home.html',results=result[0])
        
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
