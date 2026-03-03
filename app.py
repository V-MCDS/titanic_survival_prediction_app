# import necessary libraries
from flask import Flask, render_template, request
import joblib
import numpy as np

# create Flask app

app = Flask(__name__) 

# load the pre-trained model
model = joblib.load('titanic_model.pkl')

# define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])

        features = np.array([[pclass, sex, age, sibsp, parch, fare]])

        prediction = model.predict(features)
        probability = model.predict_proba(features)

        if prediction[0] == 1:
            result = 'The passenger is likely to survive.'
            confidence = float(probability[0][1]) * 100
        else:
            result = 'The passenger is likely to not survive.'
            confidence = float(probability[0][0]) * 100

        return render_template('index.html', result=result, confidence=confidence)
    

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)

    # C:\Users\M1ne\Desktop\flask demo\app.py