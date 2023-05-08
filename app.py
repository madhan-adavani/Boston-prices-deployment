from flask import Flask, request, jsonify, render_template,app
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
with open('LinModel.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define the route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    input_features = np.array([float(x) for x in request.form.values()])

    #print(input_features)

    #scale values
    input_features = scaler.transform(input_features.reshape(1,-1))

    #print(input_features)

    # Make a prediction using the loaded model

    prediction = model.predict(input_features)[0]

    # Render the prediction page with the result
    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)