import os
from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
# Load the saved models
md = pickle.load(open("price.pkl", "rb"))
sc = pickle.load(open("scale.pkl", "rb"))

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/value', methods=['POST'])
def predict():
    # Retrieve values from the form in the HTML page
    transaction_date = float(request.form['X1_transaction_date'])
    house_age = float(request.form['X2_house_age'])
    distance_to_mrt = float(request.form['X3_distance_to_the_nearest_MRT_station'])
    convenience_stores = float(request.form['X4_number_of_convenience_stores'])
    latitude = float(request.form['X5_latitude'])
    longitude = float(request.form['X6_longitude'])
    # Create a 2D array for the input data
    data = [[transaction_date, house_age, distance_to_mrt, convenience_stores, latitude,longitude]]

    # Transform the input data using the loaded scaler
    scaled_data = sc.transform(data)

    # Predict using the loaded model
    prediction = md.predict(scaled_data)

    # Render the template with the prediction result
    return render_template('predict.html', y=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
