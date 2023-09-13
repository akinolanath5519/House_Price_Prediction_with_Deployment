from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained machine learning model here
model = joblib.load('models/house.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the HTML form
        features = {
            'bedrooms': float(request.form['bedrooms']),
            'bathrooms': float(request.form['bathrooms']),
            'sqft_living': float(request.form['sqft_living']),
            'sqft_lot': float(request.form['sqft_lot']),
            'floors': float(request.form['floors']),
            'waterfront': int(request.form['waterfront']),
            'view': int(request.form['view']),
            'condition': int(request.form['condition']),
            'sqft_above': float(request.form['sqft_above']),
            'sqft_basement': float(request.form['sqft_basement']),
            'yr_built': int(request.form['yr_built']),
            'yr_renovated': int(request.form['yr_renovated'])
        }

        # Create a DataFrame from the input data
        input_data = pd.DataFrame([features])

        # Perform prediction using your loaded model
        prediction = model.predict(input_data)

        # Display the result
        return render_template('result.html', prediction="Predicted Price: ${:.2f}".format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
