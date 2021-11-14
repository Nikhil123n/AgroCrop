# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------


# Loading crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + 'mumbai'
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None



# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page
@ app.route('/')
@ app.route('/index')
def home():
    title = 'AgroCrop - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page
@ app.route('/Crop-Recommendation')
def crop_recommend():
    title = 'AgroCrop - Crop Recommendation'
    return render_template('CropRecommendation.html', title=title)

# render fertilizer recommendation form page
@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'AgroCrop - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)



# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page
@ app.route('/crop_prediction', methods=['GET', 'POST'])
def crop_prediction():
    title = 'AgroCrop - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop-result.html', prediction=final_prediction, title=title, pred='img/crop/'+final_prediction+'.jpg',)

    else:
        return render_template('try_again.html', title=title)


# render fertilizer recommendation result page
@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'AgroCrop - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
