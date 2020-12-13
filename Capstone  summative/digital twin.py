#!/usr/bin/env python
# coding: utf-8


from flask import Flask, request
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import datetime
import numpy as np
import pickle
import json
#read streaming files from uploaded csv on server
dd = pd.read_html("http://127.0.0.1:5009/csv")[0]


app = Flask(__name__)

@app.route('/weather',methods=["POST"])
#def weather(location,num):
def weather():  #function that requests wind speed information from weather api
    location = request.form["location"]
    num = request.form["days"]

    city = location
    days = int(num)
    coord_response = requests.get(
        'http://api.openweathermap.org/data/2.5/weather?q=' + city + '&appid=43e49f2fb4d17b806dfff389f21f4d27')
    coordinates = coord_response.json()
    longitude = str(coordinates['coord']['lon'])
    latitude = str(coordinates['coord']['lat'])

    response = requests.get('https://api.openweathermap.org/data/2.5/onecall?lat='+ latitude + '&lon='+ longitude + '&units=metric&exclude=minutely,hourly&appid=43e49f2fb4d17b806dfff389f21f4d27')

    wind_dict=response.json()
    date = []
    speed = []

    for i in range(days):
        date.append(wind_dict['daily'][i]['dt'])
        speed.append(wind_dict['daily'][i]['wind_speed'])

    wind_forecast_data = {'Date': date, 'wind speed': speed}

    wind_df = pd.DataFrame.from_dict(wind_forecast_data)  # convert to dataframe
    for i in wind_df['Date']:
        s = i
        fmt = "%Y-%m-%d"
        t = datetime.datetime.fromtimestamp(float(s))
        t = t.strftime(fmt)
        wind_df['Date'].replace(i, t, inplace=True)
    wind_df['wind speed'] = wind_df['wind speed'] * 3.6
    wind_df = wind_df.to_dict()
    return wind_df







@app.route('/prediction',methods=["POST"])
#dd=pd.read_html("http://127.0.0.1:5009/csv")[0]
#def prediction(hh,ww,df):
def prediction():   #function that gets the height and weight information and predicts efficiency for current day
    hh = request.form["height"]
    ww = request.form["weight"]
    df = request.form["df"]

    df = json.loads(df)
    print(df)
    df = pd.DataFrame(df)
    height = int(hh)
    weight = float(ww)
    cond_df = dd.copy()

    cond_df = cond_df[cond_df['max_height_10sec_feet'] >= height]
    new_df = cond_df[cond_df['max_payload_kg'] >= weight]

    # new_df
    print(new_df)
    new_df = new_df.drop(['Unnamed: 0', 'efficiency', 'wind_speed_km/h'], 1)

    new_df['windspeed'] = df['wind speed'][0]

    lb_make = LabelEncoder()
    new_df.model = lb_make.fit_transform(new_df.model)

    with open('prop_model.pkl', 'rb') as f:
        model = pickle.load(f)
    X = new_df
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)  # standardizing the data for prediction
    new_df['Predicted Efficiency'] = model.predict(X)

    # new_df
    new_df.model = lb_make.inverse_transform(new_df.model)

    k = new_df.groupby(['model']).max()

    k = k.reset_index()

    mm = []
    eff = []

    if len(k.prop_diam.unique()) < 5:
        for i in range(len(k.prop_diam.unique())):
            j = k[k['prop_diam'] == k.prop_diam.unique()[i]].sort_values('Predicted Efficiency', ascending=False)
            t = j.model[0:3].values
            for w in t:
                mm.append(w)
            m = j['Predicted Efficiency'][0:3].values
            t = j.model[0:3].values
            for a in m:
                eff.append(a)
    else:
        for i in range(len(k.prop_diam.unique())):
            j = k[k['prop_diam'] == k.prop_diam.unique()[i]].sort_values('Predicted Efficiency', ascending=False)
            b = j.model.tolist()
            mm.append(b[0])
            v = j['Predicted Efficiency'].tolist()
            eff.append(v[0])

    predicted_data = { 'model': mm, 'efficiency': eff}

    #p_df = pd.DataFrame.from_dict(predicted_data)  # convert to dataframe
    p_df = predicted_data
    return p_df

@app.route('/trend', methods=["POST"])
#def future_forecast(hh, ww, df,predct):
def trend():  #this function predicts effiency for n number of days
    hh = request.form["height"]
    ww = request.form["weight"]
    df = request.form["wind_df"]
    predct = request.form["forecast"]

    df = json.loads(df)
    predct = json.loads(predct)
    print(df)
    print(predct)

    df = pd.DataFrame(df)
    predct = pd.DataFrame(predct)
    height = int(hh)
    weight = float(ww)
    cond_df = dd.copy()


    cond_df = cond_df[cond_df['max_height_10sec_feet'] >= height]
    new_df = cond_df[cond_df['max_payload_kg'] >= weight]

    def forecast(df, i,dat,wind_df):
        dat=dat.model.to_list()
        pred = df[np.isin(df, dat).any(axis=1)]
        pred.reset_index(inplace=True)
        pred = pred.drop(['Unnamed: 0', 'index', 'efficiency', 'wind_speed_km/h'], 1)
        pred['windspeed'] = wind_df['wind speed'][i]
        lb_make = LabelEncoder()
        pred.model = lb_make.fit_transform(pred.model)
        with open('prop_model.pkl', 'rb') as f:
            model = pickle.load(f)
        X = pred
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)  # standardizing the data for prediction
        pred['Predicted Efficiency'] = model.predict(X)
        pred.model = lb_make.inverse_transform(pred.model)
        kk = pred.groupby(['model']).max()
        k = kk.reset_index()
        mm = []
        eff = []
        if len(k.prop_diam.unique()) < 5:
            for i in range(len(k.prop_diam.unique())):
                j = k[k['prop_diam'] == k.prop_diam.unique()[i]].sort_values('Predicted Efficiency', ascending=False)
                t = j.model[0:3].values
                for w in t:
                    mm.append(w)
                m = j['Predicted Efficiency'][0:3].values
                t = j.model[0:3].values
                for a in m:
                    eff.append(a)
        else:
            for i in range(len(k.prop_diam.unique())):
                j = k[k['prop_diam'] == k.prop_diam.unique()[i]].sort_values('Predicted Efficiency', ascending=False)
                b = j.model.tolist()
                mm.append(b[0])
                v = j['Predicted Efficiency'].tolist()
                eff.append(v[0])
        return eff

    check = {}

    for i in range(1, len(df['wind speed'])):
        check[i] = forecast(new_df, i,predct,df)

    dff = pd.DataFrame.from_dict(check)  # convert to dataframe

    dff['model'] = predct.model.values
    dff[0] = predct.efficiency.values
    dff.set_index('model',inplace=True)
    dff = dff.reindex(sorted(dff.columns), axis=1)
    dff = dff.transpose()
    dff['Date'] = df.Date.values
    dff.set_index('Date', inplace=True)
    cols = dff.columns
    dff[cols] = dff[cols].apply(pd.to_numeric, errors='coerce')
    #dff.index = pd.to_datetime(dff.index)

    print(dff)
    dff=dff.to_dict()
    print(dff)
    return dff


if __name__ == '__main__':
    app.run(port=8000, debug=True)







