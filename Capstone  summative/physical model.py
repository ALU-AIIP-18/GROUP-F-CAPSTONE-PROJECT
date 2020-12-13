import flask
import pandas as pd
from flask import render_template
#import windapi
#import current_forecast
import plotly.graph_objs as go
from flask import Flask
import io
import base64
import matplotlib.pyplot as plt
import datetime
from flask import Flask, session
import os
import datetime
import visuals
import requests
import json

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached' #decalared to allow transfer of variables from one app to another
app.config['SECRET_KEY'] = 'super secret key'


@app.route('/')
def index():
    return flask.render_template("p_index1.html")

@app.route('/analyze', methods = ["POST"])
def analyze():  #this function takes in inout parametes and runs it through our digital maodel to return predicted effiency charts

    location = flask.request.form['location']

    days = flask.request.form['days']
    height = flask.request.form['height']
    weight = flask.request.form['weight']

    weather = {}
    weather['location'] = location
    weather['days'] = days
    response = requests.post("http://127.0.0.1:8000/weather", data=weather)
    report = response.json()
    wind_df = pd.DataFrame(report)

    session['list1'] = location
    session['list2'] = days
    session['list3'] = height
    session['list4'] = weight

    wind_dff=wind_df.copy()
    wind_df = wind_df.to_dict()
    print(wind_df)
    wind_df = json.dumps(wind_df)
    print(wind_df)
    val={}
    val['height']=height
    val['weight'] = weight
    val['df']=wind_df
    response = requests.post("http://127.0.0.1:8000/prediction", data=val)

    #forecast1 = current_forecast.prediction(height,weight,wind_df)
    forecast = response.json()
    forecast1 = pd.DataFrame(forecast)
    print(forecast1)
    print(forecast)
    #forecast1 = forecast1.to_dict()
    forecast = json.dumps(forecast)
    print(wind_df)
    print(forecast)
    val2 = {}
    val2['height'] = height
    val2['weight'] = weight
    val2['wind_df'] = wind_df
    val2['forecast'] = forecast

    print(val2)

    #forecast = current_forecast.future_forecast(height, weight, wind_df,forecast1)
    response2 = requests.post("http://127.0.0.1:8000/trend", data=val2)
    forecast2 = response2.json()
    forecast2 = pd.DataFrame(forecast2)
    forecast2.index = pd.to_datetime(forecast2.index)
    print(forecast2)
    forecast2.info()

    print(forecast2)
    df = forecast1.copy()
    df2 = forecast2.copy()
    #df.plot(x='model', kind='bar', figsize=(10, 5))
    visuals.barchart(df)
    plt.title('This is ' + str(location) + "'s Propeller Efficiency Predictions Based On Today's Wind Data")
    plt.savefig('static/images/plot' + str(location)+str(days)+str(height)+str(weight) + '.png')
    plt.clf()

    #df2.plot(figsize=(15,8),marker='x')
    visuals.lineplot(df2,wind_dff)
    plt.title('This is a ' + str(days) + ' day Propeller Efficiency forecast for ' + str(location))
    plt.savefig('static/images/plot2' + str(location)+str(days)+str(height)+str(weight) +  '.png')
    return render_template('plot2.html',url='static/images/plot'+ str(location)+str(days)+str(height)+str(weight)+ '.png')
    #return df.to_html(header="true", table_id="table")

@app.route('/forecast', methods=["GET"])
def forecast():
    location = session.get('list1')
    days = session.get('list2')
    height = session.get('list3')
    weight = session.get('list4')
    return render_template('plot3.html',url='static/images/plot2'+ str(location)+str(days)+str(height)+str(weight) +  '.png')
    #return forecast.to_html(header="true", table_id="table")




if __name__ == '__main__':
    app.run(port=5006, debug=True)

