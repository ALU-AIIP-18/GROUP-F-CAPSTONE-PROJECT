import flask
import pandas as pd
from flask import render_template



from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return flask.render_template("index_file_upload.html")

@app.route("/store_file", methods = ["POST"])
def store_file(): #uploads sensor csv file and streams it online

    #storing the file here
    file_obj = flask.request.files["filename"]
    print(file_obj)
    file_obj.save("Propeller_Sensor_Data.csv")
    #return "Upload Successful"
    #return flask.redirect("/")
    #return flask.render_template("index2.html")
    #
    #
    return flask.render_template("index2.html")
@app.route("/csv")
def csv():
    df = pd.read_csv('Propeller_Sensor_Data.csv')

    return df.to_html(header="true", table_id="table")
if __name__ == '__main__':
    app.run(port=5009, debug=True)