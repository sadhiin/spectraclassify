import os
import pathlib
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from flask_cors import CORS, cross_origin

from spectraclassify import logger


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])
def home():
    logger.info("Home page")
    return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
