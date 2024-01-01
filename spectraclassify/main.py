import os
from pathlib import Path
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from flask_cors import CORS, cross_origin

from spectraclassify import logger
from spectraclassify.utility.config_manager import get_Data_conf, get_model_conf
from spectraclassify.utility.utils import save_json


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])
def home():
    logger.info("Home page")

    if request.method == 'POST':
        print('Post request recived')
        # print(request.form)
        config_dict = request.form.to_dict()
        # print(config_dict)
        save_json(path=Path('configs.json'), data=config_dict)
        print('saving config json file')
        print("Data config:",get_Data_conf(),"\n")
        print("Model config:",get_model_conf())
    return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
