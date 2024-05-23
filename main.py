import pickle
from processor import DataPreprocessor, ChurnProcessor
from flask import Flask, render_template, request, url_for, jsonify, send_file
from flask_bootstrap import Bootstrap5

# ------------------------ Initializing  ---------------------------

app = Flask(__name__)
bootstrap = Bootstrap5(app)
dp = DataPreprocessor()
cp = ChurnProcessor()

# ------------------------ Global Variables ---------------------------

CSBIN_EDGE = [300, 580, 670, 740, 800, 850]
CSLABEL = ['Bad', 'Fair', 'Good', 'Very Good', 'Excellent']
AGEBIN_EDGE = [18, 30, 40, 50, 60, 70, 80, 90, 92]
AGELABEL = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-92']

# ------------------------ inbuit methods ------------------------------

def preprocessing_data(files):
    df = dp.get_data(files)
    geo = dp.one_hotencoding(df, 'Geography')
    gender = dp.one_hotencoding(geo, 'Gender')
    cs = dp.bin_dataset(gender, 'CreditScore', CSBIN_EDGE, CSLABEL)
    ag = dp.bin_dataset(cs, 'Age', AGEBIN_EDGE, AGELABEL)
    lencoded = dp.label_encoder(ag)
    scaled_data = dp.scaler(lencoded)
    return scaled_data

def processing_data(dataset):
    churn = cp.CLASSIFIER.predict(dataset.iloc[:, 1:])
    churn_probabilities = cp.CLASSIFIER.predict_proba(dataset.iloc[:, 1:])
    working_file = cp.feature_creator(dataset, churn, churn_probabilities)
    trial = cp.customer_result(working_file)
    return trial

# ------------------------API building------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        files = request.files['file']

        if files.filename == '':
            return jsonify({'error': 'No selected file'}), 400


        if files.filename.endswith('.csv'):
            data = preprocessing_data(files)
            dataset = processing_data(data)

            with open('Churn Result.txt', 'w') as file:
                for entry in dataset:
                    file.write(entry + '\n')
            return send_file('Churn Result.txt', as_attachment=True)
        else:
            return jsonify({'error': 'Only CSV files are supported'}), 400

    except Exception as e:
        return jsonify({
            'Error uploading file': f'{e}',
            'error': 'An error occurred during upload'
            }), 500
    return render_template('index.html')


@app.route('/help')
def help():
    return render_template('help.html')


if __name__ == '__main__':
    app.run(debug=True)
