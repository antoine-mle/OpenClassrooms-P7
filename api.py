import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Load pretrained model
transformer = pickle.load(open("transformer.pkl", "rb"))
classifier = pickle.load(open("classifier.pkl", "rb"))

# Init Flask app
app = Flask(__name__)

@app.route('/git_update', methods=['POST'])
def webhook():
    if request.method == 'POST':
        repo = git.Repo('./OpenClassrooms-P7')
        origin = repo.remotes.origin
        repo.create_head('main',origin.refs.main).set_tracking_branch(origin.refs.main).checkout()
        origin.pull()
        return '', 200
    else:
        return '', 400

@app.route("/")
def hello():
    return "Machine learning API"

@app.route("/predict", methods=['POST','GET'])
def predict():
    if request.method == 'GET':
        return "Prediction page"

    if request.method == 'POST':
        client_input = request.get_json()
        # Convert dictionary to pandas dataframe
        client_input = pd.DataFrame(client_input)
        # Transforming features
        client_input = transformer.transform(client_input)
        # Making predictions
        pred = classifier.predict(client_input)[0]
        proba = classifier.predict_proba(client_input)[0][pred]
        return jsonify(prediction=int(pred), probability=round(100 * proba, 1))

if __name__ == '__main__':
    # Si en cours de DEV, mettre debug=True. Si en PROD, ne pas le mettre
    app.run(debug=True)
    #app.run(debug=False)
