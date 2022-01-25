import os
import git
import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Init Flask app
app = Flask(__name__)

# Load pretrained model
my_directory = os.path.dirname(__file__)
pickle_transformer_path = os.path.join(my_directory, "transformer.pkl")
with open(pickle_transformer_path, "rb") as p:
    transformer = pickle.load(p)
pickle_classifier_path = os.path.join(my_directory, "classifier.pkl")
with open(pickle_classifier_path, "rb") as p:
    classifier = pickle.load(p)

#transformer = pickle.load(open("transformer.pkl", "rb"))
#classifier = pickle.load(open("classifier.pkl", "rb"))

@app.route('/git_update', methods=['POST'])
def git_update():
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
        # Parse data as JSON
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
