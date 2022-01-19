from flask import Flask, request, jsonify
import joblib
import os

# Connexion au modèle: sur PythonAnywhere
#my_dir = os.path.dirname(__file__)
#file_path = os.path.join(my_dir, 'best_model.pkl')
#with open(file_path, 'rb') as f:
#    saved_mdl = joblib.load(f)

# Définition de l'API
app = Flask(__name__)


@app.route("/")
def hello():
    return "Temporary page"

@app.route("/predict", methods=['POST'])
def predict():
    json_ = request.get_json()
    data_json = json_['data']
    data_val = [list(data_json.values())]
    predict_val = saved_mdl.predict(data_val)
    prediction = predict_val.tolist()
    proba_val = saved_mdl.predict_proba(data_val)
    probabilite = proba_val.tolist()

    return jsonify(Prediction=prediction, Probabilite=probabilite)


if __name__ == '__main__':
    # Pour une utilisation en local + mettre en comm la connexion PythonAnywhere
    saved_mdl = joblib.load('best_model.pkl') # Chargement du modèle
    # Si en cours de DEV, mettre debug=True. Si en PROD, ne pas le mettre
    app.run(debug=True)
    #app.run(debug=False)
