from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load("model/wine_cultivar_model.pkl")

# ✅ ADD IT HERE (right after loading the model)
print("Model expects:", model.n_features_in_)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = [
        float(request.form['alcohol']),
        float(request.form['malic_acid']),
        float(request.form['ash']),
        float(request.form['magnesium']),
        float(request.form['color_intensity']),
        float(request.form['proline'])
    ]

    final_features = np.array(data).reshape(1, -1)
    prediction = model.predict(final_features)[0]
    cultivar = f"Cultivar {prediction + 1}"

    return render_template(
        "index.html",
        prediction_text=f"Predicted Wine Cultivar: {cultivar}"
    )

if __name__ == "__main__":
    app.run(debug=True)
