from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load mô hình đã train
model = joblib.load("ann_co2_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        engine = float(request.form["engine"])
        cylinders = int(request.form["cylinders"])
        fuel = request.form["fuel"]
        city = float(request.form["city"])
        highway = float(request.form["highway"])
        combined = float(request.form["combined"])

        data = pd.DataFrame({
            "Engine_Size_L": [engine],
            "Cylinders": [cylinders],
            "Fuel_Type": [fuel],
            "Fuel_Consumption_City_L100km": [city],
            "Fuel_Consumption_Hwy_L100km": [highway],
            "Fuel_Consumption_Comb_L100km": [combined]
        })

        prediction = model.predict(data)[0]

        return render_template(
            "index.html",
            prediction=round(prediction, 2)
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)