import joblib
import pandas as pd

model = joblib.load("ann_co2_model.pkl")

new_car = pd.DataFrame({
    "Engine_Size_L":[2.0],
    "Cylinders":[4],
    "Fuel_Type":["Regular Gasoline"],
    "Fuel_Consumption_City_L100km":[8.5],
    "Fuel_Consumption_Hwy_L100km":[6.5],
    "Fuel_Consumption_Comb_L100km":[7.6]
})

prediction = model.predict(new_car)

print("CO2 dự đoán:", round(prediction[0],2), "g/km")