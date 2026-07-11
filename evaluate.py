import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("dataset/co2_car_emissions.csv")

# Chia dữ liệu
X = df.drop("CO2_Emissions_g_km", axis=1)
y = df["CO2_Emissions_g_km"]

categorical_features = ["Fuel_Type"]
numerical_features = [
    "Engine_Size_L",
    "Cylinders",
    "Fuel_Consumption_City_L100km",
    "Fuel_Consumption_Hwy_L100km",
    "Fuel_Consumption_Comb_L100km"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=1000,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("========== ĐÁNH GIÁ MÔ HÌNH ==========")

print("MAE  :", mean_absolute_error(y_test, y_pred))
print("RMSE :", mean_squared_error(y_test, y_pred) ** 0.5)
print("R²   :", r2_score(y_test, y_pred))

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.xlabel("Giá trị thực")
plt.ylabel("Giá trị dự đoán")
plt.title("So sánh CO2 thực và CO2 dự đoán")

plt.show()