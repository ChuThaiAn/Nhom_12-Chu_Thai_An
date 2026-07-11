import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import os

print("Thư mục hiện tại:", os.getcwd())
print("Các file trong thư mục:")
print(os.listdir("dataset"))

# Đọc dữ liệu
df = pd.read_csv("dataset/co2_car_emissions.csv")

print(df.head())
print(df.columns)

# Chia dữ liệu
X = df.drop(columns=["CO2_Emissions_g_km"])
y = df["CO2_Emissions_g_km"]

# Các cột
num_features = [
    "Engine_Size_L",
    "Cylinders",
    "Fuel_Consumption_City_L100km",
    "Fuel_Consumption_Hwy_L100km",
    "Fuel_Consumption_Comb_L100km"
]

cat_features = ["Fuel_Type"]

# Tiền xử lý
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# ANN
model = Pipeline([
    ("preprocessor", preprocessor),
    ("ann", MLPRegressor(
        hidden_layer_sizes=(64,32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    ))
])

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Huấn luyện
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

print("MAE =", mean_absolute_error(y_test, y_pred))
print("R2  =", r2_score(y_test, y_pred))

# Lưu model
joblib.dump(model, "ann_co2_model.pkl")

print("Đã lưu ann_co2_model.pkl")