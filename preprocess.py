import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ==========================
# 1. Đọc dữ liệu
# ==========================
df = pd.read_csv("co2_car_emissions.csv")

# ==========================
# 2. Tách dữ liệu
# ==========================
X = df.drop("CO2_Emissions_g_km", axis=1)
y = df["CO2_Emissions_g_km"]

# ==========================
# 3. Các cột
# ==========================
categorical_features = ["Fuel_Type"]

numerical_features = [
    "Engine_Size_L",
    "Cylinders",
    "Fuel_Consumption_City_L100km",
    "Fuel_Consumption_Hwy_L100km",
    "Fuel_Consumption_Comb_L100km"
]

# ==========================
# 4. Tiền xử lý
# ==========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ==========================
# 5. Chia Train/Test
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# 6. Tạo mô hình ANN
# ==========================
model = Pipeline([
    ("preprocessor", preprocessor),
    ("ann", MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    ))
])

# ==========================
# 7. Huấn luyện
# ==========================
model.fit(X_train, y_train)

# ==========================
# 8. Dự đoán
# ==========================
y_pred = model.predict(X_test)

# ==========================
# 9. Đánh giá
# ==========================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ==========================
# 10. Lưu mô hình
# ==========================
joblib.dump(model, "ann_co2_model.pkl")

print("\nĐã lưu mô hình: ann_co2_model.pkl")