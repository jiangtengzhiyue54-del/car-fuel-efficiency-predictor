# src/train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

def train_and_evaluate():
    # データ読み込み
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.reshape(-1, 1)
    y_test  = pd.read_csv("data/processed/y_test.csv").values.reshape(-1, 1)

    # スケーラー定義
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # スケーリング
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled  = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled  = y_scaler.transform(y_test)

    # 線形回帰モデル
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)

    # 予測（スケール空間）
    y_pred_scaled = model.predict(X_test_scaled)
    # 予測を元のスケールに戻す
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    # 評価
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print("✅ モデル評価結果")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")

    # モデルとスケーラーを保存
    joblib.dump(model, "src/model.pkl")
    joblib.dump(x_scaler, "src/x_scaler.pkl")
    joblib.dump(y_scaler, "src/y_scaler.pkl")
    print("💾 モデル・スケーラーを保存しました (src/以下に保存)")

if __name__ == "__main__":
    train_and_evaluate()
