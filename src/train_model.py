# src/train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

def train_and_evaluate():
    # データ読み込み
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    # モデル定義
    model = LinearRegression()

    # 学習
    model.fit(X_train, y_train)

    # 予測
    y_pred = model.predict(X_test)

    # 評価
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("✅ モデル評価結果")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")

    # モデル保存
    joblib.dump(model, "src/model.pkl")
    print("💾 学習済みモデルを src/model.pkl に保存しました")

if __name__ == "__main__":
    train_and_evaluate()
