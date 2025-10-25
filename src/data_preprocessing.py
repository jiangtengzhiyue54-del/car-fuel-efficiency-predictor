# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 保存フォルダを作成
os.makedirs("data/processed", exist_ok=True)

def load_and_preprocess_data(csv_path="data/raw/auto-mpg.csv"):
    # データ読み込み
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # 列名の空白を除去

    # 不要な列を削除
    if 'car name' in df.columns:
        df = df.drop('car name', axis=1)

    # 欠損値 "?" を NaN に置き換え → 欠損行削除
    df = df.replace('?', pd.NA)
    df = df.dropna()

    # 数値型に変換
    df = df.astype(float)

    # 特徴量と目的変数を分離
    X = df.drop("mpg", axis=1)
    y = df["mpg"]

    # データ分割（8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # スケーリングは train_model.py 側で行うため、ここでは生データを保存
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("✅ 前処理完了。スケーリングなしでデータを保存しました。")

if __name__ == "__main__":
    load_and_preprocess_data()
