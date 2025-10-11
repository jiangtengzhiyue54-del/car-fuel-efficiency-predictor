# src/data_preprocessing.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path=r"D:\AIプログラミングⅡ\car-fuel-efficiency-predictor\data\raw\auto-mpg.csv"):

    # 出力フォルダ自動生成
    os.makedirs("data/processed", exist_ok=True)

    # データ読み込み
    df = pd.read_csv(csv_path)

    # 列名調整（データセットによっては空白が混ざる）
    df.columns = df.columns.str.strip()

    # 不要列（車名など）を削除
    if 'car name' in df.columns:
        df = df.drop('car name', axis=1)

    # 欠損値処理（? を NaN にして平均値で補完）
    df = df.replace('?', pd.NA)
    df = df.dropna()

    # 数値型に変換
    df = df.astype(float)

    # 特徴量と目的変数を分離
    X = df.drop("mpg", axis=1)
    y = df["mpg"]

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 前処理後データを保存
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("✅ データ前処理完了：data/processed/ に保存しました")

if __name__ == "__main__":
    load_and_preprocess_data()
