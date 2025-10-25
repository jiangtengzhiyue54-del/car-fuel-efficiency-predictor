# src/app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.title("🚗 車の燃費予測アプリ")

# モデル・スケーラー読み込み
model = joblib.load("src/model.pkl")
x_scaler = joblib.load("src/x_scaler.pkl")
y_scaler = joblib.load("src/y_scaler.pkl")

# 入力フォーム
cylinders = st.number_input("シリンダー数", min_value=3, max_value=12, value=4)
displacement = st.number_input("排気量 (cu inches)", min_value=50, max_value=500, value=200)
horsepower = st.number_input("馬力 (hp)", min_value=40, max_value=250, value=100)
weight = st.number_input("重量 (lbs)", min_value=1500, max_value=5000, value=2500)
acceleration = st.number_input("加速度 (0-60mph)", min_value=5.0, max_value=25.0, value=15.0)
model_year = st.slider("モデル年式", 70, 82, 76)
origin = st.selectbox("製造国 (origin)", [1, 2, 3], index=0,
                      format_func=lambda x: {1:"USA", 2:"Europe", 3:"Japan"}[x])

if st.button("燃費を予測"):
    # 入力値をDataFrameに
    X_new = pd.DataFrame([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]],
                         columns=["cylinders","displacement","horsepower","weight","acceleration","model year","origin"])
    
    # スケーリング（入力を学習時のスケールに変換）
    X_new_scaled = x_scaler.transform(X_new)

    # 予測（スケール空間 → 逆変換してMPGに戻す）
    mpg_pred_scaled = model.predict(X_new_scaled)
    mpg_pred = y_scaler.inverse_transform(mpg_pred_scaled.reshape(-1, 1)).item()

    st.success(f"予測燃費: {mpg_pred:.2f} MPG")

    # --- テストデータの評価可視化 ---
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.reshape(-1, 1)

    # スケーリング
    X_test_scaled = x_scaler.transform(X_test)

    # 予測（スケール空間 → 逆変換）
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    # 散布図: 実測値 vs 予測値
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    sns.scatterplot(x=y_test.flatten(), y=y_pred.flatten(), ax=ax1)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_xlabel("実測値 MPG")
    ax1.set_ylabel("予測値 MPG")
    ax1.set_title("実測値 vs 予測値")
    st.pyplot(fig1)

    # 誤差ヒストグラム
    errors = y_test.flatten() - y_pred.flatten()
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.histplot(errors, bins=20, kde=True, ax=ax2)
    ax2.set_xlabel("誤差 (実測 - 予測)")
    ax2.set_title("予測誤差の分布")
    st.pyplot(fig2)

    # RMSE と R²スコア表示
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.write(f"テストデータ RMSE: {rmse:.3f}")
    st.write(f"テストデータ R²スコア: {r2:.3f}")
