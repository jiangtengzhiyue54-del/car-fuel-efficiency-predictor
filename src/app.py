# src/app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from sklearn.metrics import mean_squared_error
import numpy as np

st.title("🚗 車の燃費予測アプリ")

# モデル読み込み
model = joblib.load("src/model.pkl")

# 入力フォーム
cylinders = st.number_input("シリンダー数", min_value=3, max_value=12, value=4)
displacement = st.number_input("排気量 (cu inches)", min_value=50, max_value=500, value=200)
horsepower = st.number_input("馬力 (hp)", min_value=40, max_value=250, value=100)
weight = st.number_input("重量 (lbs)", min_value=1500, max_value=5000, value=2500)
acceleration = st.number_input("加速度 (0-60mph)", min_value=5.0, max_value=25.0, value=15.0)
model_year = st.slider("モデル年式", 70, 82, 76)
origin = st.selectbox("製造国 (origin)", [1, 2, 3], index=0,
                      format_func=lambda x: {1:"USA",2:"Europe",3:"Japan"}[x])

if st.button("燃費を予測"):
    X_new = pd.DataFrame([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]],
                         columns=["cylinders","displacement","horsepower","weight","acceleration","model year", "origin"])
    mpg_pred = model.predict(X_new).item()
    st.success(f"予測燃費: {mpg_pred:.2f} MPG")

        # --- 可視化 ---
    # テストデータ読み込み
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    y_pred = model.predict(X_test)

    # 散布図: 実測値 vs 予測値
    fig1, ax1 = plt.subplots(figsize=(5,5))
    sns.scatterplot(x=y_test.values.flatten(), y=y_pred.flatten(), ax=ax1)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45度線
    ax1.set_xlabel("実測値 MPG")
    ax1.set_ylabel("予測値 MPG")
    ax1.set_title("実測値 vs 予測値")
    st.pyplot(fig1)

    # 誤差ヒストグラム
    errors = y_test.values.flatten() - y_pred.flatten()
    fig2, ax2 = plt.subplots(figsize=(5,4))
    sns.histplot(errors, bins=20, kde=True, ax=ax2)
    ax2.set_xlabel("誤差 (実測 - 予測)")
    ax2.set_title("予測誤差の分布")
    st.pyplot(fig2)

    # RMSE表示
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"テストデータ RMSE: {rmse:.3f}")