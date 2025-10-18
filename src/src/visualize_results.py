# src/visualize_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# データ読み込み
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")
model = joblib.load("src/model.pkl")

# 予測
y_pred = model.predict(X_test)

# -----------------------------
# 1. 実測値 vs 予測値の散布図
# -----------------------------
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test.values.flatten(), y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45度線
plt.xlabel("実測値 MPG")
plt.ylabel("予測値 MPG")
plt.title("実測値 vs 予測値")
plt.grid(True)
plt.savefig("docs/figures/scatter_pred_vs_actual.png")
plt.show()

# -----------------------------
# 2. 誤差ヒストグラム
# -----------------------------
errors = y_test.values.flatten() - y_pred
plt.figure(figsize=(6,4))
sns.histplot(errors, bins=20, kde=True)
plt.xlabel("誤差 (実測 - 予測)")
plt.title("予測誤差の分布")
plt.grid(True)
plt.savefig("docs/figures/error_histogram.png")
plt.show()

# -----------------------------
# 3. 学習曲線（RMSE）
# -----------------------------
# 学習曲線は複数の学習データサイズでモデルを学習させる必要があります
# 簡易的に全データでRMSEを表示
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ テストデータ RMSE: {rmse:.3f}")

# -----------------------------
# 4. R²スコア
# -----------------------------
r2 = r2_score(y_test, y_pred)
print(f"✅ テストデータ R²スコア: {r2:.3f}")