import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2000

# 経過年数: 1〜5年の整数（混在）
T = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.2, 0.25, 0.25, 0.15, 0.15])

# 潜在スコア（低いほどリスク高）
latent = rng.normal(0, 1, size=n)

# T年累積PD（Tが大きいほど累積PDが高い傾向）
# 真の年率PD ~ logistic(latent)
true_annual_pd = 1 / (1 + np.exp(latent + 1.5))
# T年累積PD = 1 - (1 - annual_pd)^T
pd_cumulative = 1 - (1 - true_annual_pd) ** T
# ノイズ付加
pd_cumulative = np.clip(pd_cumulative + rng.normal(0, 0.02, n), 1e-4, 0.9999)

# 実績デフォルト（真の年率PDに基づく確率でサンプリング）
default_flag = rng.binomial(1, true_annual_pd)

# 追加属性列（現実的な与信データを模倣）
age = rng.integers(22, 70, size=n)
loan_amount = rng.integers(500_000, 10_000_000, size=n, endpoint=True)
income = rng.integers(2_000_000, 15_000_000, size=n, endpoint=True)
employment_type = rng.choice(["正社員", "契約社員", "自営業", "パート"], size=n, p=[0.55, 0.2, 0.15, 0.1])
industry = rng.choice(["製造業", "IT", "金融", "小売", "建設", "医療"], size=n)

df = pd.DataFrame({
    "customer_id": [f"C{str(i+1).zfill(5)}" for i in range(n)],
    "age": age,
    "employment_type": employment_type,
    "industry": industry,
    "loan_amount": loan_amount,
    "annual_income": income,
    "elapsed_years": T,
    "pd_score": np.round(pd_cumulative, 6),
    "actual_default": default_flag,
})

df.to_csv("sample_data.csv", index=False)
print(f"保存完了: sample_data.csv ({len(df):,}行)")
print(df.head())
print(f"\nデフォルト率: {df['actual_default'].mean():.2%}")
print(f"PD中央値: {df['pd_score'].median():.4f}")
print(f"経過年数分布:\n{df['elapsed_years'].value_counts().sort_index()}")
