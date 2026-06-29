import numpy as np
import pandas as pd

rng = np.random.default_rng(123)
n = 3000

# 経過年数: 0.5〜7年（小数あり、T<1も含む）
T = rng.choice([0.5, 1, 1.5, 2, 3, 5, 7], size=n,
               p=[0.05, 0.15, 0.10, 0.25, 0.20, 0.15, 0.10])

# 信用スコア（低いほどリスク高）
credit_score = rng.integers(300, 850, size=n)

# 真の年率PD（信用スコアに基づく）
true_annual_pd = 1 / (1 + np.exp((credit_score - 550) / 80))

# T年累積PD + ノイズ
pd_cumulative = 1 - (1 - true_annual_pd) ** T
pd_cumulative = np.clip(pd_cumulative + rng.normal(0, 0.015, n), 1e-4, 0.9999)

# 実績デフォルト
is_default = rng.binomial(1, true_annual_pd)

# 追加属性（列名を意図的に日本語・英語混在にして汎用性テスト）
regions = rng.choice(["東京", "大阪", "名古屋", "福岡", "札幌"], size=n,
                     p=[0.35, 0.25, 0.15, 0.15, 0.10])
loan_type = rng.choice(["住宅ローン", "カードローン", "事業資金", "マイカーローン"], size=n,
                       p=[0.30, 0.35, 0.20, 0.15])
loan_balance = rng.integers(100_000, 50_000_000, size=n, endpoint=True)
num_delinquencies = rng.integers(0, 5, size=n)
debt_ratio = np.round(rng.uniform(0.05, 0.95, n), 3)

df = pd.DataFrame({
    "loan_id":           [f"L{str(i+1).zfill(6)}" for i in range(n)],
    "region":            regions,
    "loan_type":         loan_type,
    "credit_score":      credit_score,
    "loan_balance":      loan_balance,
    "debt_ratio":        debt_ratio,
    "num_delinquencies": num_delinquencies,
    "years_since_orig":  T,           # ← 経過年数（列名が違う）
    "default_prob":      np.round(pd_cumulative, 6),  # ← PD列名が違う
    "default_flag":      is_default,  # ← デフォルト列名が違う
})

df.to_csv("sample_data2.csv", index=False)
print(f"保存完了: sample_data2.csv ({len(df):,}行)")
print(df.head())
print(f"\nデフォルト率: {df['default_flag'].mean():.2%}")
print(f"PD中央値: {df['default_prob'].median():.4f}")
print(f"経過年数分布:\n{df['years_since_orig'].value_counts().sort_index()}")
print(f"\nT < 1 の件数: {(df['years_since_orig'] < 1).sum()}")
