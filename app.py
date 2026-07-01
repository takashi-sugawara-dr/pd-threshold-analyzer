import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
from sklearn.metrics import roc_auc_score, roc_curve

# 日本語グリフ警告を防ぐため、ユニコード対応フォントを優先設定
def _set_font():
    candidates = ["Hiragino Sans", "Hiragino Kaku Gothic Pro", "AppleGothic",
                  "Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.family"] = c
            return
_set_font()

st.set_page_config(page_title="PD調整ツール", layout="wide")
st.title("PD 年率調整・閾値分析ツール")

SAMPLE_PATH = "sample_data.csv"

# ══════════════════════════════════════════════════════════════════════
# サイドバー
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("データ読み込み")
    uploaded = st.file_uploader("CSVファイルをアップロード")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        data_key = uploaded.name
        st.success(f"{len(df):,} 行読み込み完了")
    else:
        if os.path.exists(SAMPLE_PATH):
            df = pd.read_csv(SAMPLE_PATH)
            data_key = SAMPLE_PATH
            st.info(f"サンプルデータを使用中（{len(df):,} 行）")

    if uploaded is None and not os.path.exists(SAMPLE_PATH):
        st.warning("CSVをアップロードしてください")

if uploaded is None and not os.path.exists(SAMPLE_PATH):
    st.info("左のサイドバーからCSVファイルをアップロードしてください。")
    st.stop()

# データソースが変わったらsession stateをリセット
if st.session_state.get("data_key") != data_key:
    st.session_state["data_key"]      = data_key
    st.session_state["col_confirmed"] = False
    st.session_state["adjusted"]      = False
    st.session_state["pd_annual"]     = None

with st.sidebar:
    st.header("列選択")
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    def guess(candidates):
        for c in candidates:
            if c in cols:
                return cols.index(c)
        if numeric_cols:
            return cols.index(numeric_cols[0])
        return 0

    col_default = st.selectbox("実績デフォルト列（0/1）", cols,
        index=guess(["actual_default", "default_flag", "default", "def", "デフォルト"]))
    col_pd = st.selectbox("貸し倒れ確率（PD）列", cols,
        index=guess(["pd_score", "default_prob", "pd", "PD", "prob_default"]))
    col_t = st.selectbox("経過年数（T）列", cols,
        index=guess(["elapsed_years", "years_since_orig", "years", "t", "T", "term"]))

    confirmed = st.button("この列で分析を開始", type="primary")

    st.divider()
    st.header("収益単価設定")
    st.caption("3シナリオ比較タブでも使用されます")
    unit_tn = st.number_input("TN 1件あたり収益（円）",          value=100_000,  step=10_000)
    unit_fn = st.number_input("FN 1件あたり損失（負値で入力）",  value=-500_000, step=10_000)
    unit_tp = st.number_input("TP 1件あたり機会利得（円）",      value=50_000,   step=10_000)
    unit_fp = st.number_input("FP 1件あたり機会損失（負値で入力）", value=-80_000, step=10_000)
    if unit_fn > 0:
        st.warning("FNは貸し倒れ損失のため、通常は負値を入力します。")
    if unit_fp > 0:
        st.warning("FPは機会損失のため、通常は負値を入力します。")

# ── 列確定前 ──────────────────────────────────────────────────────────
if "col_confirmed" not in st.session_state:
    st.session_state["col_confirmed"] = False
if confirmed:
    st.session_state["col_confirmed"] = True
    st.session_state["adjusted"] = False
    st.session_state["pd_annual"] = None

if not st.session_state["col_confirmed"]:
    st.info("左サイドバーで列を選択し「この列で分析を開始」を押してください。")
    st.dataframe(df.head(20))
    st.stop()

try:
    y_true = df[col_default].astype(float)
    pd_raw  = df[col_pd].astype(float)
    T_vals  = df[col_t].astype(float)
except Exception as e:
    st.error(f"列の数値変換に失敗しました: {e}")
    st.session_state["col_confirmed"] = False
    st.stop()

unique_vals = set(y_true.dropna().unique())
if not unique_vals.issubset({0.0, 1.0}):
    preview = sorted(unique_vals)[:5]
    suffix = f"… 他{len(unique_vals)-5}種類" if len(unique_vals) > 5 else ""
    st.error(f"実績デフォルト列「{col_default}」に0/1以外の値が含まれています: {preview}{suffix}")
    st.session_state["col_confirmed"] = False
    st.stop()

if pd_raw.dropna().max() > 1.0 or pd_raw.dropna().min() < 0.0:
    st.warning(f"PD列「{col_pd}」に0〜1の範囲外の値があります。自動的にclipします。")
    pd_raw = pd_raw.clip(0, 1)

# ── ユーティリティ ────────────────────────────────────────────────────
def calc_metrics(threshold, pd_arr, y_arr):
    approved  = pd_arr < threshold
    tp = int(((~approved) & (y_arr == 1)).sum())
    fp = int(((~approved) & (y_arr == 0)).sum())
    fn = int((approved    & (y_arr == 1)).sum())
    tn = int((approved    & (y_arr == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return tp, fp, fn, tn, precision, recall, float(approved.mean())

def calc_revenue(tp_, fp_, fn_, tn_):
    return tn_ * unit_tn + fn_ * unit_fn + tp_ * unit_tp + fp_ * unit_fp

THRESHOLDS = np.linspace(0.01, 0.99, 200)

@st.cache_data
def scan_thresholds(pd_arr, y_arr, u_tn, u_fn, u_tp, u_fp):
    """全閾値を一度だけ走査してDataFrameに集約する。単価が変わると自動再計算。"""
    rows = []
    for thr in THRESHOLDS:
        tp_, fp_, fn_, tn_, prec_, rec_, ar_ = calc_metrics(thr, pd_arr, y_arr)
        rev_ = tn_ * u_tn + fn_ * u_fn + tp_ * u_tp + fp_ * u_fp
        rows.append(dict(thr=thr, tp=tp_, fp=fp_, fn=fn_, tn=tn_,
                         precision=prec_, recall=rec_, ar=ar_, revenue=rev_))
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════
# タブ
# ══════════════════════════════════════════════════════════════════════
tab_main, tab_compare = st.tabs(["分析", "3シナリオ比較"])

# ──────────────────────────────────────────────────────────────────────
# TAB 1: メイン分析
# ──────────────────────────────────────────────────────────────────────
with tab_main:

    # 1. データプレビュー
    st.header("1. データプレビュー")
    st.dataframe(df.head(50))
    st.caption(f"行数: {len(df):,}　列数: {len(df.columns)}")

    # 2. 選択データの可視化
    st.header("2. 選択データの可視化")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(pd_raw.dropna(), bins=50, color="#4C72B0", edgecolor="white")
    axes[0].set_title("PD Distribution (Raw)"); axes[0].set_xlabel("PD"); axes[0].set_ylabel("Count")
    axes[1].hist(T_vals.dropna(), bins=30, color="#55A868", edgecolor="white")
    axes[1].set_title("Elapsed Years (T) Distribution"); axes[1].set_xlabel("T (years)"); axes[1].set_ylabel("Count")
    axes[2].bar(["Normal (0)", "Default (1)"],
                [int((y_true == 0).sum()), int((y_true == 1).sum())],
                color=["#4C72B0", "#C44E52"])
    axes[2].set_title("Actual Default Breakdown"); axes[2].set_ylabel("Count")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # 3. 年率調整
    st.header("3. 年率調整")
    st.latex(r"PD_{annual} = 1 - (1 - PD_T)^{\frac{1}{T}}")
    st.caption(
        "データには異なる経過年数 T の債権が混在しています。"
        "この式で T 年間の累積PDを1年あたりの年率PDに換算し、全件を同一基準で比較します。"
        "　**T < 1（融資から1年未満）は 1/T > 1 となりPDが膨らむため、年率換算をスキップし元のPDを使用します。**"
    )
    do_adjust = st.button("年率調整を実行")

    if "pd_annual" not in st.session_state:
        st.session_state["pd_annual"] = None
    if "adjusted" not in st.session_state:
        st.session_state["adjusted"] = False

    if do_adjust:
        t_ok = T_vals >= 1
        T_safe = T_vals.clip(lower=1.0)
        pd_converted = 1 - (1 - pd_raw.clip(0, 1 - 1e-9)) ** (1.0 / T_safe)
        pd_annual = pd_converted.where(t_ok, pd_raw)
        n_skipped = int((~t_ok & T_vals.notna()).sum())
        if n_skipped > 0:
            st.warning(f"T < 1 のデータが {n_skipped:,} 件あります。これらは年率調整をスキップし元のPDを使用します。")
        st.session_state["pd_annual"] = pd_annual.clip(0, 1)
        st.session_state["adjusted"] = True

    if st.session_state["adjusted"]:
        pd_annual = st.session_state["pd_annual"]
        pd_use = pd_annual

        valid_mask_raw = pd_raw.notna() & T_vals.notna() & pd_annual.notna()
        t_unique = sorted(T_vals[valid_mask_raw].unique())
        cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(t_unique), 1))

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
        axes2[0].hist(pd_raw.dropna(), bins=50, alpha=0.6, color="#4C72B0", label="Before", edgecolor="white")
        axes2[0].hist(pd_annual.dropna(), bins=50, alpha=0.6, color="#C44E52", label="After (Annual)", edgecolor="white")
        axes2[0].set_title("PD Distribution: Before vs After")
        axes2[0].set_xlabel("PD"); axes2[0].set_ylabel("Count"); axes2[0].legend()
        for idx, t_val in enumerate(t_unique):
            mask_t = valid_mask_raw & (T_vals == t_val)
            axes2[1].scatter(pd_raw[mask_t], pd_annual[mask_t], alpha=0.4, s=5, color=cmap(idx), label=f"T={t_val:g}")
        axes2[1].plot([0, 1], [0, 1], "r--", linewidth=1, label="y=x (T=1)")
        axes2[1].set_title("Scatter: Raw PD vs Annual PD")
        axes2[1].set_xlabel("Raw PD"); axes2[1].set_ylabel("Annual PD")
        axes2[1].legend(title="Elapsed Years", markerscale=3, fontsize=8)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

        stats_df = pd.DataFrame({
            "統計量":           ["平均", "中央値", "25%点", "75%点", "90%点"],
            "調整前 PD":        [f"{pd_raw.mean():.4f}",    f"{pd_raw.quantile(0.50):.4f}",
                                 f"{pd_raw.quantile(0.25):.4f}", f"{pd_raw.quantile(0.75):.4f}",
                                 f"{pd_raw.quantile(0.90):.4f}"],
            "調整後 PD（年率）": [f"{pd_annual.mean():.4f}", f"{pd_annual.quantile(0.50):.4f}",
                                  f"{pd_annual.quantile(0.25):.4f}", f"{pd_annual.quantile(0.75):.4f}",
                                  f"{pd_annual.quantile(0.90):.4f}"],
        })
        st.dataframe(stats_df, hide_index=True)
    else:
        pd_use = pd_raw
        st.info("年率調整を行わない場合はそのまま進めます。「年率調整を実行」ボタンで適用できます。")

    # 共通マスク
    valid_mask = pd_use.notna() & y_true.notna()
    pd_clean = pd_use[valid_mask].values
    y_clean  = y_true[valid_mask].values

    # 閾値スキャン（キャッシュ済み）
    scan = scan_thresholds(pd_clean, y_clean, unit_tn, unit_fn, unit_tp, unit_fp)

    if st.session_state["adjusted"]:
        st.success("以降の分析はすべて **年率調整済みPD** を使用しています。")
    else:
        st.warning("以降の分析はすべて **調整前の元PD** を使用しています。年率調整を行う場合はセクション3のボタンを押してください。")

    # 4. 閾値設定
    st.header("4. 閾値設定と指標算出")
    threshold_mode = st.radio(
        "閾値の設定方法",
        ["PDしきい値を直接指定", "承認率から逆算", "最大利益Threshold（自動）"],
        horizontal=True,
    )

    if threshold_mode == "PDしきい値を直接指定":
        pd_threshold = st.slider("PDしきい値（これ以上は否決）",
                                 min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f")
    elif threshold_mode == "承認率から逆算":
        ar_target = st.slider("目標承認率（%）", min_value=1, max_value=100, value=80, step=1) / 100.0
        pd_threshold = float(np.percentile(pd_clean, ar_target * 100))
        st.write(f"対応するPDしきい値: **{pd_threshold:.4f}**")
        st.caption("※ 同一PDスコアが多数の場合、実際の承認率が目標値と微小にズレることがあります。")
    else:
        st.info("収益単価（サイドバー）をもとに最大利益Thresholdを自動算出します。")
        pd_threshold = float(scan.loc[scan["revenue"].idxmax(), "thr"])
        st.success(f"最大利益Threshold: **{pd_threshold:.3f}**")

    tp, fp, fn, tn, precision, recall, approval_rate = calc_metrics(pd_threshold, pd_clean, y_clean)
    actual_default_rate = y_clean.mean()
    try:
        auc = roc_auc_score(y_clean, pd_clean)
    except Exception:
        auc = float("nan")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("承認率",             f"{approval_rate:.1%}")
    m2.metric("実績デフォルト率",   f"{actual_default_rate:.2%}")
    m3.metric("AUC",               f"{auc:.4f}")
    m4.metric("適合率（Precision）", f"{precision:.2%}")
    m5.metric("再現率（Recall）",   f"{recall:.2%}")

    # 5. 混同行列
    st.header("5. 混同行列")
    cm_data = np.array([[tn, fp], [fn, tp]])
    cell_labels = [["TN\n(Approved & Normal)", "FP\n(Rejected & Normal)"],
                   ["FN\n(Approved & Default)", "TP\n(Rejected & Default)"]]
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.imshow(cm_data, cmap="Blues")
    for i in range(2):
        for j in range(2):
            val = cm_data[i, j]
            ax3.text(j, i, f"{cell_labels[i][j]}\n{val:,}",
                     ha="center", va="center", fontsize=10,
                     color="white" if val > cm_data.max() * 0.6 else "black")
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Approved (Pred)", "Rejected (Pred)"])
    ax3.set_yticklabels(["Normal (Actual)", "Default (Actual)"])
    ax3.set_xlabel("Model Prediction"); ax3.set_ylabel("Actual Outcome")
    ax3.set_title("Confusion Matrix")
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    # 6. 収益試算
    st.header("6. 収益試算")
    st.markdown("""
| セル | 意味 | 収益/損失 |
|------|------|-----------|
| **TN** | 承認 & 正常 | 収益 |
| **FN** | 承認 & デフォルト（見逃し） | 損失 |
| **TP** | 否決 & デフォルト（正しく否決） | 機会利得 |
| **FP** | 否決 & 正常（誤って否決） | 機会損失 |
""")
    rev_tn = tn * unit_tn
    rev_fn = fn * unit_fn
    rev_tp = tp * unit_tp
    rev_fp = fp * unit_fp
    total_revenue = rev_tn + rev_fn + rev_tp + rev_fp

    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    rc1.metric("TN 収益",   f"¥{rev_tn:,.0f}")
    rc2.metric("FN 損益",   f"¥{rev_fn:,.0f}")
    rc3.metric("TP 機会利得", f"¥{rev_tp:,.0f}")
    rc4.metric("FP 機会損失", f"¥{rev_fp:,.0f}")
    rc5.metric("合計収益",   f"¥{total_revenue:,.0f}", delta=f"¥{total_revenue:,.0f}")

    # Waterfall chart
    st.subheader("収益 Waterfall Chart")
    wf_labels = ["TN", "FN", "TP", "FP", "Total"]
    wf_values = [rev_tn, rev_fn, rev_tp, rev_fp, total_revenue]
    running = 0; bottoms, heights, bar_colors = [], [], []
    for v in wf_values[:-1]:
        bottoms.append(min(running, running + v)); heights.append(abs(v))
        bar_colors.append("#4C72B0" if v >= 0 else "#C44E52"); running += v
    bottoms.append(min(0, total_revenue)); heights.append(abs(total_revenue))
    bar_colors.append("#2ca02c" if total_revenue >= 0 else "#d62728")
    fig_wf, ax_wf = plt.subplots(figsize=(8, 4))
    x = np.arange(len(wf_labels))
    ax_wf.bar(x, heights, bottom=bottoms, color=bar_colors, edgecolor="white", width=0.5)
    spread = max(abs(v) for v in wf_values) or 1
    for i, (b, h, v) in enumerate(zip(bottoms, heights, wf_values)):
        ax_wf.text(x[i], b + h + spread * 0.01, f"JPY {v:,.0f}", ha="center", va="bottom", fontsize=8)
        if i < len(wf_values) - 2:
            ax_wf.plot([x[i]+0.25, x[i+1]-0.25], [b+h if v>=0 else b]*2, "k--", linewidth=0.7, alpha=0.5)
    ax_wf.set_xticks(x); ax_wf.set_xticklabels(wf_labels)
    ax_wf.axhline(0, color="black", linewidth=0.8)
    ax_wf.set_title("Revenue Waterfall"); ax_wf.set_ylabel("Amount (JPY)")
    ax_wf.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"JPY {v:,.0f}"))
    plt.tight_layout(); st.pyplot(fig_wf); plt.close()

    # 7. 最大利益Threshold
    st.header("7. 最大利益Threshold")
    best_idx = scan["revenue"].idxmax()
    best_thr = float(scan.loc[best_idx, "thr"])
    best_rev = float(scan.loc[best_idx, "revenue"])
    st.columns(2)[0].metric("最大利益", f"¥{best_rev:,.0f}")
    st.columns(2)[1].metric("最適Threshold", f"{best_thr:.3f}")
    fig5, ax5 = plt.subplots(figsize=(9, 4))
    ax5.plot(scan["thr"], scan["revenue"] / 1e6, color="#4C72B0", linewidth=2)
    ax5.axvline(best_thr, color="#C44E52", linestyle="--", linewidth=1.5, label=f"Optimal: {best_thr:.3f}")
    ax5.axvline(pd_threshold, color="#55A868", linestyle=":", linewidth=1.5, label=f"Current: {pd_threshold:.3f}")
    ax5.scatter([best_thr], [best_rev / 1e6], color="#C44E52", zorder=5, s=60)
    ax5.set_title("Revenue vs PD Threshold"); ax5.set_xlabel("PD Threshold"); ax5.set_ylabel("Revenue (M JPY)")
    ax5.legend(); ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"JPY {v:.1f}M"))
    plt.tight_layout(); st.pyplot(fig5); plt.close()

    # 8. 承認率 vs 利益
    st.header("8. 承認率 vs 利益 トレードオフ")
    ar_curve = scan["ar"].values
    rv_curve = scan["revenue"].values
    fig6, ax6 = plt.subplots(figsize=(9, 4))
    ax6.plot(ar_curve * 100, rv_curve / 1e6, color="#4C72B0", linewidth=2)
    ax6.scatter([approval_rate * 100], [total_revenue / 1e6], color="#55A868", zorder=5, s=80,
                label=f"Current (AR={approval_rate:.0%})")
    ax6.scatter([ar_curve[best_idx] * 100], [best_rev / 1e6], color="#C44E52", zorder=5, s=80,
                marker="*", label=f"Optimal (AR={ar_curve[best_idx]:.0%})")
    ax6.set_title("Approval Rate vs Revenue Tradeoff")
    ax6.set_xlabel("Approval Rate (%)"); ax6.set_ylabel("Revenue (M JPY)")
    ax6.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"JPY {v:.1f}M"))
    ax6.legend(); plt.tight_layout(); st.pyplot(fig6); plt.close()

    # 9. ROC曲線
    st.header("9. ROC曲線")
    fpr_arr, tpr_arr, _ = roc_curve(y_clean, pd_clean)
    # 現在閾値のFPR/TPRは混同行列から直接計算（roc_curve thresholdは降順+先頭∞のためズレが生じる）
    n_pos = int(y_clean.sum()); n_neg = len(y_clean) - n_pos
    current_fpr = fp / n_neg if n_neg > 0 else 0.0
    current_tpr = tp / n_pos if n_pos > 0 else 0.0
    fig7, ax7 = plt.subplots(figsize=(6, 5))
    ax7.plot(fpr_arr, tpr_arr, color="#4C72B0", linewidth=2, label=f"ROC (AUC={auc:.4f})")
    ax7.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax7.scatter([current_fpr], [current_tpr], color="#C44E52", zorder=5, s=100,
                label=f"Threshold={pd_threshold:.3f}\nFPR={current_fpr:.3f}, TPR={current_tpr:.3f}")
    ax7.set_title("ROC Curve"); ax7.set_xlabel("False Positive Rate"); ax7.set_ylabel("True Positive Rate")
    ax7.legend(fontsize=8); ax7.set_xlim(0, 1); ax7.set_ylim(0, 1)
    plt.tight_layout(); st.pyplot(fig7); plt.close()

    # 10. PD分布と閾値
    st.header("10. PD分布と閾値")
    fig8, ax8 = plt.subplots(figsize=(10, 4))
    ax8.hist(pd_clean[y_clean == 0], bins=60, alpha=0.6, color="#4C72B0", label="Normal (0)", edgecolor="white")
    ax8.hist(pd_clean[y_clean == 1], bins=60, alpha=0.6, color="#C44E52", label="Default (1)", edgecolor="white")
    ax8.axvline(pd_threshold, color="black", linewidth=2, linestyle="--", label=f"Threshold = {pd_threshold:.3f}")
    ymax = ax8.get_ylim()[1]
    ax8.text(pd_threshold + 0.01, ymax * 0.85, f">> Reject\n   (PD >= {pd_threshold:.3f})", color="black", fontsize=9)
    ax8.text(pd_threshold - 0.01, ymax * 0.85, f"Approve <<\n(PD < {pd_threshold:.3f})", color="black", fontsize=9, ha="right")
    ax8.set_title("PD Distribution with Threshold"); ax8.set_xlabel("PD"); ax8.set_ylabel("Count"); ax8.legend()
    plt.tight_layout(); st.pyplot(fig8); plt.close()

    # 11. サマリ
    st.header("11. サマリ")
    summary_df = pd.DataFrame({
        "指標": ["AUC", "実績デフォルト率", "承認率", "PDしきい値",
                  "適合率（Precision）", "再現率（Recall）", "合計収益（円）", "最適Threshold", "最大利益（円）"],
        "値":   [f"{auc:.4f}", f"{actual_default_rate:.2%}", f"{approval_rate:.1%}",
                 f"{pd_threshold:.4f}", f"{precision:.2%}", f"{recall:.2%}",
                 f"¥{total_revenue:,.0f}", f"{best_thr:.4f}", f"¥{best_rev:,.0f}"],
    })
    st.dataframe(summary_df, hide_index=True)

    # 12. ダウンロード
    st.header("12. レポートダウンロード")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("サマリをCSVダウンロード",
                           summary_df.to_csv(index=False, encoding="utf-8-sig"),
                           file_name="pd_summary.csv", mime="text/csv")
    with dl2:
        export_df = df.copy()
        export_df["pd_annual"] = np.nan
        export_df.loc[pd_use.index, "pd_annual"] = pd_use.values
        export_df["approved"] = 0
        export_df.loc[pd_use.index, "approved"] = (pd_use < pd_threshold).astype(int)
        st.download_button("件別スコアをCSVダウンロード",
                           export_df.to_csv(index=False, encoding="utf-8-sig"),
                           file_name="pd_detail.csv", mime="text/csv")

# ──────────────────────────────────────────────────────────────────────
# TAB 2: 3シナリオ比較
# ──────────────────────────────────────────────────────────────────────
with tab_compare:
    st.header("3シナリオ KPI比較")
    st.caption("同じデータ・単価設定のもとで3つの閾値戦略を並べて比較します。単価はサイドバーで変更できます。")

    # 年率調整済みPDが必要
    if not st.session_state["col_confirmed"]:
        st.info("「分析」タブで列を確定してください。")
        st.stop()

    valid_mask2 = pd_use.notna() & y_true.notna()
    pd_c = pd_use[valid_mask2].values
    y_c  = y_true[valid_mask2].values

    # シナリオ1: ユーザー指定PD閾値
    st.subheader("シナリオ入力")
    sc1, sc2 = st.columns(2)
    with sc1:
        s1_thr = st.slider("① PDしきい値", min_value=0.0, max_value=1.0, value=0.5, step=0.001,
                           format="%.3f", key="s1_thr")
    with sc2:
        s2_ar = st.slider("② 目標承認率（%）", min_value=1, max_value=100, value=80, step=1,
                          key="s2_ar") / 100.0
        s2_thr = float(np.percentile(pd_c, s2_ar * 100))

    # シナリオ3: 最大利益（自動）- キャッシュ済みscanを再利用
    scan2 = scan_thresholds(pd_c, y_c, unit_tn, unit_fn, unit_tp, unit_fp)
    s3_thr = float(scan2.loc[scan2["revenue"].idxmax(), "thr"])

    scenarios = {
        "① PD Threshold": s1_thr,
        "② Approval Rate": s2_thr,
        "③ Max Revenue":   s3_thr,
    }

    label_ja = {
        "① PD Threshold":  "① PDしきい値直接指定",
        "② Approval Rate": "② 承認率から逆算",
        "③ Max Revenue":   "③ 最大利益（自動）",
    }

    # KPI計算
    rows = []
    for name, thr in scenarios.items():
        tp_, fp_, fn_, tn_, prec_, rec_, ar_ = calc_metrics(thr, pd_c, y_c)
        rev_ = calc_revenue(tp_, fp_, fn_, tn_)
        rows.append({
            "シナリオ":            label_ja[name],
            "PDしきい値":          f"{thr:.4f}",
            "承認率":              f"{ar_:.1%}",
            "適合率（Precision）": f"{prec_:.2%}",
            "再現率（Recall）":    f"{rec_:.2%}",
            "TN":                  f"{tn_:,}",
            "FN":                  f"{fn_:,}",
            "TP":                  f"{tp_:,}",
            "FP":                  f"{fp_:,}",
            "合計収益（円）":      f"¥{rev_:,.0f}",
            "_rev":                rev_,
            "_ar":                 ar_,
        })

    compare_df = pd.DataFrame(rows)

    # KPIテーブル（_rev, _arは表示しない）
    st.subheader("KPI一覧")
    display_cols = [c for c in compare_df.columns if not c.startswith("_")]
    st.dataframe(compare_df[display_cols], hide_index=True)

    # 収益棒グラフ比較
    st.subheader("合計収益の比較")
    fig_c1, ax_c1 = plt.subplots(figsize=(8, 4))
    sc_names  = list(scenarios.keys())   # 英語キー（matplotlib用）
    sc_revs   = [r["_rev"] for r in rows]
    bar_cols  = ["#4C72B0" if v >= 0 else "#C44E52" for v in sc_revs]
    bars = ax_c1.bar(sc_names, sc_revs, color=bar_cols, edgecolor="white")
    ax_c1.axhline(0, color="black", linewidth=0.8)
    spread_c = max(abs(v) for v in sc_revs) or 1
    for bar, val in zip(bars, sc_revs):
        ax_c1.text(bar.get_x() + bar.get_width() / 2,
                   val + spread_c * 0.02,
                   f"JPY {val:,.0f}", ha="center", va="bottom", fontsize=9)
    ax_c1.set_title("Revenue by Scenario"); ax_c1.set_ylabel("Amount (JPY)")
    ax_c1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"JPY {v:,.0f}"))
    plt.tight_layout(); st.pyplot(fig_c1); plt.close()

    # 承認率・Precision・Recall レーダー的棒グラフ
    st.subheader("承認率 / Precision / Recall の比較")
    kpi_names = ["Approval Rate", "Precision", "Recall"]
    sc_thrs   = list(scenarios.values())
    kpi_vals  = [
        [r["_ar"] for r in rows],
        [calc_metrics(thr, pd_c, y_c)[4] for thr in sc_thrs],
        [calc_metrics(thr, pd_c, y_c)[5] for thr in sc_thrs],
    ]
    x_pos = np.arange(len(sc_names))
    width = 0.25
    fig_c2, ax_c2 = plt.subplots(figsize=(9, 4))
    colors_kpi = ["#4C72B0", "#55A868", "#C44E52"]
    for i, (kname, kvals) in enumerate(zip(kpi_names, kpi_vals)):
        ax_c2.bar(x_pos + i * width, kvals, width=width, label=kname,
                  color=colors_kpi[i], edgecolor="white", alpha=0.85)
    ax_c2.set_xticks(x_pos + width); ax_c2.set_xticklabels(sc_names, fontsize=9)
    ax_c2.set_ylim(0, 1.1); ax_c2.set_ylabel("Rate")
    ax_c2.set_title("Approval Rate / Precision / Recall by Scenario")
    ax_c2.legend(); ax_c2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout(); st.pyplot(fig_c2); plt.close()

    # CSV ダウンロード
    st.download_button("シナリオ比較をCSVダウンロード",
                       compare_df[display_cols].to_csv(index=False, encoding="utf-8-sig"),
                       file_name="pd_scenario_compare.csv", mime="text/csv")
