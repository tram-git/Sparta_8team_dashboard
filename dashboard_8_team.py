import re
import os
import base64
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="줄기세포 분화 공정 대시보드", layout="wide", page_icon="📅")

_IMG_PATH = os.path.join(os.path.dirname(__file__), "dna_watermark_clean.png")
with open(_IMG_PATH, "rb") as _f:
    _DNA_B64 = base64.b64encode(_f.read()).decode()



st.markdown(
    '<style>'
    '[data-testid="stSidebar"] {'
    '    background: linear-gradient(180deg, #1e2a3a 0%, #243447 100%) !important;'
    '    border-right: 1px solid #2e3f55 !important;'
    '}'
    '[data-testid="stSidebar"] * {'
    '    color: #dce8f5 !important;'
    '}'
    '[data-testid="stSidebar"] hr {'
    '    border-color: #2e3f55 !important;'
    '}'
    '[data-testid="stMetric"] {'
    '    background: #ffffff;'
    '    border: 1px solid #e8ecf0;'
    '    border-radius: 12px;'
    '    padding: 18px 20px !important;'
    '    box-shadow: 0 2px 8px rgba(0,0,0,0.07);'
    '    min-height: 150px;'
    '}'
    '[data-testid="stMetricLabel"] p {'
    '    font-size: 0.82em !important;'
    '    color: #6b7280 !important;'
    '    font-weight: 600 !important;'
    '}'
    '[data-testid="stMetricValue"] {'
    '    font-size: 1.9em !important;'
    '    font-weight: 700 !important;'
    '    color: #111827 !important;'
    '}'
    'div[data-testid="stSelectbox"] > div > div {'
    '    border: 1.5px solid #d1d5db !important;'
    '    border-radius: 8px !important;'
    '    box-shadow: 0 1px 3px rgba(0,0,0,0.05);'
    '}'
    'hr { border-color: #e5e7eb !important; }'
    'h2, h3 { color: #111827 !important; font-weight: 700 !important; }'
    '[data-testid="stExpander"] {'
    '    border: 1px solid #e5e7eb !important;'
    '    border-radius: 10px !important;'
    '    box-shadow: 0 1px 4px rgba(0,0,0,0.05);'
    '}'
    'thead tr th {'
    '    background: #f3f4f6 !important;'
    '    font-weight: 600 !important;'
    '}'
    '.page-banner {'
    '    position:relative; overflow:hidden;'
    '    background:linear-gradient(135deg,rgba(255,255,255,0.97) 0%,rgba(250,252,255,0.94) 55%,rgba(241,246,255,0.92) 100%);'
    '    border:1px solid #D8E3F2; border-radius:24px;'
    '    padding:28px 34px 26px 34px;'
    '    box-shadow:0 10px 28px rgba(23,50,92,0.07); margin-bottom:18px;'
    '}'
    '.page-banner::after {'
    '    content:""; position:absolute; inset:auto 0 0 0; height:4px;'
    '    background:linear-gradient(90deg,#2D67C7 0%,#7EA5E8 55%,rgba(126,165,232,0) 100%);'
    '}'
    '.banner-topline { display:flex; align-items:center; gap:16px; }'
    '.banner-icon {'
    '    width:68px; height:68px; border-radius:50%;'
    '    display:flex; align-items:center; justify-content:center;'
    '    background:linear-gradient(180deg,#F4F8FF 0%,#EAF2FF 100%);'
    '    border:1px solid #D7E4FA;'
    '    box-shadow:inset 0 1px 0 rgba(255,255,255,0.9),0 4px 10px rgba(45,103,199,0.08);'
    '    font-size:30px; color:#2D67C7; flex:0 0 68px;'
    '}'
    '.banner-title {'
    '    font-size:3.0rem; line-height:1.1; font-weight:800; color:#17325C; margin:0;'
    '    letter-spacing:-0.03em;'
    '}'
    '.banner-subtitle {'
    '    margin:10px 0 0 0; font-size:1.05rem; color:#7B8BA8; font-weight:600;'
    '}'
    '</style>',
    unsafe_allow_html=True
)

st.markdown(
    f"""<style>
    .stApp {{
        background: linear-gradient(rgba(255,255,255,0.75), rgba(255,255,255,0.75)),
                    url("data:image/png;base64,{_DNA_B64}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    }}
    </style>""",
    unsafe_allow_html=True
)

def render_page_banner(title, subtitle, icon='◉'):
    st.markdown(f"""
    <div class='page-banner'>
        <div class='banner-topline'>
            <div class='banner-icon'>{icon}</div>
            <div>
                <div class='banner-title'>{title}</div>
                <div class='banner-subtitle'>{subtitle}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 헬퍼 함수
# ══════════════════════════════════════════════════════════════════
def add_gradient_features(df):
    df = df.copy()
    for a, b, c in [
        ("dd1 Glucose Concentration", "dd0 Glucose Concentration", "dd0-dd1 Glucose Gradient"),
        ("dd3 Glucose Concentration", "dd1 Glucose Concentration", "dd1-dd3 Glucose Gradient"),
        ("dd5 Glucose Concentration", "dd3 Glucose Concentration", "dd3-dd5 Glucose Gradient"),
        ("dd7 Glucose Concentration", "dd5 Glucose Concentration", "dd5-dd7 Glucose Gradient"),
        ("dd1 Lactate Concentration", "dd0 Lactate Concentration", "dd0-dd1 Lactate Gradient"),
        ("dd3 Lactate Concentration", "dd1 Lactate Concentration", "dd1-dd3 Lactate Gradient"),
        ("dd5 Lactate Concentration", "dd3 Lactate Concentration", "dd3-dd5 Lactate Gradient"),
        ("dd7 Lactate Concentration", "dd5 Lactate Concentration", "dd5-dd7 Lactate Gradient"),
    ]:
        df[c] = df[a] - df[b]
    return df

def col_max_day(col):
    c = col.strip()
    if c.startswith("Overall"):
        return 7
    m = re.match(r"dd(\d+)-dd(\d+)", c)
    if m:
        return int(m.group(2))
    all_dd = [int(x) for x in re.findall(r"dd(\d+)", c)]
    return max(all_dd) if all_dd else 0

def categorize_col(col):
    c = col.lower().strip()
    if "cell density" in c:  return "세포 밀도"
    if "aggregate" in c:     return "응집체 크기"
    if "glucose" in c:       return "포도당"
    if "lactate" in c:       return "젖산"
    if "ph" in c:            return "pH"
    if "do" in c or "dissolved" in c or "2nd derivative" in c:
        return "용존산소"
    return "공정 조건"

# ══════════════════════════════════════════════════════════════════
# 데이터 로드 (고정)
# ══════════════════════════════════════════════════════════════════
N_ESTIMATORS  = 190
CONTAMINATION = 0.20
CM_THRESHOLD  = 70

@st.cache_data
def load_raw():
    base  = os.path.dirname(__file__)
    train = pd.read_csv(os.path.join(base, "train_data_merged_cleaned.csv"), encoding="utf-8")
    test  = pd.read_csv(os.path.join(base, "test_data_merged_cleaned.csv"),  encoding="utf-8")
    return train, test

df_train_raw, df_test_raw = load_raw()
df_train_feat = add_gradient_features(df_train_raw)
df_test_feat  = add_gradient_features(df_test_raw)

FEATURE_COLS = [c for c in df_train_feat.columns if c != "dd10 CM Content"]
X_all_train  = df_train_feat[FEATURE_COLS]
X_all_test   = df_test_feat[FEATURE_COLS]

y_train = np.where(df_train_raw["dd10 CM Content"] >= CM_THRESHOLD, 1, 0)

# ══════════════════════════════════════════════════════════════════
# 사이드바 — 날짜 선택 + 페이지 네비게이션
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📅 측정 기준 날짜")
    selected_day = st.select_slider(
        label="날짜",
        options=[0, 1, 2, 3, 5, 7],
        format_func=lambda x: f"Day {x}",
        value=7,
        label_visibility="collapsed",
    )
    avail_cols = [c for c in FEATURE_COLS if col_max_day(c) <= selected_day]
    st.caption(f"사용 가능 변수: {len(avail_cols)} / {len(FEATURE_COLS)}개")

    st.markdown("---")

    st.markdown("## 📄 페이지")
    page = st.radio(
        label="페이지",
        options=["🏠 공정 KPI", "🎯 예측 결과", "🔎 이상 원인 분석", "📋 데이터 분포"],
        label_visibility="collapsed",
    )

# ══════════════════════════════════════════════════════════════════
# 공통 전처리 (날짜 기준)
# ══════════════════════════════════════════════════════════════════
cat_map    = {col: categorize_col(col) for col in avail_cols}
cat_df     = pd.DataFrame(list(cat_map.items()), columns=["변수명", "카테고리"])
cat_counts = cat_df["카테고리"].value_counts()

X_train_day = X_all_train[avail_cols]
X_test_day  = X_all_test[avail_cols]

@st.cache_data
def run_model(day):
    a_cols = [c for c in FEATURE_COLS if col_max_day(c) <= day]
    Xtr = X_all_train[a_cols].values
    Xte = X_all_test[a_cols].values
    iso = IsolationForest(n_estimators=N_ESTIMATORS,
                          contamination=CONTAMINATION, random_state=42)
    iso.fit(Xtr)
    raw    = iso.predict(Xte)
    ypred  = np.where(raw == 1, 1, 0)
    sc     = iso.decision_function(Xte)
    return ypred.tolist(), sc.tolist()

y_pred_list, scores_list = run_model(selected_day)
y_pred = np.array(y_pred_list)
scores = np.array(scores_list)

# ══════════════════════════════════════════════════════════════════
# 공통 — 퍼터베이션 기여도 계산
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def compute_perturbation_importance(day, sidx):
    a_cols   = [c for c in FEATURE_COLS if col_max_day(c) <= day]
    Xtr_arr  = X_all_train[a_cols].values
    Xte_arr  = X_all_test[a_cols].values
    ytr      = np.where(df_train_raw["dd10 CM Content"] >= CM_THRESHOLD, 1, 0)

    iso = IsolationForest(n_estimators=N_ESTIMATORS,
                          contamination=CONTAMINATION, random_state=42)
    iso.fit(Xtr_arr)

    sample     = Xte_arr[sidx].copy()
    base_score = float(iso.decision_function(sample.reshape(1, -1))[0])

    train_median = np.median(Xtr_arr, axis=0)
    train_mean   = Xtr_arr.mean(axis=0)
    train_std    = Xtr_arr.std(axis=0)

    suc_mask = ytr == 1
    suc      = Xtr_arr[suc_mask] if suc_mask.any() else Xtr_arr
    suc_mean = suc.mean(axis=0)
    suc_q25  = np.percentile(suc, 25, axis=0)
    suc_q75  = np.percentile(suc, 75, axis=0)

    rows = []
    for j, col in enumerate(a_cols):
        pert = sample.copy()
        pert[j] = train_median[j]
        new_score = float(iso.decision_function(pert.reshape(1, -1))[0])
        delta     = new_score - base_score
        actual    = float(sample[j])
        std_j     = float(train_std[j]) or 1e-9
        z         = float((actual - train_mean[j]) / std_j)
        suc_z     = float((actual - suc_mean[j])   / std_j)
        pct       = float(np.mean(Xtr_arr[:, j] <= actual) * 100)
        in_range  = float(suc_q25[j]) <= actual <= float(suc_q75[j])

        rows.append({
            "변수":         col.strip(),
            "카테고리":     categorize_col(col),
            "샘플 실제값":  round(actual, 4),
            "Train 중앙값": round(float(train_median[j]), 4),
            "성공 Q25":     round(float(suc_q25[j]), 4),
            "성공 Q75":     round(float(suc_q75[j]), 4),
            "Z-Score":      round(z, 3),
            "성공군 Z":     round(suc_z, 3),
            "Train 백분위": round(pct, 1),
            "정상범위 내":  in_range,
            "방향":         "↑ 높음" if actual > float(train_mean[j]) else "↓ 낮음",
            "Score 기여도": round(delta, 6),
            "_col_orig":    col,
        })

    df_imp = pd.DataFrame(rows).sort_values("Score 기여도", ascending=False)
    return df_imp, base_score

# ══════════════════════════════════════════════════════════════════
# 페이지 헤더
# ══════════════════════════════════════════════════════════════════
if page != "🏠 공정 KPI":
    icon_map = {"🎯 예측 결과": "◎", "🔎 이상 원인 분석": "◉", "📋 데이터 분포": "▣"}
    render_page_banner(
        page.split(' ', 1)[1] if ' ' in page else page,
        f"Day {selected_day} 기준 | 사용 변수 {len(avail_cols)}개 | Test 샘플 {len(y_pred)}개 | 성공 기준 CM ≥ {CM_THRESHOLD}%",
        icon_map.get(page, "◉")
    )

# ══════════════════════════════════════════════════════════════════
# PAGE 0 · 공정 KPI (표지)
# ══════════════════════════════════════════════════════════════════
if page == "🏠 공정 KPI":
    n_total   = len(y_pred)
    n_success = int((y_pred == 1).sum())
    n_fail    = int((y_pred == 0).sum())
    n_caution = int(((scores > 0) & (scores < 0.01)).sum())

    render_page_banner(
        "줄기세포 분화 공정",
        f"Day {selected_day} 기준 | 사용 변수 {len(avail_cols)}개 | Test 샘플 {len(y_pred)}개 | 성공 기준 CM ≥ {CM_THRESHOLD}%",
        icon="▣"
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧫 전체 배치 수", f"{n_total}개")
    c2.metric("✅ 정상 예측", f"{n_success/n_total*100:.1f}%", f"{n_success}개", delta_color="normal")
    c3.metric("❌ 이상 탐지", f"{n_fail/n_total*100:.1f}%",   f"{n_fail}개",    delta_color="inverse")
    c4.metric("⚠️ 주의 샘플 수", f"{n_caution}개", "score 0 ~ 0.01")

    # ── 측정 기준일별 이상 탐지 수 변화 ──────────────────────────
    st.markdown("---")
    st.subheader("📈 측정 기준일별 이상 탐지 수 변화")
    st.caption("초기 데이터만으로도 최종 분화 실패를 얼마나 조기에 예측할 수 있는지 보여줍니다.")

    all_days      = [d for d in [0, 1, 2, 3, 5, 7] if d <= selected_day]
    day_counts    = []
    caution_counts = []
    for d in all_days:
        yp_d, sc_d = run_model(d)
        sc_arr = np.array(sc_d)
        day_counts.append(sum(1 for p in yp_d if p == 0))
        caution_counts.append(int(((sc_arr > 0) & (sc_arr < 0.01)).sum()))

    cur_count     = day_counts[all_days.index(selected_day)]
    cur_caution   = caution_counts[all_days.index(selected_day)]
    cur_yp, cur_sc = run_model(selected_day)

    anomaly_rows = sorted(
        [{"구분": "❌ 이상", "배치": f"S{i}", "Anomaly Score": round(cur_sc[i], 4)}
         for i, p in enumerate(cur_yp) if p == 0],
        key=lambda r: r["Anomaly Score"],
    )
    caution_rows = sorted(
        [{"구분": "⚠️ 주의", "배치": f"S{i}", "Anomaly Score": round(cur_sc[i], 4)}
         for i, s in enumerate(cur_sc)
         if 0 < s < 0.01],
        key=lambda r: r["Anomaly Score"],
    )

    lc, rc = st.columns([2, 1])

    with lc:
        x_labels = [f"Day {d}" for d in all_days]
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=x_labels, y=day_counts,
            mode="lines+markers+text",
            name="이상 탐지 수",
            line=dict(color="#F44336", width=2.5),
            marker=dict(size=8, color="#F44336"),
            text=[f"{c}개" for c in day_counts],
            textposition="top center",
        ))
        fig_trend.add_trace(go.Scatter(
            x=x_labels, y=caution_counts,
            mode="lines+markers+text",
            name="주의 샘플 수",
            line=dict(color="#FDD835", width=2.5, dash="dot"),
            marker=dict(size=8, color="#FDD835"),
            text=[f"{c}개" for c in caution_counts],
            textposition="bottom center",
        ))
        fig_trend.update_layout(
            title="측정 기준일별 이상 탐지 수",
            xaxis_title="측정 기준일",
            yaxis_title="샘플 수",
            height=350,
            margin=dict(t=50, b=30),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with rc:
        st.markdown(f"**Day {selected_day} 이상 / 주의 샘플**")
        combined = anomaly_rows + caution_rows
        if combined:
            def _tbl_style(row):
                color = "#ffebee" if row["구분"].startswith("❌") else "#fffde7"
                return [f"background-color:{color}"] * len(row)
            st.dataframe(
                pd.DataFrame(combined).style.apply(_tbl_style, axis=1),
                use_container_width=True,
                hide_index=True,
                height=350,
            )
        else:
            st.success("이상 / 주의 샘플 없음")



# ══════════════════════════════════════════════════════════════════
# PAGE 1 · 예측 결과
# ══════════════════════════════════════════════════════════════════
elif page == "🎯 예측 결과":

    n_det  = int((y_pred == 0).sum())
    n_norm = int((y_pred == 1).sum())

    st.markdown(
        f"""
        <div style='display:flex; gap:16px; margin:12px 0;'>
            <div style='flex:1; background:#ffffff; border:1px solid #e0e0e0;
                        border-radius:12px; padding:20px 24px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
                <div style='font-size:1.1em; color:#888; font-weight:500;
                            margin-bottom:8px;'>✅ 정상 예측 수</div>
                <div style='font-size:1.9em; font-weight:700; color:#212121;
                            line-height:1.2;'>{n_norm} / {len(y_pred)}</div>
            </div>
            <div style='flex:1; background:#ffffff; border:1px solid #e0e0e0;
                        border-radius:12px; padding:20px 24px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
                <div style='font-size:1.1em; color:#888; font-weight:500;
                            margin-bottom:8px;'>⚠️ 이상 탐지 수</div>
                <div style='font-size:1.9em; font-weight:700; color:#212121;
                            line-height:1.2;'>{n_det} / {len(y_pred)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.subheader("전체 샘플 Anomaly Score")

    anomaly_samples = [f"S{i}" for i, p in enumerate(y_pred) if p == 0]
    caution_samples = [f"S{i}" for i, (p, s) in enumerate(zip(y_pred, scores)) if p == 1 and 0 < s < 0.01]

    if anomaly_samples:
        st.error(f"**{', '.join(anomaly_samples)}** 는 이상 샘플로 예측되었습니다. 폐기를 권장합니다.")
    if caution_samples:
        st.warning(f"**{', '.join(caution_samples)}** 는 주의 샘플로 예측되었습니다. 조치를 권장합니다.")

    xlabels = [f"S{i}" for i in range(len(scores))]
    fig_bar = go.Figure(go.Bar(
        x=xlabels, y=scores,
        marker_color=["#F44336" if p == 0 else "#2196F3" for p in y_pred],
        text=["⚠️" if 0 < s < 0.01 else "" for s in scores],
        textposition="outside",
        textfont=dict(size=20),
    ))
    fig_bar.add_hline(y=0, line_dash="dash", line_color="gray",
                      annotation_text="Decision boundary")
    fig_bar.update_layout(
        xaxis_tickangle=-45, yaxis_title="Score", height=500,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── 권장 조치 방법 (주의 샘플) ──────────────────────────────
    if caution_samples:
        st.markdown("---")
        st.subheader("🛠️ 권장 조치 방법")
        st.caption(f"주의 샘플의 이상 기여도 상위 변수를 기반으로 조치 방법을 제안합니다.  |  Day {selected_day} 측정 변수 기준")

        ACTION_MAP = {
            ("세포 밀도",  "↑ 높음"): "접종 밀도를 낮추거나 계대 주기를 단축하세요.",
            ("세포 밀도",  "↓ 낮음"): "배지 교환 주기를 줄이고 성장 인자 농도를 점검하세요.",
            ("응집체 크기","↑ 높음"): "교반 속도를 높여 응집체를 분산시키세요.",
            ("응집체 크기","↓ 낮음"): "응집 촉진 조건(세포 농도, 교반 속도)을 재확인하세요.",
            ("포도당",     "↑ 높음"): "포도당 공급량을 줄이고 과잉 축적 여부를 모니터링하세요.",
            ("포도당",     "↓ 낮음"): "포도당 공급 스케줄을 앞당기거나 농도를 높이세요.",
            ("젖산",       "↑ 높음"): "포도당 공급을 줄이고 환기량 또는 pH 조절을 점검하세요.",
            ("젖산",       "↓ 낮음"): "대사 활성 저하 여부를 확인하고 배지 조성을 재검토하세요.",
            ("pH",         "↑ 높음"): "CO₂ 공급량을 높이거나 산성 보정액 투입을 검토하세요.",
            ("pH",         "↓ 낮음"): "염기 보정액(NaHCO₃ 등) 투입량을 늘리고 환기 조건을 점검하세요.",
            ("용존산소",   "↑ 높음"): "교반 속도 또는 O₂ 공급량을 줄이세요.",
            ("용존산소",   "↓ 낮음"): "O₂ 공급량 또는 교반 속도를 높여 산소 전달률을 개선하세요.",
            ("공정 조건",  "↑ 높음"): "해당 공정 파라미터를 정상 범위로 낮추는 조치가 필요합니다.",
            ("공정 조건",  "↓ 낮음"): "해당 공정 파라미터를 정상 범위로 높이는 조치가 필요합니다.",
        }

        CAT_ICON = {
            "세포 밀도":   "🧫",
            "응집체 크기": "🔬",
            "포도당":      "🍬",
            "젖산":        "⚗️",
            "pH":          "🧪",
            "용존산소":    "💨",
            "공정 조건":   "⚙️",
        }

        caution_idx = [i for i, (p, s) in enumerate(zip(y_pred, scores))
                       if p == 1 and 0 < s < 0.01]

        def _score_color(sc):
            if sc < 0.005: return "#E53935", "고위험"
            return          "#F9A825", "저위험"

        # ── 전체 기여도 데이터 수집 ──────────────────────────────
        all_card_data = []
        with st.spinner("기여도 분석 중…"):
            for idx in caution_idx:
                imp_df, _ = compute_perturbation_importance(selected_day, idx)
                day_imp   = imp_df[imp_df["변수"].apply(col_max_day) == selected_day]
                top3      = day_imp[day_imp["Score 기여도"] > 0].head(3)
                if top3.empty:
                    continue
                sc_val            = float(scores[idx])
                badge_color, risk = _score_color(sc_val)
                top_row           = top3.iloc[0]
                action            = ACTION_MAP.get(
                    (top_row["카테고리"], top_row["방향"]),
                    f"{top_row['변수']} 값을 정상 범위로 조정하는 조치가 필요합니다.",
                )
                all_card_data.append({
                    "idx":         idx,
                    "sc_val":      sc_val,
                    "badge_color": badge_color,
                    "risk":        risk,
                    "top3":        top3,
                    "action":      action,
                })

        # ── ① 카테고리별 그룹핑 요약 테이블 ────────────────────
        st.markdown("#### 카테고리별 문제 현황 요약")

        cat_group: dict = {}
        for card in all_card_data:
            for _, row in card["top3"].iterrows():
                cat  = row["카테고리"]
                icon = CAT_ICON.get(cat, "📌")
                key  = f"{icon} {cat}"
                if key not in cat_group:
                    cat_group[key] = {
                        "샘플": [],
                        "권장 조치": ACTION_MAP.get(
                            (cat, row["방향"]),
                            "해당 공정 파라미터를 점검하세요."
                        ),
                    }
                sid = f"S{card['idx']}"
                if sid not in cat_group[key]["샘플"]:
                    cat_group[key]["샘플"].append(sid)

        summary_rows = [
            {
                "문제 카테고리": k,
                "해당 샘플":     ", ".join(v["샘플"]),
                "샘플 수":       len(v["샘플"]),
                "권장 조치":     v["권장 조치"],
            }
            for k, v in sorted(cat_group.items(), key=lambda x: -len(x[1]["샘플"]))
        ]
        summary_df = pd.DataFrame(summary_rows)

        def _style_summary(row):
            n = row["샘플 수"]
            if n >= 3:   bg = "#FFEBEE"
            elif n >= 2: bg = "#FFF3E0"
            else:        bg = "#FFFDE7"
            return [f"background-color:{bg}"] * len(row)

        st.dataframe(
            summary_df.style.apply(_style_summary, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

        # ── ② 샘플별 토글 카드 (expander) ───────────────────────
        st.markdown("#### 샘플별 권장 조치 내역")
        st.caption("카드 제목을 클릭하면 상세 내용을 펼치거나 닫을 수 있습니다.")

        for card in all_card_data:
            idx         = card["idx"]
            sc_val      = card["sc_val"]
            badge_color = card["badge_color"]
            risk        = card["risk"]
            top3        = card["top3"]
            action      = card["action"]

            exp_label = (
                f"⚠️  S{idx}"
                f"   |   Anomaly Score: {sc_val:.4f}"
                f"   |   {risk}"
            )

            with st.expander(exp_label, expanded=False):
                col_var, col_dir, col_act = st.columns([3, 1.4, 3])

                with col_var:
                    st.markdown(
                        "<div style='background:#F5F5F5; border-radius:8px 0 0 8px;"
                        " padding:10px 14px;'>"
                        "<p style='font-size:0.75em; color:#888; font-weight:600;"
                        " margin:0 0 8px 0; letter-spacing:0.5px;'>🔴 문제 변수</p>",
                        unsafe_allow_html=True,
                    )
                    for _, row in top3.iterrows():
                        icon = CAT_ICON.get(row["카테고리"], "📌")
                        st.markdown(
                            f"<p style='margin:5px 0; font-size:0.88em; color:#212121;'>"
                            f"{icon} <b>{row['변수']}</b>"
                            f"<span style='color:#999; font-size:0.82em;'>"
                            f"  ({row['카테고리']})</span></p>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_dir:
                    st.markdown(
                        "<div style='background:#FAFAFA; padding:10px 14px;'>"
                        "<p style='font-size:0.75em; color:#888; font-weight:600;"
                        " margin:0 0 8px 0; letter-spacing:0.5px;'>📊 이상 방향</p>",
                        unsafe_allow_html=True,
                    )
                    for _, row in top3.iterrows():
                        up    = row["방향"] == "↑ 높음"
                        dclr  = "#E53935" if up else "#1565C0"
                        arrow = "▲" if up else "▼"
                        lbl   = "높음" if up else "낮음"
                        st.markdown(
                            f"<p style='margin:5px 0; font-size:0.9em;"
                            f" color:{dclr}; font-weight:700;'>"
                            f"{arrow} {lbl}</p>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_act:
                    st.markdown(
                        "<div style='background:#FFF8E1; border-radius:0 8px 8px 0;"
                        " padding:10px 14px;'>"
                        "<p style='font-size:0.75em; color:#888; font-weight:600;"
                        " margin:0 0 8px 0; letter-spacing:0.5px;'>🛠️ 권장 조치</p>"
                        f"<p style='margin:0; font-size:0.9em; color:#4E342E;"
                        f" line-height:1.7;'>→ {action}</p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )



# ══════════════════════════════════════════════════════════════════
# PAGE 2 · 이상 원인 분석
# ══════════════════════════════════════════════════════════════════
elif page == "🔎 이상 원인 분석":

    # 샘플 선택 — 이상 / 주의 샘플만
    flagged_idx = [
        i for i in range(len(y_pred))
        if y_pred[i] == 0 or (0 < scores[i] < 0.01)
    ]

    if not flagged_idx:
        st.info("이상 또는 주의 샘플이 없습니다.")
        st.stop()

    def _fmt(i):
        if y_pred[i] == 0:
            tag = "이상 ❌"
        else:
            tag = "주의 ⚠️"
        return f"샘플 {i}  |  {tag}  |  Score: {scores[i]:.4f}"

    st.markdown("""
        <style>
        div[data-testid="stSelectbox"] label {
            font-size: 1.1em;
            font-weight: 700;
            color: #2E58A6;
        }
        </style>
    """, unsafe_allow_html=True)

    t4_idx = st.selectbox(
        "분석할 Test 샘플 (이상 / 주의)",
        options=flagged_idx,
        format_func=_fmt,
    )

    is_anomaly = y_pred[t4_idx] == 0
    is_caution = (not is_anomaly) and (0 < scores[t4_idx] < 0.01)

    if is_anomaly:
        label_color, label_text, score_color = "#E53935", "이상 ❌", "#E53935"
    elif is_caution:
        label_color, label_text, score_color = "#F9A825", "주의 ⚠️", "#F9A825"
    else:
        label_color, label_text, score_color = "#43A047", "정상 ✅", "#43A047"

    st.markdown(
        f"""
        <div style='display:flex; gap:16px; margin:12px 0;'>
            <div style='flex:1; background:#ffffff; border:1px solid #e0e0e0;
                        border-radius:12px; padding:20px 24px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
                <div style='font-size:0.82em; color:#888; font-weight:500;
                            margin-bottom:8px;'>예측 레이블</div>
                <div style='font-size:1.9em; font-weight:700; color:#212121;
                            line-height:1.2;'>{label_text}</div>
            </div>
            <div style='flex:1; background:#ffffff; border:1px solid #e0e0e0;
                        border-radius:12px; padding:20px 24px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
                <div style='font-size:0.82em; color:#888; font-weight:500;
                            margin-bottom:8px;'>Anomaly Score (양수=정상, 음수=이상)</div>
                <div style='font-size:1.9em; font-weight:700; color:{score_color};
                            line-height:1.2;'>{scores[t4_idx]:.4f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("기여도 계산 중…"):
        imp_df, base_score = compute_perturbation_importance(selected_day, t4_idx)

    pos_imp = imp_df[imp_df["Score 기여도"] > 0]

    # ── 자연어 요약 ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("이상 탐지 원인 요약")

    if is_anomaly or is_caution:
        top5 = pos_imp.head(5)
        if is_anomaly:
            st.error(f"샘플 {t4_idx}  -  **이상(anomaly)** 으로 탐지되었습니다.")
        else:
            st.warning(
                f"샘플 {t4_idx}  -  **주의(caution)** 로 탐지되었습니다. "
                f"(Score = {base_score:.4f}, 정상 경계 근접)."
            )
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            direction = "**높아서**" if row["방향"] == "↑ 높음" else "**낮아서**"
            sev = ("🔴 매우 심각" if abs(row["Z-Score"]) > 3 else
                   "🟠 심각"      if abs(row["Z-Score"]) > 2 else "🟡 경미")
            out = " *(성공 사분위 범위 벗어남)*" if not row["정상범위 내"] else ""
            st.markdown(
                f"**{rank}위.** `{row['변수']}` {sev}  \n"
                f"→ 실제값 **{row['샘플 실제값']}** 이 정상 기준({row['Train 중앙값']}) 대비 "
                f"{direction} 이상에 기여했습니다{out}.  \n"
                f"&nbsp;&nbsp;&nbsp;(Train 내 {row['Train 백분위']:.0f}th 백분위 / "
                f"Z = {row['Z-Score']:+.2f} / 기여도 = {row['Score 기여도']:+.5f})"
            )
    else:
        st.success(f"샘플 {t4_idx}은 정상으로 탐지되었습니다 (Score = {base_score:.4f}).")

    # ── 기여도 수평 바 차트 ──────────────────────────────────────
    st.markdown("---")
    st.subheader("이상 기여도 상위 변수")
    st.caption("각 변수를 Train 중앙값(정상 기준)으로 교체했을 때 Score 상승량 = 이상 기여도")

    top15 = imp_df.head(15)
    bar_colors = [
        "#B71C1C" if d > 0.004 else
        "#E53935" if d > 0.002 else
        "#EF9A9A" if d > 0     else
        "#B0BEC5"
        for d in top15["Score 기여도"]
    ]
    hover_texts = [
        (f"카테고리: {r['카테고리']}<br>실제값: {r['샘플 실제값']}<br>"
         f"Train 중앙값: {r['Train 중앙값']}<br>"
         f"성공 Q25~Q75: {r['성공 Q25']} ~ {r['성공 Q75']}<br>"
         f"Z-Score: {r['Z-Score']:+.2f}<br>방향: {r['방향']}")
        for _, r in top15.iterrows()
    ]
    fig_imp = go.Figure(go.Bar(
        y=top15["변수"], x=top15["Score 기여도"],
        orientation="h", marker_color=bar_colors,
        hovertext=hover_texts, hoverinfo="text",
        text=[f"{r['방향']}  실제:{r['샘플 실제값']}  /  정상:{r['Train 중앙값']}"
              for _, r in top15.iterrows()],
        textposition="outside",
    ))
    fig_imp.add_vline(x=0, line_color="black", line_width=1.5)
    fig_imp.update_layout(
        title="막대가 길수록 이 변수가 이상 탐지의 주 원인",
        xaxis_title="Score 기여도",
        yaxis=dict(autorange="reversed"),
        height=max(400, len(top15) * 32),
        margin=dict(l=20, r=220, t=50),
    )
    st.plotly_chart(fig_imp, use_container_width=True)



# ══════════════════════════════════════════════════════════════════
# PAGE 3 · 데이터 분포
# ══════════════════════════════════════════════════════════════════
else:

    cc, vc = st.columns(2)
    with cc:
        sel_cat = st.selectbox("카테고리", ["전체"] + list(cat_counts.index))
    with vc:
        fc = avail_cols if sel_cat == "전체" else \
             cat_df[cat_df["카테고리"] == sel_cat]["변수명"].tolist()

        def _base(col):
            c = col.strip()
            # 접두사: "dd0-dd1 ..." 형태 (gradient prefix)
            m = re.match(r'^dd\d+-dd\d+\s+(.+)$', c)
            if m: return m.group(1)
            # 접두사: "dd0 ..." 형태
            m = re.match(r'^dd\d+\s+(.+)$', c)
            if m: return m.group(1)
            # 접미사: "... d0" 또는 "... dd0" 형태 (DO 계열)
            m = re.match(r'^(.+?)\s+dd?\d+$', c)
            if m: return m.group(1)
            return c

        base_options = list(dict.fromkeys(_base(c) for c in fc))
        sel_base = st.selectbox("변수 그룹", base_options)

    def _x_sort(col):
        c = col.strip()
        m = re.match(r'^dd\d+-dd(\d+)\s+', c)
        if m: return int(m.group(1))
        m = re.match(r'^dd(\d+)\s+', c)
        if m: return int(m.group(1))
        # 접미사 패턴: "... d3" 또는 "... dd3"
        m = re.search(r'\s+dd?(\d+)$', c)
        if m: return int(m.group(1))
        return col_max_day(col)

    def _x_label(col):
        c = col.strip()
        m = re.match(r'^(dd\d+-dd\d+)\s+', c)
        if m: return m.group(1)
        m = re.match(r'^dd(\d+)\s+', c)
        if m: return f"Day {m.group(1)}"
        # 접미사 패턴
        m = re.search(r'\s+dd?(\d+)$', c)
        if m: return f"Day {m.group(1)}"
        return f"Day {col_max_day(col)}"

    group_cols = sorted(
        [c for c in fc if _base(c) == sel_base],
        key=_x_sort,
    )

    # ── 일자별 값 분포 ───────────────────────────────────────────
    st.subheader("일자별 값 분포")
    x_order = [_x_label(c) for c in group_cols]

    pt_rows = []
    for i in range(len(X_all_test)):
        lbl = "정상 예측" if y_pred[i] == 1 else "이상 예측"
        for col in group_cols:
            pt_rows.append({
                "샘플": f"S{i}",
                "x":    _x_label(col),
                "x_s":  _x_sort(col),
                "값":   float(X_all_test[col].iloc[i]),
                "예측": lbl,
            })
    pt_df   = pd.DataFrame(pt_rows)
    mean_df = pt_df.groupby(["x", "x_s", "예측"])["값"].mean().reset_index()

    fig_line = px.scatter(
        pt_df, x="x", y="값", color="예측",
        color_discrete_map={"정상 예측": "#2196F3", "이상 예측": "#F44336"},
        opacity=0.45,
        title=f"{sel_base} — 일자별 값",
        labels={"x": "", "값": sel_base},
        hover_data=["샘플"],
        category_orders={"x": x_order},
    )
    mean_mode = "lines+markers" if len(group_cols) > 1 else "markers"
    for lbl, clr in [("정상 예측", "#1565C0"), ("이상 예측", "#C62828")]:
        grp = mean_df[mean_df["예측"] == lbl].sort_values("x_s")
        fig_line.add_trace(go.Scatter(
            x=grp["x"], y=grp["값"],
            mode=mean_mode,
            line=dict(color=clr, width=2.5),
            marker=dict(size=9, symbol="diamond"),
            name=f"{lbl} 평균",
        ))
    fig_line.update_layout(
        xaxis=dict(categoryorder="array", categoryarray=x_order),
        height=350, margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # ── 변수 분포 탐색 ────────────────────────────────────────────
    st.subheader("변수 분포 탐색")
    sel_col = group_cols[-1] if group_cols else (fc[0] if fc else avail_cols[0])
    plot_test = X_all_test[[sel_col]].copy()
    plot_test["예측"] = ["정상 예측" if p == 1 else "이상 예측" for p in y_pred]

    fig_box = px.box(
        plot_test, y=sel_col, x="예측", color="예측",
        color_discrete_map={"정상 예측": "#2196F3", "이상 예측": "#F44336"},
        points="all", title=f"{sel_col.strip()} — Box Plot",
    )
    st.plotly_chart(fig_box, use_container_width=True)
