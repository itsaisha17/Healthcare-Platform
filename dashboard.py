import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime, timedelta
import json
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx


API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


def _color_for_risk(level: str) -> str:
    return {
        "LOW": "#10b981",
        "MEDIUM": "#f59e0b",
        "HIGH": "#ef4444",
        "CRITICAL": "#7c3aed"
    }.get(str(level).upper(), "#6b7280")


def _severity_color(sev: str) -> str:
    return {
        "LOW": "#10b981", "MEDIUM": "#f59e0b",
        "HIGH": "#ef4444", "CRITICAL": "#7c3aed"
    }.get(str(sev).upper(), "#6b7280")


@st.cache_data(ttl=30, show_spinner=False)
def api_get(path: str, params: Optional[Dict[str, Any]] = None):
    try:
        with httpx.Client(timeout=20.0) as client:
            r = client.get(f"{API_BASE}{path}", params=params or {})
            if r.status_code == 200:
                return r.json()
    except Exception:
        return None
    return None


@st.cache_data(ttl=10, show_spinner=False)
def api_post(path: str, payload: Dict[str, Any]):
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(f"{API_BASE}{path}", json=payload, timeout=30.0)
            if r.status_code == 200:
                return r.json()
    except Exception as e:
        return {"error": str(e)}
    return None


def _check_api_alive() -> bool:
    data = api_get("/health")
    return data is not None


def _build_filters_ui():
    st.sidebar.markdown("### Filters")
    data_summary = api_get("/api/data-summary") or {}
    default_start = datetime.now() - timedelta(days=365)
    default_end = datetime.now()
    c1, c2 = st.sidebar.columns(2)
    start_date = c1.date_input("Start Date", value=default_start, max_value=default_end)
    end_date = c2.date_input("End Date", value=default_end, min_value=start_date)
    hospitals = data_summary.get("hospitals", [])
    departments = data_summary.get("departments", [])
    risk_levels = data_summary.get("risk_levels", [])
    sel_hosp = st.sidebar.multiselect("Hospital(s)", hospitals)
    sel_dept = st.sidebar.multiselect("Department(s)", departments)
    sel_risk = st.sidebar.multiselect("Risk Level(s)", risk_levels)
    params = {}
    params["start_date"] = start_date.isoformat()
    params["end_date"] = end_date.isoformat()
    if sel_hosp:
        params["hospital"] = sel_hosp
    if sel_dept:
        params["department"] = sel_dept
    if sel_risk:
        params["risk_level"] = sel_risk
    return params


def _render_kpi_card(label: str, value: Any, delta: str = "", color: str = "#3b82f6", icon: str = ""):
    st.markdown(
        f"""
        <div style="background-color:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:20px;margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div style="color:#94a3b8;font-size:13px;margin-bottom:6px;letter-spacing:0.3px;">{label.upper()}</div>
              <div style="color:#f8fafc;font-size:28px;font-weight:700;">{value}</div>
              {"<div style='color:#22c55e;font-size:12px;margin-top:6px;'>" + delta + "</div>" if delta else ""}
            </div>
            <div style="font-size:28px;opacity:0.8;">{icon}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def page_overview():
    st.title("🏥 Healthcare Analytics Overview")
    st.caption(f"Live dashboard powered by FastAPI + Plotly · Data updated every 30s")

    api_ok = _check_api_alive()
    if not api_ok:
        st.error(f"⚠️ Backend API is not reachable at {API_BASE}. Please start the backend (`python main.py`). Showing cached/mock data.")
        st.stop() if False else None

    params = _build_filters_ui()
    with st.spinner("Loading dashboard..."):
        dashboard = api_get("/api/dashboard/summary", params=params)

    if not dashboard:
        st.info("Waiting for backend data... Please ensure `python main.py` is running on port 8000.")
        st.stop()

    kpis = dashboard.get("kpis", {})

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _render_kpi_card("Total Patients", f"{kpis.get('total_patients', 0):,}", icon="👥")
    with k2:
        _render_kpi_card("Total Admissions", f"{kpis.get('total_admissions', 0):,}", icon="🏥")
    with k3:
        rr = kpis.get("readmission_rate", 0)
        delta = f"{rr}% readmitted within 30 days" if rr else ""
        _render_kpi_card("Readmission Rate", f"{rr}%", delta=delta, icon="🔁", color="#ef4444")
    with k4:
        _render_kpi_card("High Risk Patients", f"{kpis.get('high_risk_patient_count', 0):,}", icon="⚠️", color="#7c3aed")

    k5, k6, k7, k8 = st.columns(4)
    with k5:
        _render_kpi_card("Avg Length of Stay", f"{kpis.get('average_length_of_stay', 0)} days", icon="📅")
    with k6:
        _render_kpi_card("Avg Treatment Cost", f"${kpis.get('average_treatment_cost', 0):,.0f}", icon="💲")
    with k7:
        _render_kpi_card("Active Anomalies", f"{kpis.get('anomaly_count', 0)}", icon="🚨", color="#ef4444")
    with k8:
        _render_kpi_card("Departments", f"{kpis.get('total_departments', 0)}", icon="🏗️")

    st.divider()

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### 📈 Admissions Over Time")
        adm = dashboard.get("admissions_over_time", {}).get("data", [])
        if adm:
            df = pd.DataFrame(adm)
            fig = px.line(df, x="date", y="admissions", markers=True,
                          color_discrete_sequence=["#3b82f6"])
            fig.update_layout(height=320, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                              font_color="#e2e8f0", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No admissions data available for filters.")

    with g2:
        st.markdown("#### 🔁 Readmission Rate Trend (%)")
        rt = dashboard.get("readmission_trends", {}).get("data", [])
        if rt:
            df = pd.DataFrame(rt)
            fig = px.bar(df, x="date", y="rate", color_discrete_sequence=["#ef4444"],
                         text_auto=True)
            fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            fig.update_layout(height=320, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                              font_color="#e2e8f0", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No readmission trend data available.")

    g3, g4 = st.columns(2)
    with g3:
        st.markdown("#### ⚠️ Patient Risk Distribution")
        rd = dashboard.get("risk_distribution", {}).get("data", [])
        if rd:
            df = pd.DataFrame(rd)
            color_map = {r["risk_level"]: _color_for_risk(r["risk_level"]) for _, r in df.iterrows()}
            fig = px.bar(df, x="risk_level", y="count", color="risk_level",
                         color_discrete_map=color_map, text_auto=True)
            fig.update_layout(height=320, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                              font_color="#e2e8f0", showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with g4:
        st.markdown("#### 🏥 Length of Stay Distribution (days)")
        los = dashboard.get("los_distribution", {}).get("data", [])
        if los:
            df = pd.DataFrame(los)
            fig = px.bar(df, x="bucket", y="count", color_discrete_sequence=["#06b6d4"], text_auto=True)
            fig.update_layout(height=320, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                              font_color="#e2e8f0", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    g5, g6 = st.columns([3, 2])
    with g5:
        st.markdown("#### 💰 Average Treatment Cost Trend")
        ct = dashboard.get("cost_trends", {}).get("data", [])
        if ct:
            df = pd.DataFrame(ct)
            fig = px.area(df, x="date", y="avg_cost", color_discrete_sequence=["#10b981"])
            fig.update_layout(height=340, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                              font_color="#e2e8f0", margin=dict(l=10, r=10, t=20, b=10),
                              yaxis_tickprefix="$")
            st.plotly_chart(fig, use_container_width=True)

    with g6:
        st.markdown("#### 🧬 Top 10 Diagnoses")
        diag = dashboard.get("diagnosis_distribution", {}).get("data", [])
        if diag:
            df = pd.DataFrame(diag)
            fig = px.pie(df, names="diagnosis", values="count", hole=0.45)
            fig.update_layout(height=340, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                              font_color="#e2e8f0", margin=dict(l=10, r=10, t=20, b=10),
                              legend=dict(font=dict(size=10), orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.05))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### 📊 Department Performance")
    dp = dashboard.get("department_performance", {}).get("data", [])
    if dp:
        df = pd.DataFrame(dp)
        styled_cols = ["department", "admissions", "avg_los", "avg_cost", "readmission_rate"]
        col_config = {
            "department": "Department",
            "admissions": st.column_config.NumberColumn("Admissions", format="%d"),
            "avg_los": st.column_config.NumberColumn("Avg LOS (days)", format="%.2f"),
            "avg_cost": st.column_config.NumberColumn("Avg Cost ($)", format="$%.2f"),
            "readmission_rate": st.column_config.ProgressColumn("Readmission %", format="%.1f%%", min_value=0, max_value=50)
        }
        st.dataframe(df[styled_cols], use_container_width=True, hide_index=True, column_config=col_config, height=360)

    st.divider()
    st.markdown("#### 🚨 Anomalies by Severity Over Time")
    at = dashboard.get("anomaly_trends", {}).get("data", [])
    if at:
        df = pd.DataFrame(at)
        fig = px.bar(df, x="date", y="count", color="severity", barmode="stack",
                     color_discrete_map={s: _severity_color(s) for s in df["severity"].unique()})
        fig.update_layout(height=320, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                          font_color="#e2e8f0", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)


def page_patients():
    st.title("👥 Patient Records")
    st.caption("Browse and search patient admissions, filter by risk level and department.")
    api_ok = _check_api_alive()
    if not api_ok:
        st.error(f"Backend API not reachable at {API_BASE}. Start it with `python main.py`.")
        st.stop()

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    search = c1.text_input("Search (Patient ID / Diagnosis)", placeholder="e.g. P10042 or Diabetes")
    ds = api_get("/api/data-summary") or {}
    dept = c2.selectbox("Department", ["All"] + list(ds.get("departments", [])))
    risk = c3.selectbox("Risk Level", ["All"] + list(ds.get("risk_levels", [])))
    page_size = c4.number_input("Page Size", min_value=10, max_value=200, value=50, step=10)

    params = {"limit": int(page_size)}
    if search.strip():
        params["search"] = search.strip()
    if dept != "All":
        params["department"] = dept
    if risk != "All":
        params["risk_level"] = risk

    data = api_get("/api/patients", params=params) or []
    st.markdown(f"**{len(data)} records shown**")
    if data:
        df = pd.DataFrame(data)
        display_cols = ["id", "patient_id", "age", "gender", "department", "diagnosis_description",
                        "length_of_stay", "treatment_cost", "risk_level", "readmitted", "admission_date"]
        df = df[[c for c in display_cols if c in df.columns]]
        def style_risk(row):
            return [f"background-color: {_color_for_risk(row)}; color: white; padding: 4px 8px; border-radius: 6px; text-align: center; font-weight: 600;"] * len(row) if False else [""] * len(row)
        col_config = {
            "id": "#",
            "patient_id": "Patient ID",
            "age": st.column_config.NumberColumn("Age", format="%d"),
            "gender": "Gender",
            "department": "Department",
            "diagnosis_description": "Diagnosis",
            "length_of_stay": st.column_config.NumberColumn("LOS (days)", format="%d"),
            "treatment_cost": st.column_config.NumberColumn("Cost ($)", format="$%.2f"),
            "risk_level": st.column_config.Column("Risk Level"),
            "readmitted": st.column_config.CheckboxColumn("Readmitted"),
            "admission_date": st.column_config.DatetimeColumn("Admitted", format="YYYY-MM-DD")
        }

        styler = df.style
        def _risk_style(v):
            color = _color_for_risk(v)
            return f"background-color: {color}; color: white; border-radius: 4px; text-align: center; font-weight: 600;"
        if "risk_level" in df.columns:
            styler = styler.map(_risk_style, subset=["risk_level"])
        if "readmitted" in df.columns:
            def _readm_style(v):
                return "color: #ef4444; font-weight: 600;" if v else "color: #10b981;"
            styler = styler.map(_readm_style, subset=["readmitted"])
        st.dataframe(styler, use_container_width=True, height=600, column_config=col_config, hide_index=True)


def page_risk_predictor():
    st.title("🧠 Readmission Risk Predictor")
    st.caption("Run individual risk predictions using the active XGBoost readmission model.")

    api_ok = _check_api_alive()
    if not api_ok:
        st.error(f"Backend API not reachable.")
        st.stop()

    models = api_get("/api/models") or []
    if models:
        with st.expander("📦 Active Model Registry", expanded=False):
            mdf = pd.DataFrame(models)
            st.dataframe(mdf[["model_name", "model_version", "model_type", "metrics", "training_rows", "is_active"]],
                         use_container_width=True, hide_index=True)

    st.markdown("### Enter Patient Data")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 95, 60)
        gender = st.selectbox("Gender", ["M", "F", "Other"])
        num_diagnoses = st.slider("Number of Diagnoses", 1, 20, 3)
        num_medications = st.slider("Number of Medications", 0, 30, 5)
        num_lab_tests = st.slider("Number of Lab Tests", 1, 50, 12)
    with c2:
        length_of_stay = st.slider("Current Length of Stay (days)", 1, 30, 5)
        previous_admissions = st.slider("Previous Admissions (12mo)", 0, 10, 1)
        previous_readmissions = st.slider("Previous Readmissions", 0, previous_admissions, max(0, previous_admissions // 3))
        treatment_cost = st.number_input("Treatment Cost to Date ($)", 500, 200000, 18000, 500)
        department = st.selectbox("Department", ["Cardiology", "Internal Medicine", "Oncology", "Neurology", "Orthopedics", "ER"])

    if st.button("🔮 Predict Readmission Risk", type="primary", use_container_width=True):
        with st.spinner("Running inference..."):
            payload = {
                "patient_data": {
                    "age": age, "gender": gender,
                    "num_diagnoses": num_diagnoses, "num_medications": num_medications,
                    "num_lab_tests": num_lab_tests, "length_of_stay": length_of_stay,
                    "previous_admissions": previous_admissions,
                    "previous_readmissions": previous_readmissions,
                    "treatment_cost": treatment_cost, "department": department
                }
            }
            result = api_post("/api/predict", payload)
        if not result or "error" in result:
            st.error("Prediction failed. Please try again.")
        else:
            r1, r2, r3 = st.columns(3)
            score = result.get("risk_score", 0)
            level = result.get("risk_level", "LOW")
            prob = result.get("probability", 0)
            color = _color_for_risk(level)
            with r1:
                st.metric("Risk Score", f"{score:.1f} / 100", delta=level, delta_color="off")
            with r2:
                st.markdown(f"<div style='background:{color};color:white;padding:22px;border-radius:14px;text-align:center;'><div style='opacity:0.85;font-size:12px;'>RISK LEVEL</div><div style='font-size:30px;font-weight:800;letter-spacing:1px;'>{level}</div></div>",
                            unsafe_allow_html=True)
            with r3:
                st.metric("Readmission Probability", f"{prob * 100:.1f}%")

            st.divider()
            st.markdown("### Top Feature Contributions")
            top_feats = result.get("top_features", [])
            if top_feats:
                fdf = pd.DataFrame(top_feats)
                fig = px.bar(fdf, x="feature", y="contribution", color="contribution",
                             color_continuous_scale="Reds", text_auto=".3f")
                fig.update_layout(height=340, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                                  font_color="#e2e8f0", margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.info(f"📝 Model: **{result.get('model_name', 'N/A')} v{result.get('model_version', '')}**  \n{result.get('disclaimer', '')}")


def page_anomalies():
    st.title("🚨 Anomaly Detection")
    st.caption("System-detected anomalies across cost, readmission, LOS and volume metrics.")
    api_ok = _check_api_alive()
    if not api_ok:
        st.error(f"Backend API not reachable.")
        st.stop()
    anomalies = api_get("/api/anomalies") or []
    if not anomalies:
        st.info("No anomalies detected.")
        return

    df = pd.DataFrame(anomalies)
    cols = st.columns(4)
    sev_counts = df["severity"].value_counts()
    total = len(df)
    for idx, (level, count) in enumerate([("CRITICAL", sev_counts.get("CRITICAL", 0)),
                                          ("HIGH", sev_counts.get("HIGH", 0)),
                                          ("MEDIUM", sev_counts.get("MEDIUM", 0)),
                                          ("LOW", sev_counts.get("LOW", 0))]):
        with cols[idx]:
            color = _severity_color(level)
            st.markdown(f"<div style='border:1px solid {color}33;border-radius:12px;padding:14px;background:#0f172a;'>"
                        f"<div style='color:#94a3b8;font-size:12px;'>{level.upper()}</div>"
                        f"<div style='font-size:28px;font-weight:700;color:{color}'>{count}</div>"
                        f"<div style='color:#64748b;font-size:11px;margin-top:4px;'>of {total} total anomalies</div>"
                        f"</div>", unsafe_allow_html=True)

    st.divider()
    severity_filter = st.multiselect("Filter by Severity", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM"])
    if severity_filter:
        df = df[df["severity"].isin(severity_filter)]

    for _, r in df.iterrows():
        color = _severity_color(r["severity"])
        with st.container(border=True):
            hcol1, hcol2, hcol3, hcol4 = st.columns([4, 1, 1, 1])
            with hcol1:
                st.markdown(f"<div style='display:flex;align-items:center;gap:10px;'>"
                            f"<span style='background:{color};color:white;padding:3px 10px;border-radius:10px;font-size:11px;font-weight:700;letter-spacing:0.5px;'>{r['severity']}</span>"
                            f"<span style='background:#1e293b;color:#93c5fd;padding:3px 10px;border-radius:10px;font-size:11px;font-weight:600;'>{r['anomaly_type']}</span>"
                            f"<span style='color:#e2e8f0;font-size:15px;font-weight:600;'>{r.get('description','Anomaly')}</span>"
                            f"</div>", unsafe_allow_html=True)
            with hcol2:
                st.caption(f"**{r['affected_department']}**")
            with hcol3:
                st.caption(f"Score: **{r['anomaly_score']:.2f}**")
            with hcol4:
                dt = r.get("affected_date") or r.get("created_at", "")
                st.caption(str(dt)[:10] if dt else "")

            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Expected", f"${r.get('expected_value', 0):,.0f}")
            with bc2:
                delta_pct = 0
                if r.get("expected_value", 0) > 0:
                    delta_pct = ((r.get("actual_value", 0) - r.get("expected_value", 0)) / r.get("expected_value", 1)) * 100
                st.metric("Actual", f"${r.get('actual_value', 0):,.0f}", delta=f"{delta_pct:+.1f}%")
            with bc3:
                st.json(r.get("supporting_data", {}), expanded=False)


def page_insights():
    st.title("💡 AI-Generated Insights")
    st.caption("Automated trend analysis, drivers, and recommended investigations.")
    api_ok = _check_api_alive()
    if not api_ok:
        st.error("Backend API not reachable.")
        st.stop()

    insights = api_get("/api/insights") or []
    if not insights:
        st.info("No insights generated yet.")
        return

    sev = st.sidebar.multiselect("Severity Filter", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM"])
    if sev:
        insights = [i for i in insights if i.get("severity") in sev]

    st.markdown(f"**{len(insights)} insights**")
    for ins in insights:
        color = _severity_color(ins.get("severity", "MEDIUM"))
        conf = ins.get("confidence", 0) * 100
        with st.container(border=True):
            t1, t2, t3 = st.columns([5, 1, 1])
            with t1:
                st.markdown(f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
                            f"<span style='background:{color};color:white;padding:4px 12px;border-radius:10px;font-weight:700;letter-spacing:0.5px;'>{ins.get('severity','')}</span>"
                            f"<span style='background:#1e293b;color:#a5b4fc;padding:4px 12px;border-radius:10px;font-weight:600;font-size:12px;'>{ins.get('insight_type','')}</span>"
                            f"</div>", unsafe_allow_html=True)
                st.subheader(ins.get("title", "Untitled Insight"))
            with t2:
                st.metric("Confidence", f"{conf:.0f}%")
            with t3:
                st.caption(ins.get("created_at", "")[:10])

            st.markdown(f"**Finding:** {ins.get('finding', '')}")
            st.markdown(f"**What changed:** {ins.get('what_changed', '')}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Possible Drivers**")
                drivers = ins.get("possible_drivers") or []
                for d in drivers:
                    st.markdown(f"- ⚙️ {d}")
            with c2:
                st.markdown("**Evidence**")
                st.json(ins.get("evidence", {}), expanded=False)

            with st.expander("🔍 Recommended Investigation Plan"):
                st.write(ins.get("recommended_investigation", ""))


def page_whatif():
    st.title("🎯 What-If Scenario Simulator")
    st.caption("Simulate the impact of interventions on key population-level KPIs.")
    api_ok = _check_api_alive()
    if not api_ok:
        st.error("Backend API not reachable.")
        st.stop()

    baseline = api_get("/api/kpis") or {}
    if not baseline:
        baseline = {"readmission_rate": 16.5, "average_length_of_stay": 5.2, "average_treatment_cost": 15400, "high_risk_patient_count": 80}

    with st.form("whatif_form"):
        st.markdown("### Define Scenario")
        s_name = st.text_input("Scenario Name", value="Discharge Planning Optimization")
        s_desc = st.text_area("Scenario Description (optional)",
                              value="Enhanced discharge follow-up + TCM phone calls within 48 hours for high-risk patients.")
        st.caption("Adjust the intervention sliders below to simulate impact on baseline KPIs.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Readmission Rate (%)**")
            rr_cur = float(baseline.get("readmission_rate", 0))
            rr_new = st.slider("Target Readmission Rate %", min_value=max(1.0, rr_cur * 0.4),
                               max_value=rr_cur * 1.2, value=rr_cur * 0.85, step=0.1, key="rr_new",
                               help="Intervention target readmission rate.")
            st.caption(f"Current: **{rr_cur:.2f}%** → Target: **{rr_new:.2f}%** ({(rr_new - rr_cur):+.2f}pp)")

            st.markdown("**Average LOS (days)**")
            los_cur = float(baseline.get("average_length_of_stay", 0))
            los_new = st.slider("Target Avg LOS (days)", min_value=max(0.5, los_cur * 0.5),
                                max_value=los_cur * 1.3, value=los_cur * 0.9, step=0.1, key="los_new")
            st.caption(f"Current: **{los_cur:.2f}d** → Target: **{los_new:.2f}d**")

        with c2:
            st.markdown("**Avg Treatment Cost ($)**")
            cost_cur = float(baseline.get("average_treatment_cost", 0))
            cost_new = st.slider("Target Avg Cost ($)", min_value=max(500.0, cost_cur * 0.5),
                                 max_value=cost_cur * 1.3, value=cost_cur * 0.95, step=100.0, key="cost_new")
            st.caption(f"Current: **${cost_cur:,.0f}** → Target: **${cost_new:,.0f}**")

            st.markdown("**High-Risk Patient Count**")
            hr_cur = float(baseline.get("high_risk_patient_count", 0))
            hr_new = st.slider("Target High-Risk Count", min_value=max(0.0, hr_cur * 0.3),
                               max_value=hr_cur * 1.2, value=hr_cur * 0.75, step=1.0, key="hr_new")
            st.caption(f"Current: **{hr_cur:.0f}** → Target: **{hr_new:.0f}**")

        submitted = st.form_submit_button("▶️  Run Simulation", type="primary", use_container_width=True)

    if submitted:
        params = [
            {"name": "readmission_rate", "current_value": rr_cur, "new_value": rr_new,
             "description": "Intervention target on 30-day readmission rate"},
            {"name": "average_length_of_stay", "current_value": los_cur, "new_value": los_new,
             "description": "Target reduction in average length of stay"},
            {"name": "average_treatment_cost", "current_value": cost_cur, "new_value": cost_new,
             "description": "Target average treatment cost per admission"},
            {"name": "high_risk_patient_count", "current_value": hr_cur, "new_value": hr_new,
             "description": "Target number of high-risk patients after intervention"}
        ]
        payload = {"scenario_name": s_name, "description": s_desc, "parameters": params}
        with st.spinner("Running scenario simulation..."):
            result = api_post("/api/what-if", payload)

        if not result or "error" in result:
            st.error(f"Simulation failed: {result.get('error', 'Unknown error') if result else 'No response'}")
        else:
            st.success("✅ Simulation complete")
            cur = result.get("current_metrics", {})
            sim = result.get("simulated_metrics", {})
            rows = []
            label_map = {
                "readmission_rate": "Readmission Rate (%)",
                "average_length_of_stay": "Avg Length of Stay (days)",
                "average_treatment_cost": "Avg Treatment Cost ($)",
                "high_risk_patient_count": "High-Risk Patient Count",
                "total_admissions": "Total Admissions"
            }
            for k in ["readmission_rate", "average_length_of_stay", "average_treatment_cost", "high_risk_patient_count", "total_admissions"]:
                c_val = cur.get(k, 0)
                s_val = sim.get(k, c_val)
                delta = s_val - c_val if c_val else 0
                delta_pct = (delta / c_val * 100) if c_val else 0
                rows.append({"Metric": label_map.get(k, k), "Baseline": c_val, "Simulated": s_val,
                             "Δ Abs": delta, "Δ %": delta_pct})
            comp_df = pd.DataFrame(rows)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                rr_delta = (sim.get("readmission_rate", 0) - cur.get("readmission_rate", 0))
                st.metric("Readmission Rate", f"{sim.get('readmission_rate',0):.2f}%", delta=f"{rr_delta:+.2f}pp")
            with m2:
                los_delta = (sim.get("average_length_of_stay", 0) - cur.get("average_length_of_stay", 0))
                st.metric("Avg LOS", f"{sim.get('average_length_of_stay',0):.2f}d", delta=f"{los_delta:+.2f}d")
            with m3:
                cost_delta = (sim.get("average_treatment_cost", 0) - cur.get("average_treatment_cost", 0))
                st.metric("Avg Cost", f"${sim.get('average_treatment_cost',0):,.0f}", delta=f"${cost_delta:+,.0f}")
            with m4:
                hr_delta = (sim.get("high_risk_patient_count", 0) - cur.get("high_risk_patient_count", 0))
                st.metric("High-Risk Patients", f"{sim.get('high_risk_patient_count',0):.0f}", delta=f"{hr_delta:+.0f}")

            st.divider()
            st.markdown("### Comparison Chart")
            long_rows = []
            for k in ["readmission_rate", "average_length_of_stay", "average_treatment_cost", "high_risk_patient_count"]:
                long_rows.append({"Metric": label_map.get(k, k), "Scenario": "Baseline", "Value": cur.get(k, 0)})
                long_rows.append({"Metric": label_map.get(k, k), "Scenario": "Simulated", "Value": sim.get(k, 0)})
            ldf = pd.DataFrame(long_rows)
            fig = px.bar(ldf, x="Metric", y="Value", color="Scenario", barmode="group",
                         color_discrete_map={"Baseline": "#64748b", "Simulated": "#10b981"},
                         text_auto=True)
            fig.update_layout(height=420, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                              font_color="#e2e8f0", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📄 Simulation Results Table"):
                def fmt(v, k):
                    if "Cost" in k:
                        return f"${v:,.2f}"
                    if "Rate" in k or "%" in k:
                        return f"{v:.2f}%"
                    if "Stay" in k or "days" in k.lower():
                        return f"{v:.2f}d"
                    return f"{v:,.0f}"
                comp_df["Baseline"] = comp_df.apply(lambda r: fmt(r["Baseline"], r["Metric"]), axis=1)
                comp_df["Simulated"] = comp_df.apply(lambda r: fmt(r["Simulated"], r["Metric"]), axis=1)
                comp_df["Δ Abs"] = comp_df.apply(lambda r: (f"+" if r["Δ Abs"] > 0 else "") + (
                    f"${r['Δ Abs']:,.2f}" if "Cost" in r["Metric"] else
                    (f"{r['Δ Abs']:+.2f}pp" if "Rate" in r["Metric"] else
                     (f"{r['Δ Abs']:+.2f}d" if "Stay" in r["Metric"] else f"{r['Δ Abs']:+,.0f}"))), axis=1)
                comp_df["Δ %"] = comp_df["Δ %"].apply(lambda v: f"{v:+.1f}%")
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

            st.caption(f"ℹ️ {result.get('disclaimer', '')}")


def main():
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] { background-color: #020617; }
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetric"] label { font-size: 11px !important; letter-spacing: 0.3px; }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("🏥 Healthcare Analytics")
    st.sidebar.caption("Decision Intelligence Platform v1.0")
    status = "✅ API OK" if _check_api_alive() else "❌ API Offline"
    color = "#10b981" if _check_api_alive() else "#ef4444"
    st.sidebar.markdown(f"<div style='color:{color};font-weight:600;'>{status} · {API_BASE}</div>", unsafe_allow_html=True)
    st.sidebar.divider()

    pages = {
        "📊 Overview Dashboard": page_overview,
        "👥 Patient Records": page_patients,
        "🧠 Risk Predictor": page_risk_predictor,
        "🚨 Anomaly Detection": page_anomalies,
        "💡 AI Insights": page_insights,
        "🎯 What-If Simulator": page_whatif
    }
    sel = st.sidebar.radio("Navigation", list(pages.keys()))
    st.sidebar.divider()
    pages[sel]()


if __name__ == "__main__":
    main()
