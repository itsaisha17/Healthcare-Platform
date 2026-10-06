import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings

from backend.schemas.schemas import (
    KPIsResponse, ChartDataResponse, DashboardDataResponse,
    PatientRecordResponse, PatientRecordCreate,
    ModelInfoResponse, PredictionRequest, PredictionResponse,
    AnomalyRecordResponse, AIInsightResponse,
    DecisionActionResponse, WhatIfRequest, WhatIfResponse,
    DataSummaryResponse, DashboardFilters
)


_MOCK_PATIENTS: List[Dict[str, Any]] = []
_MOCK_INSIGHTS: List[Dict[str, Any]] = []
_MOCK_ANOMALIES: List[Dict[str, Any]] = []
_MOCK_MODELS: List[Dict[str, Any]] = []


def _generate_mock_patients(n: int = 500) -> List[Dict[str, Any]]:
    hospitals = ["General Hospital", "St. Mary's Medical", "City Health Center", "Metropolitan Hospital"]
    departments = ["Cardiology", "Neurology", "Orthopedics", "Internal Medicine", "Pediatrics", "Oncology", "ER"]
    diagnoses = [
        ("I21", "Acute Myocardial Infarction"), ("J44", "COPD"), ("E11", "Type 2 Diabetes"),
        ("I10", "Hypertension"), ("C34", "Lung Cancer"), ("G47", "Sleep Apnea"),
        ("M54", "Back Pain"), ("K80", "Cholelithiasis"), ("N18", "Chronic Kidney Disease"),
        ("F32", "Major Depression")
    ]
    procedures = [
        ("36.06", "Coronary Angioplasty"), ("81.51", "Total Knee Replacement"),
        ("88.91", "CT Scan of Head"), ("00.15", "Laparoscopic Cholecystectomy"),
        ("37.22", "Coronary Artery Bypass"), ("86.22", "Bone Marrow Biopsy")
    ]
    genders = ["M", "F"]
    ethnicities = ["Caucasian", "African American", "Hispanic", "Asian", "Other"]
    insurance = ["Medicare", "Medicaid", "Private", "Self-Pay", "VA"]
    admission_types = ["Emergency", "Elective", "Urgent", "Newborn"]
    discharge_dispositions = ["Home", "Home Health Care", "Skilled Nursing", "Rehab", "Expired", "Transfer"]
    risk_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    now = datetime.now()

    rows = []
    for i in range(n):
        days_back = random.randint(0, 365)
        adm_date = now - timedelta(days=days_back, hours=random.randint(0, 23))
        los = random.randint(1, 21)
        disc_date = adm_date + timedelta(days=los)
        diag_code, diag_desc = random.choice(diagnoses)
        proc_code, proc_desc = random.choice(procedures)
        num_diag = random.randint(1, 8)
        num_proc = random.randint(0, 5)
        num_meds = random.randint(0, 15)
        num_labs = random.randint(1, 30)
        prev_adm = random.randint(0, 6)
        prev_readm = random.randint(0, prev_adm)
        readmitted = random.random() < 0.18
        base_cost = 3000 + los * 800 + num_labs * 50 + num_meds * 30
        cost = round(base_cost * random.uniform(0.85, 1.4), 2)
        risk_score = round(random.random() * 100, 2)
        if risk_score < 25:
            risk = "LOW"
        elif risk_score < 55:
            risk = "MEDIUM"
        elif risk_score < 80:
            risk = "HIGH"
        else:
            risk = "CRITICAL"

        rows.append({
            "id": i + 1,
            "patient_id": f"P{10000 + i}",
            "age": random.randint(18, 92),
            "gender": random.choice(genders),
            "ethnicity": random.choice(ethnicities),
            "admission_id": f"A{20000 + i}",
            "admission_date": adm_date,
            "discharge_date": disc_date,
            "hospital": random.choice(hospitals),
            "department": random.choice(departments),
            "diagnosis_code": diag_code,
            "diagnosis_description": diag_desc,
            "procedure_code": proc_code,
            "procedure_description": proc_desc,
            "num_diagnoses": num_diag,
            "num_procedures": num_proc,
            "num_medications": num_meds,
            "num_lab_tests": num_labs,
            "length_of_stay": los,
            "treatment_cost": cost,
            "previous_admissions": prev_adm,
            "previous_readmissions": prev_readm,
            "readmitted": readmitted,
            "risk_score": risk_score,
            "risk_level": risk,
            "insurance_type": random.choice(insurance),
            "admission_type": random.choice(admission_types),
            "discharge_disposition": random.choice(discharge_dispositions),
            "created_at": adm_date,
            "updated_at": now
        })
    return rows


def _generate_mock_models() -> List[Dict[str, Any]]:
    return [
        {
            "id": 1,
            "model_name": "readmission_risk_v1",
            "model_version": "1.2.0",
            "model_type": "XGBoost Classifier",
            "parameters": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
            "metrics": {"accuracy": 0.87, "precision": 0.82, "recall": 0.71, "f1": 0.76, "roc_auc": 0.92},
            "feature_importance": [
                {"feature": "previous_readmissions", "importance": 0.23},
                {"feature": "num_diagnoses", "importance": 0.18},
                {"feature": "length_of_stay", "importance": 0.15},
                {"feature": "age", "importance": 0.12},
                {"feature": "num_medications", "importance": 0.09}
            ],
            "training_timestamp": datetime.now() - timedelta(days=7),
            "training_rows": 8500,
            "file_path": "/data/models/readmission_risk_v1.joblib",
            "is_active": True,
            "created_at": datetime.now() - timedelta(days=7)
        },
        {
            "id": 2,
            "model_name": "cost_forecast_v1",
            "model_version": "0.9.1",
            "model_type": "Random Forest Regressor",
            "parameters": {"n_estimators": 150, "max_depth": 10},
            "metrics": {"r2": 0.82, "mae": 1280.5, "rmse": 2105.3},
            "feature_importance": [
                {"feature": "length_of_stay", "importance": 0.41},
                {"feature": "num_lab_tests", "importance": 0.22},
                {"feature": "department", "importance": 0.14}
            ],
            "training_timestamp": datetime.now() - timedelta(days=14),
            "training_rows": 7200,
            "file_path": "/data/models/cost_forecast_v1.joblib",
            "is_active": True,
            "created_at": datetime.now() - timedelta(days=14)
        }
    ]


def _generate_mock_insights(patients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    df = pd.DataFrame(patients)
    insights = []
    titles_findings = [
        (
            "Cardiology Readmission Rate Rising",
            "Cardiology department readmission rate is 24.3%, up 6.2pp quarter-over-quarter, driven by post-MI patients with diabetes comorbidity.",
            "Readmission rate moved from 18.1% to 24.3% in Cardiology.",
            ["Increased diabetic comorbidity index", "Reduced follow-up appointment adherence", "Medication non-compliance"],
            "MEDIUM"
        ),
        (
            "HIGH Treatment Cost Anomaly in Oncology",
            "Oncology average cost per admission spiked 31% above baseline ($18,400 vs $14,050) during the last 30 days, concentrated in 2 physician groups.",
            "Oncology cost trend: +31% month-over-month, +$4,350 avg per case.",
            ["New chemotherapy agent usage", "Increased imaging frequency", "Longer inpatient stays"],
            "HIGH"
        ),
        (
            "Weekend ER-to-Admission Bottleneck Detected",
            "Weekend ER admissions show 2.1x higher length-of-stay vs weekday, consistent pattern over 6 weeks, affecting 180+ patients.",
            "Weekend avg LOS 6.8 days vs weekday 3.2 days (ratio 2.1x).",
            ["Reduced weekend staffing ratios", "Delayed specialist consults", "Diagnostic imaging backlog"],
            "HIGH"
        ),
        (
            "CRITICAL Risk Concentration in COPD Patients",
            "112 COPD patients (22.4% of cohort) are classified as HIGH/CRITICAL risk, with average risk score of 82.3 and 38% readmission rate.",
            "COPD subgroup HIGH risk: 22.4% vs all-patients 14.1% baseline.",
            ["Smoking status persistence", "Inhaler technique non-compliance", "Limited pulmonary rehab access"],
            "CRITICAL"
        ),
        (
            "Length of Stay Disparity by Insurance",
            "Self-Pay patients have 1.8 days shorter average LOS than Medicare patients (4.1 vs 5.9 days), controlling for DRG severity.",
            "Self-Pay avg LOS: 4.1 days, Medicare avg LOS: 5.9 days.",
            ["Earlier discharge pressure", "Reduced elective procedure access", "Post-acute care placement delays"],
            "MEDIUM"
        )
    ]
    for i, (title, finding, what_changed, drivers, severity) in enumerate(titles_findings):
        confidence = round(random.uniform(0.78, 0.97), 2)
        insights.append({
            "id": i + 1,
            "insight_type": random.choice(["READMISSION_TREND", "COST_ANOMALY", "DEPARTMENT_PERFORMANCE", "RISK_CONCENTRATION", "LOS_DISPARITY"]),
            "title": title,
            "finding": finding,
            "what_changed": what_changed,
            "evidence": {"p_value": round(random.uniform(0.001, 0.045), 4), "sample_size": random.randint(80, 600), "effect_size": round(random.uniform(0.2, 0.8), 2)},
            "possible_drivers": drivers,
            "recommended_investigation": "Cross-reference with case managers, validate against chart-review sample, and schedule root-cause-analysis meeting.",
            "confidence": confidence,
            "severity": severity,
            "created_at": datetime.now() - timedelta(days=random.randint(0, 12), hours=random.randint(0, 23))
        })
    return insights


def _generate_mock_anomalies(patients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    df = pd.DataFrame(patients)
    departments = df["department"].unique().tolist() if len(df) else ["Cardiology"]
    anomalies = []
    types_scores = [
        ("COST_OUTLIER", 0.87, "HIGH"),
        ("READMISSION_SPIKE", 0.74, "MEDIUM"),
        ("LOS_ANOMALY", 0.91, "HIGH"),
        ("VOLUME_SURGE", 0.68, "MEDIUM"),
        ("MORTALITY_SHIFT", 0.82, "CRITICAL"),
        ("READMISSION_SPIKE", 0.55, "LOW")
    ]
    for i, (atype, score, sev) in enumerate(types_scores):
        dept = random.choice(departments)
        exp = round(random.uniform(5000, 15000), 2)
        actual = round(exp * random.choice([0.4, 0.55, 1.5, 1.8, 2.2]), 2)
        anomalies.append({
            "id": i + 1,
            "anomaly_type": atype,
            "anomaly_score": score,
            "severity": sev,
            "affected_department": dept,
            "affected_date": datetime.now() - timedelta(days=random.randint(1, 21)),
            "affected_metric": random.choice(["treatment_cost", "readmission_rate", "length_of_stay", "admission_volume", "mortality_rate"]),
            "expected_value": exp,
            "actual_value": actual,
            "supporting_data": {"z_score": round(random.uniform(2.1, 4.8), 2), "window": "7-day rolling"},
            "description": f"{atype} detected in {dept}: expected ~${exp:,.0f}, observed ${actual:,.0f}.",
            "created_at": datetime.now() - timedelta(days=random.randint(0, 10))
        })
    return anomalies


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _MOCK_PATIENTS, _MOCK_INSIGHTS, _MOCK_ANOMALIES, _MOCK_MODELS
    _MOCK_PATIENTS = _generate_mock_patients(500)
    _MOCK_MODELS = _generate_mock_models()
    _MOCK_INSIGHTS = _generate_mock_insights(_MOCK_PATIENTS)
    _MOCK_ANOMALIES = _generate_mock_anomalies(_MOCK_PATIENTS)
    yield


app = FastAPI(
    title="Healthcare Analytics Platform",
    description="AI-Powered Healthcare Decision Intelligence Dashboard",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _apply_filters(df: pd.DataFrame, filters: Optional[DashboardFilters]) -> pd.DataFrame:
    if not filters:
        return df
    d = df.copy()
    if filters.start_date and "admission_date" in d.columns:
        d = d[d["admission_date"] >= pd.to_datetime(filters.start_date)]
    if filters.end_date and "admission_date" in d.columns:
        d = d[d["admission_date"] <= pd.to_datetime(filters.end_date)]
    if filters.hospital and "hospital" in d.columns:
        d = d[d["hospital"].isin(filters.hospital)]
    if filters.department and "department" in d.columns:
        d = d[d["department"].isin(filters.department)]
    if filters.risk_level and "risk_level" in d.columns:
        d = d[d["risk_level"].isin(filters.risk_level)]
    return d


def _compute_kpis(df: pd.DataFrame) -> KPIsResponse:
    if len(df) == 0:
        return KPIsResponse(
            total_patients=0, total_admissions=0, readmission_rate=0.0,
            average_length_of_stay=0.0, high_risk_patient_count=0,
            average_treatment_cost=0.0, anomaly_count=len(_MOCK_ANOMALIES),
            total_departments=0, total_diagnoses=0
        )
    total_patients = int(df["patient_id"].nunique())
    total_admissions = int(len(df))
    readmission_rate = round(float(df["readmitted"].mean() * 100), 2) if "readmitted" in df.columns else 0.0
    avg_los = round(float(df["length_of_stay"].mean()), 2) if "length_of_stay" in df.columns else 0.0
    high_risk = int(((df["risk_level"] == "HIGH") | (df["risk_level"] == "CRITICAL")).sum()) if "risk_level" in df.columns else 0
    avg_cost = round(float(df["treatment_cost"].mean()), 2) if "treatment_cost" in df.columns else 0.0
    total_depts = int(df["department"].nunique()) if "department" in df.columns else 0
    total_diag = int(df["diagnosis_code"].nunique()) if "diagnosis_code" in df.columns else 0
    return KPIsResponse(
        total_patients=total_patients,
        total_admissions=total_admissions,
        readmission_rate=readmission_rate,
        average_length_of_stay=avg_los,
        high_risk_patient_count=high_risk,
        average_treatment_cost=avg_cost,
        anomaly_count=len(_MOCK_ANOMALIES),
        total_departments=total_depts,
        total_diagnoses=total_diag
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat(), "patients_loaded": len(_MOCK_PATIENTS)}


@app.get("/api/dashboard/summary", response_model=DashboardDataResponse)
def get_dashboard_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    hospital: Optional[List[str]] = Query(default=None),
    department: Optional[List[str]] = Query(default=None),
    risk_level: Optional[List[str]] = Query(default=None),
):
    filters = DashboardFilters(
        start_date=pd.to_datetime(start_date) if start_date else None,
        end_date=pd.to_datetime(end_date) if end_date else None,
        hospital=hospital, department=department, risk_level=risk_level
    )
    df = _apply_filters(pd.DataFrame(_MOCK_PATIENTS), filters)
    kpis = _compute_kpis(df)
    if len(df) == 0:
        df = pd.DataFrame(_MOCK_PATIENTS)

    d = df.copy()
    d["admission_month"] = pd.to_datetime(d["admission_date"]).dt.to_period("M").dt.to_timestamp()
    adm_time = d.groupby("admission_month").size().reset_index(name="count")
    admissions_over_time = ChartDataResponse(
        chart_type="line",
        title="Admissions Over Time",
        data=[{"date": r["admission_month"].isoformat(), "admissions": int(r["count"])} for _, r in adm_time.iterrows()]
    )

    readm = d.groupby("admission_month").agg(
        total=("readmitted", "size"), readmitted=("readmitted", "sum")
    ).reset_index()
    readm["rate"] = round((readm["readmitted"] / readm["total"]) * 100, 2)
    readmission_trends = ChartDataResponse(
        chart_type="bar",
        title="Readmission Rate Trend (%)",
        data=[{"date": r["admission_month"].isoformat(), "rate": float(r["rate"])} for _, r in readm.iterrows()]
    )

    dept = d.groupby("department").agg(
        admissions=("patient_id", "size"),
        avg_los=("length_of_stay", "mean"),
        avg_cost=("treatment_cost", "mean"),
        readmission_rate=("readmitted", lambda x: round(x.mean() * 100, 2))
    ).reset_index()
    department_performance = ChartDataResponse(
        chart_type="table",
        title="Department Performance",
        data=[
            {
                "department": r["department"],
                "admissions": int(r["admissions"]),
                "avg_los": round(float(r["avg_los"]), 2),
                "avg_cost": round(float(r["avg_cost"]), 2),
                "readmission_rate": float(r["readmission_rate"])
            }
            for _, r in dept.iterrows()
        ]
    )

    diag = d["diagnosis_description"].value_counts().head(10).reset_index()
    diag.columns = ["diagnosis", "count"]
    diagnosis_distribution = ChartDataResponse(
        chart_type="pie",
        title="Top 10 Diagnoses",
        data=[{"diagnosis": r["diagnosis"], "count": int(r["count"])} for _, r in diag.iterrows()]
    )

    los_bins = [0, 2, 5, 8, 14, 21, 999]
    los_labels = ["0-2", "3-5", "6-8", "9-14", "15-21", "22+"]
    d["los_bucket"] = pd.cut(d["length_of_stay"], bins=los_bins, labels=los_labels, right=False)
    los_hist = d["los_bucket"].value_counts().sort_index().reset_index()
    los_hist.columns = ["bucket", "count"]
    los_distribution = ChartDataResponse(
        chart_type="bar",
        title="Length of Stay Distribution (days)",
        data=[{"bucket": str(r["bucket"]), "count": int(r["count"])} for _, r in los_hist.iterrows()]
    )

    risk = d["risk_level"].value_counts().reset_index()
    risk.columns = ["risk_level", "count"]
    risk_distribution = ChartDataResponse(
        chart_type="bar",
        title="Patient Risk Distribution",
        data=[{"risk_level": r["risk_level"], "count": int(r["count"])} for _, r in risk.iterrows()]
    )

    cost_time = d.groupby("admission_month")["treatment_cost"].mean().reset_index()
    cost_time.columns = ["date", "avg_cost"]
    cost_trends = ChartDataResponse(
        chart_type="line",
        title="Average Treatment Cost Trend",
        data=[{"date": r["date"].isoformat(), "avg_cost": round(float(r["avg_cost"]), 2)} for _, r in cost_time.iterrows()]
    )

    anomaly_df = pd.DataFrame(_MOCK_ANOMALIES)
    if len(anomaly_df):
        anomaly_df["date"] = pd.to_datetime(anomaly_df["affected_date"]).dt.to_period("W").dt.to_timestamp()
        anom_group = anomaly_df.groupby(["date", "severity"]).size().reset_index(name="count")
        anomaly_data = [
            {"date": r["date"].isoformat(), "severity": r["severity"], "count": int(r["count"])}
            for _, r in anom_group.iterrows()
        ]
    else:
        anomaly_data = []
    anomaly_trends = ChartDataResponse(
        chart_type="bar",
        title="Anomalies by Severity Over Time",
        data=anomaly_data
    )

    return DashboardDataResponse(
        kpis=kpis,
        admissions_over_time=admissions_over_time,
        readmission_trends=readmission_trends,
        department_performance=department_performance,
        diagnosis_distribution=diagnosis_distribution,
        los_distribution=los_distribution,
        risk_distribution=risk_distribution,
        cost_trends=cost_trends,
        anomaly_trends=anomaly_trends,
    )


@app.get("/api/patients", response_model=List[PatientRecordResponse])
def list_patients(
    skip: int = 0, limit: int = 50,
    search: Optional[str] = None,
    risk_level: Optional[str] = None,
    department: Optional[str] = None
):
    data = _MOCK_PATIENTS
    if search:
        s = search.lower()
        data = [p for p in data if s in str(p["patient_id"]).lower() or s in str(p.get("diagnosis_description", "")).lower()]
    if risk_level:
        data = [p for p in data if p["risk_level"] == risk_level]
    if department:
        data = [p for p in data if p["department"] == department]
    page = data[skip: skip + limit]
    return [PatientRecordResponse(**p) for p in page]


@app.get("/api/patients/{patient_id}", response_model=PatientRecordResponse)
def get_patient(patient_id: int):
    for p in _MOCK_PATIENTS:
        if p["id"] == patient_id:
            return PatientRecordResponse(**p)
    raise HTTPException(status_code=404, detail="Patient not found")


@app.get("/api/kpis", response_model=KPIsResponse)
def get_kpis():
    df = pd.DataFrame(_MOCK_PATIENTS)
    return _compute_kpis(df)


@app.get("/api/models", response_model=List[ModelInfoResponse])
def list_models():
    return [ModelInfoResponse(**m) for m in _MOCK_MODELS]


@app.post("/api/predict", response_model=PredictionResponse)
def predict_risk(req: PredictionRequest):
    pd_data = req.patient_data
    age = float(pd_data.get("age", 50))
    num_diag = float(pd_data.get("num_diagnoses", 2))
    prev_readm = float(pd_data.get("previous_readmissions", 0))
    los = float(pd_data.get("length_of_stay", 3))
    base = 0.15 + (age / 92) * 0.25 + (num_diag / 8) * 0.25 + (prev_readm / 6) * 0.2 + (los / 21) * 0.15
    probability = float(np.clip(base, 0.02, 0.98))
    risk_score = round(probability * 100, 2)
    if risk_score < 25:
        level = "LOW"
    elif risk_score < 55:
        level = "MEDIUM"
    elif risk_score < 80:
        level = "HIGH"
    else:
        level = "CRITICAL"
    top = [
        {"feature": "previous_readmissions", "value": prev_readm, "contribution": round(prev_readm / 6 * 0.20, 3)},
        {"feature": "num_diagnoses", "value": num_diag, "contribution": round(num_diag / 8 * 0.25, 3)},
        {"feature": "age", "value": age, "contribution": round(age / 92 * 0.25, 3)},
        {"feature": "length_of_stay", "value": los, "contribution": round(los / 21 * 0.15, 3)}
    ]
    return PredictionResponse(
        risk_score=risk_score,
        risk_level=level,
        probability=round(probability, 4),
        top_features=top,
        model_name="readmission_risk_v1",
        model_version="1.2.0",
        disclaimer="This is a demo prediction based on synthetic coefficients for illustration purposes only. Not a substitute for clinical judgment."
    )


@app.get("/api/anomalies", response_model=List[AnomalyRecordResponse])
def list_anomalies(severity: Optional[str] = None):
    data = _MOCK_ANOMALIES
    if severity:
        data = [a for a in data if a["severity"] == severity]
    return [AnomalyRecordResponse(**a) for a in data]


@app.get("/api/insights", response_model=List[AIInsightResponse])
def list_insights(severity: Optional[str] = None):
    data = _MOCK_INSIGHTS
    if severity:
        data = [i for i in data if i["severity"] == severity]
    return [AIInsightResponse(**i) for i in data]


@app.get("/api/data-summary", response_model=DataSummaryResponse)
def data_summary():
    df = pd.DataFrame(_MOCK_PATIENTS)
    if len(df) == 0:
        return DataSummaryResponse(
            total_rows=0, total_columns=0,
            hospitals=[], departments=[], diagnoses=[],
            risk_levels=[], insurance_types=[], admission_types=[],
            column_types={}, numeric_summary={}
        )
    dr = list(df["admission_date"].dropna())
    date_range = {
        "min": min(dr).isoformat() if dr else None,
        "max": max(dr).isoformat() if dr else None
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_summary = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s):
            numeric_summary[col] = {
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4)
            }
    col_types = {c: str(t) for c, t in df.dtypes.items()}
    return DataSummaryResponse(
        total_rows=int(len(df)),
        total_columns=int(len(df.columns)),
        date_range=date_range,
        hospitals=sorted(df["hospital"].dropna().unique().tolist()),
        departments=sorted(df["department"].dropna().unique().tolist()),
        diagnoses=sorted(df["diagnosis_description"].dropna().unique().tolist()),
        risk_levels=sorted(df["risk_level"].dropna().unique().tolist()),
        insurance_types=sorted(df["insurance_type"].dropna().unique().tolist()),
        admission_types=sorted(df["admission_type"].dropna().unique().tolist()),
        column_types=col_types,
        numeric_summary=numeric_summary
    )


@app.post("/api/what-if", response_model=WhatIfResponse)
def what_if_simulation(req: WhatIfRequest):
    df = pd.DataFrame(_MOCK_PATIENTS)
    kpi = _compute_kpis(df)
    current = {
        "readmission_rate": kpi.readmission_rate,
        "average_length_of_stay": kpi.average_length_of_stay,
        "average_treatment_cost": kpi.average_treatment_cost,
        "high_risk_patient_count": kpi.high_risk_patient_count,
        "total_admissions": kpi.total_admissions
    }
    factors = {"readmission_rate": 1.0, "average_length_of_stay": 1.0, "average_treatment_cost": 1.0, "high_risk_patient_count": 1.0}
    for p in req.parameters:
        name = p.name.lower()
        if "los" in name or "length" in name:
            try:
                factors["average_length_of_stay"] = float(p.new_value) / max(float(p.current_value), 0.001)
            except Exception:
                pass
        elif "readmission" in name:
            try:
                factors["readmission_rate"] = float(p.new_value) / max(float(p.current_value), 0.001)
            except Exception:
                pass
        elif "cost" in name:
            try:
                factors["average_treatment_cost"] = float(p.new_value) / max(float(p.current_value), 0.001)
            except Exception:
                pass
        elif "risk" in name or "high_risk" in name:
            try:
                factors["high_risk_patient_count"] = float(p.new_value) / max(float(p.current_value), 0.001)
            except Exception:
                pass
    simulated = {k: round(v * factors.get(k, 1.0), 2) for k, v in current.items()}
    params_modified = [{"name": p.name, "current_value": p.current_value, "new_value": p.new_value, "description": p.description} for p in req.parameters]
    return WhatIfResponse(
        id=random.randint(100, 9999),
        scenario_name=req.scenario_name,
        description=req.description,
        parameters_modified=params_modified,
        current_metrics=current,
        simulated_metrics=simulated,
        created_at=datetime.now(),
        disclaimer="Simulation results are synthetic and use linear factor modeling. Validate hypotheses with a production-grade simulation engine before operational decisions."
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True
    )
