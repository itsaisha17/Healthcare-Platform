from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, date


class PatientRecordBase(BaseModel):
    patient_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    admission_id: Optional[str] = None
    admission_date: Optional[datetime] = None
    discharge_date: Optional[datetime] = None
    hospital: Optional[str] = None
    department: Optional[str] = None
    diagnosis_code: Optional[str] = None
    diagnosis_description: Optional[str] = None
    procedure_code: Optional[str] = None
    procedure_description: Optional[str] = None
    num_diagnoses: Optional[int] = 0
    num_procedures: Optional[int] = 0
    num_medications: Optional[int] = 0
    num_lab_tests: Optional[int] = 0
    length_of_stay: Optional[int] = 0
    treatment_cost: Optional[float] = 0.0
    previous_admissions: Optional[int] = 0
    previous_readmissions: Optional[int] = 0
    readmitted: Optional[bool] = False
    risk_score: Optional[float] = 0.0
    risk_level: Optional[str] = "LOW"
    insurance_type: Optional[str] = None
    admission_type: Optional[str] = None
    discharge_disposition: Optional[str] = None


class PatientRecordCreate(PatientRecordBase):
    pass


class PatientRecordResponse(PatientRecordBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class DataQualityReportResponse(BaseModel):
    id: int
    dataset_name: str
    total_rows: int
    total_columns: int
    missing_value_percentage: float
    duplicate_count: int
    invalid_value_count: int
    missing_columns: Optional[Dict] = None
    duplicate_details: Optional[Dict] = None
    invalid_details: Optional[Dict] = None
    column_quality: Optional[Dict] = None
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetUploadResponse(BaseModel):
    message: str
    dataset_name: str
    rows_imported: int
    data_quality: DataQualityReportResponse


class KPIsResponse(BaseModel):
    total_patients: int
    total_admissions: int
    readmission_rate: float
    average_length_of_stay: float
    high_risk_patient_count: int
    average_treatment_cost: float
    anomaly_count: int
    total_departments: int
    total_diagnoses: int


class ChartDataResponse(BaseModel):
    chart_type: str
    title: str
    data: List[Dict[str, Any]]
    layout: Optional[Dict[str, Any]] = None


class DashboardDataResponse(BaseModel):
    kpis: KPIsResponse
    admissions_over_time: ChartDataResponse
    readmission_trends: ChartDataResponse
    department_performance: ChartDataResponse
    diagnosis_distribution: ChartDataResponse
    los_distribution: ChartDataResponse
    risk_distribution: ChartDataResponse
    cost_trends: ChartDataResponse
    anomaly_trends: ChartDataResponse


class DashboardFilters(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    hospital: Optional[List[str]] = None
    department: Optional[List[str]] = None
    diagnosis: Optional[List[str]] = None
    patient_segment: Optional[List[str]] = None
    risk_level: Optional[List[str]] = None


class ModelInfoResponse(BaseModel):
    id: int
    model_name: str
    model_version: str
    model_type: str
    parameters: Optional[Dict] = None
    metrics: Optional[Dict] = None
    feature_importance: Optional[List[Dict]] = None
    training_timestamp: Optional[datetime] = None
    training_rows: Optional[int] = None
    is_active: bool

    class Config:
        from_attributes = True


class ModelTrainingRequest(BaseModel):
    model_types: List[str] = Field(default=["logistic_regression", "random_forest", "xgboost"])
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = Field(default=42)
    target_column: str = "readmitted"


class PredictionRequest(BaseModel):
    patient_data: Dict[str, Any]


class PredictionResponse(BaseModel):
    risk_score: float
    risk_level: str
    probability: float
    top_features: List[Dict[str, Any]]
    model_name: str
    model_version: str
    disclaimer: str


class BatchPredictionRequest(BaseModel):
    patient_ids: Optional[List[str]] = None
    filters: Optional[DashboardFilters] = None


class AnomalyRecordResponse(BaseModel):
    id: int
    anomaly_type: str
    anomaly_score: float
    severity: str
    affected_department: Optional[str] = None
    affected_date: Optional[datetime] = None
    affected_metric: Optional[str] = None
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    supporting_data: Optional[Dict] = None
    description: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AnomalyDetectionRequest(BaseModel):
    detection_methods: List[str] = Field(default=["isolation_forest", "lof", "statistical"])
    contamination: float = Field(default=0.05, ge=0.01, le=0.2)
    run_types: Optional[List[str]] = None


class RAGDocumentUploadResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    category: str
    status: str
    chunks_count: int
    uploaded_at: datetime


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=4, ge=1, le=10)
    doc_category: Optional[str] = None


class RAGSource(BaseModel):
    filename: str
    page: Optional[int] = None
    chunk_id: int
    relevance_score: float
    content: str


class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[RAGSource]
    confidence: float
    limitations: str
    query: str


class NLQueryRequest(BaseModel):
    query: str
    generate_chart: bool = True


class NLQueryResponse(BaseModel):
    natural_query: str
    generated_sql: Optional[str] = None
    query_result: Optional[Any] = None
    explanation: str
    chart_type: Optional[str] = None
    chart_data: Optional[Dict] = None
    execution_time_ms: int
    success: bool
    error_message: Optional[str] = None


class AIInsightResponse(BaseModel):
    id: int
    insight_type: str
    title: str
    finding: str
    what_changed: str
    evidence: Optional[Dict] = None
    possible_drivers: Optional[List[str]] = None
    recommended_investigation: str
    confidence: float
    severity: str
    created_at: datetime

    class Config:
        from_attributes = True


class DecisionActionResponse(BaseModel):
    id: int
    insight_id: Optional[int] = None
    problem: str
    evidence: Dict
    possible_causes: List[str]
    recommended_actions: List[str]
    expected_impacts: List[Dict]
    impact_type: str
    created_at: datetime

    class Config:
        from_attributes = True


class WhatIfParameter(BaseModel):
    name: str
    current_value: Any
    new_value: Any
    description: Optional[str] = None


class WhatIfRequest(BaseModel):
    scenario_name: str
    description: Optional[str] = None
    parameters: List[WhatIfParameter]


class WhatIfResponse(BaseModel):
    id: int
    scenario_name: str
    description: Optional[str] = None
    parameters_modified: List[Dict]
    current_metrics: Dict
    simulated_metrics: Dict
    comparison_chart: Optional[Dict] = None
    created_at: datetime
    disclaimer: str


class DataSummaryResponse(BaseModel):
    total_rows: int
    total_columns: int
    date_range: Optional[Dict] = None
    hospitals: List[str]
    departments: List[str]
    diagnoses: List[str]
    risk_levels: List[str]
    insurance_types: List[str]
    admission_types: List[str]
    column_types: Dict[str, str]
    numeric_summary: Dict[str, Dict]
