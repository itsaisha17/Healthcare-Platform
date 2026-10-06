import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import datetime
import io
import os


class DataIngestion:
    @staticmethod
    def read_csv(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    @staticmethod
    def read_excel(file_path: str) -> pd.DataFrame:
        return pd.read_excel(file_path)

    @staticmethod
    def read_file(content: bytes, filename: str) -> pd.DataFrame:
        if filename.endswith('.csv'):
            return pd.read_csv(io.BytesIO(content))
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported file format: {filename}")


class DataPreprocessor:
    CATEGORICAL_COLUMNS = [
        'gender', 'ethnicity', 'hospital', 'department', 'diagnosis_code',
        'diagnosis_description', 'procedure_code', 'procedure_description',
        'insurance_type', 'admission_type', 'discharge_disposition'
    ]

    NUMERIC_COLUMNS = [
        'age', 'num_diagnoses', 'num_procedures', 'num_medications',
        'num_lab_tests', 'length_of_stay', 'treatment_cost',
        'previous_admissions', 'previous_readmissions'
    ]

    BOOLEAN_COLUMNS = ['readmitted']

    DATE_COLUMNS = ['admission_date', 'discharge_date']

    COLUMN_MAPPINGS = {
        'patient_id': ['patient_id', 'patientid', 'pt_id', 'id_patient'],
        'age': ['age', 'patient_age', 'years_old'],
        'gender': ['gender', 'sex', 'patient_gender'],
        'ethnicity': ['ethnicity', 'race', 'ethnic_group'],
        'admission_id': ['admission_id', 'admissionid', 'adm_id', 'encounter_id'],
        'admission_date': ['admission_date', 'adm_date', 'date_of_admission', 'admit_date'],
        'discharge_date': ['discharge_date', 'disch_date', 'date_of_discharge'],
        'hospital': ['hospital', 'hospital_name', 'facility', 'facility_name'],
        'department': ['department', 'dept', 'ward', 'unit', 'service'],
        'diagnosis_code': ['diagnosis_code', 'dx_code', 'icd_code', 'primary_diagnosis_code'],
        'diagnosis_description': ['diagnosis_description', 'diagnosis', 'dx_desc', 'primary_diagnosis'],
        'procedure_code': ['procedure_code', 'proc_code', 'cpt_code', 'surgery_code'],
        'procedure_description': ['procedure_description', 'procedure', 'proc_desc', 'surgery'],
        'num_diagnoses': ['num_diagnoses', 'number_diagnoses', 'n_diagnoses', 'diagnosis_count'],
        'num_procedures': ['num_procedures', 'number_procedures', 'n_procedures', 'procedure_count'],
        'num_medications': ['num_medications', 'number_medications', 'n_medications', 'medication_count'],
        'num_lab_tests': ['num_lab_tests', 'number_lab_tests', 'n_lab_tests', 'lab_test_count', 'num_lab_procedures'],
        'length_of_stay': ['length_of_stay', 'los', 'stay_length', 'days_in_hospital', 'time_in_hospital'],
        'treatment_cost': ['treatment_cost', 'cost', 'total_cost', 'charges', 'total_charges', 'medical_cost'],
        'previous_admissions': ['previous_admissions', 'prior_admissions', 'num_previous_admissions', 'admissions_prior'],
        'previous_readmissions': ['previous_readmissions', 'prior_readmissions', 'num_previous_readmissions', 'readmissions_prior'],
        'readmitted': ['readmitted', 'readmission', 'is_readmitted', 'was_readmitted'],
        'insurance_type': ['insurance_type', 'insurance', 'payer', 'payer_type', 'coverage_type'],
        'admission_type': ['admission_type', 'adm_type', 'type_of_admission', 'visit_type'],
        'discharge_disposition': ['discharge_disposition', 'discharge_type', 'disposition', 'discharge_location'],
    }

    @classmethod
    def standardize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        reverse_map = {}
        for canonical, variants in cls.COLUMN_MAPPINGS.items():
            for v in variants:
                reverse_map[v.lower()] = canonical
        df.columns = [reverse_map.get(c, c) for c in df.columns]
        return df

    @classmethod
    def convert_data_types(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in cls.DATE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        for col in cls.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        for col in cls.BOOLEAN_COLUMNS:
            if col in df.columns:
                df[col] = df[col].map(
                    lambda x: True if str(x).lower() in ('yes', 'true', '1', 'y')
                    else (False if str(x).lower() in ('no', 'false', '0', 'n') else x)
                )
                df[col] = df[col].astype(bool) if df[col].notna().all() else df[col]
        return df

    @classmethod
    def handle_missing_values(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        df = df.copy()
        missing_info = {}
        for col in df.columns:
            pct_missing = df[col].isna().sum() / len(df) * 100
            if pct_missing > 0:
                missing_info[col] = {
                    'count': int(df[col].isna().sum()),
                    'percentage': round(pct_missing, 2)
                }
            if col in cls.CATEGORICAL_COLUMNS:
                if col in df.columns and df[col].isna().any():
                    df[col] = df[col].fillna('Unknown')
            elif col in cls.NUMERIC_COLUMNS:
                if col in df.columns and df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median() if len(df[col].dropna()) > 0 else 0)
            elif col in cls.BOOLEAN_COLUMNS:
                if col in df.columns and df[col].isna().any():
                    df[col] = df[col].fillna(False)
        return df, missing_info

    @classmethod
    def remove_duplicates(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        initial = len(df)
        df = df.drop_duplicates()
        removed = initial - len(df)
        return df, removed

    @classmethod
    def validate_ranges(cls, df: pd.DataFrame) -> List[Dict]:
        issues = []
        if 'age' in df.columns:
            invalid_age = df[(df['age'] < 0) | (df['age'] > 120)]
            if len(invalid_age) > 0:
                issues.append({'column': 'age', 'count': len(invalid_age), 'message': 'Age outside 0-120 range'})
        if 'length_of_stay' in df.columns:
            invalid_los = df[df['length_of_stay'] < 0]
            if len(invalid_los) > 0:
                issues.append({'column': 'length_of_stay', 'count': len(invalid_los), 'message': 'Negative length of stay'})
        if 'treatment_cost' in df.columns:
            invalid_cost = df[df['treatment_cost'] < 0]
            if len(invalid_cost) > 0:
                issues.append({'column': 'treatment_cost', 'count': len(invalid_cost), 'message': 'Negative treatment cost'})
        return issues

    @classmethod
    def engineer_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'admission_date' in df.columns and 'discharge_date' in df.columns:
            mask = df['length_of_stay'].isna() | (df['length_of_stay'] == 0)
            df.loc[mask, 'length_of_stay'] = (
                (df.loc[mask, 'discharge_date'] - df.loc[mask, 'admission_date']).dt.days
            )
            df['length_of_stay'] = df['length_of_stay'].fillna(0).clip(lower=0).astype(int)
        if 'admission_date' in df.columns:
            df['admission_month'] = df['admission_date'].dt.month
            df['admission_year'] = df['admission_date'].dt.year
            df['admission_quarter'] = df['admission_date'].dt.quarter
            df['is_weekend'] = df['admission_date'].dt.dayofweek >= 5
        if 'previous_admissions' in df.columns:
            df['has_prior_admission'] = df['previous_admissions'] > 0
        if 'age' in df.columns:
            bins = [0, 18, 45, 65, 80, 120]
            labels = ['0-18', '19-45', '46-65', '66-80', '80+']
            df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)
            df['age_group'] = df['age_group'].astype(str).replace('nan', 'Unknown')
        return df

    @classmethod
    def full_pipeline(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = cls.standardize_columns(df)
        df = cls.convert_data_types(df)
        df, missing_info = cls.handle_missing_values(df)
        df, duplicates_removed = cls.remove_duplicates(df)
        range_issues = cls.validate_ranges(df)
        df = cls.engineer_features(df)
        report = {
            'missing_info': missing_info,
            'duplicates_removed': duplicates_removed,
            'range_issues': range_issues,
            'columns_after_processing': list(df.columns),
            'rows_after_processing': len(df)
        }
        return df, report


class DataQualityChecker:
    @staticmethod
    def generate_report(df: pd.DataFrame, dataset_name: str = "dataset",
                        preprocess_report: Dict = None) -> Dict[str, Any]:
        total_rows = len(df)
        total_columns = len(df.columns)
        missing_total = df.isna().sum().sum()
        pct_missing = (missing_total / (total_rows * total_columns) * 100) if total_rows > 0 else 0.0
        duplicate_count = int(df.duplicated().sum())
        invalid_count = 0
        invalid_details = {}
        if 'age' in df.columns:
            inv = len(df[(df['age'] < 0) | (df['age'] > 120) | df['age'].isna()])
            if inv > 0:
                invalid_details['age'] = inv
                invalid_count += inv
        if 'length_of_stay' in df.columns:
            inv = len(df[df['length_of_stay'] < 0])
            if inv > 0:
                invalid_details['length_of_stay'] = inv
                invalid_count += inv
        if 'treatment_cost' in df.columns:
            inv = len(df[df['treatment_cost'] < 0])
            if inv > 0:
                invalid_details['treatment_cost'] = inv
                invalid_count += inv
        missing_columns = {}
        for col in df.columns:
            mc = int(df[col].isna().sum())
            if mc > 0:
                missing_columns[col] = {
                    'count': mc,
                    'percentage': round(mc / total_rows * 100, 2)
                }
        duplicate_details = {}
        if duplicate_count > 0:
            dup_mask = df.duplicated(keep=False)
            duplicate_details['sample_indices'] = df[dup_mask].head(10).index.tolist()
        column_quality = {}
        for col in df.columns:
            pct = df[col].isna().sum() / total_rows * 100 if total_rows > 0 else 0
            column_quality[col] = {
                'type': str(df[col].dtype),
                'missing_count': int(df[col].isna().sum()),
                'missing_percentage': round(pct, 2),
                'unique_count': int(df[col].nunique(dropna=True))
            }
        return {
            'dataset_name': dataset_name,
            'total_rows': total_rows,
            'total_columns': total_columns,
            'missing_value_percentage': round(pct_missing, 2),
            'duplicate_count': duplicate_count,
            'invalid_value_count': invalid_count,
            'missing_columns': missing_columns,
            'duplicate_details': duplicate_details,
            'invalid_details': invalid_details,
            'column_quality': column_quality
        }
