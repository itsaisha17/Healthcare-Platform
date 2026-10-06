import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings('ignore')


class AnomalyDetector:
    ANOMALY_TYPES = {
        'ADMISSION_SPIKE': 'Sudden increase in admissions',
        'READMISSION_SPIKE': 'Unusual readmission spike',
        'ABNORMAL_COST': 'Abnormal treatment cost',
        'WORKLOAD_ANOMALY': 'Unusual department workload',
        'EXTREME_LOS': 'Extreme length-of-stay pattern',
        'DIAGNOSIS_FREQ': 'Unexpected diagnosis frequency',
        'GENERIC_POINT': 'Multivariate outlier in patient records'
    }

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.numeric_cols = [
            'age', 'num_diagnoses', 'num_procedures', 'num_medications',
            'num_lab_tests', 'length_of_stay', 'treatment_cost',
            'previous_admissions', 'previous_readmissions'
        ]

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        available = [c for c in self.numeric_cols if c in df.columns]
        if not available:
            raise ValueError("No numeric columns available for anomaly detection")
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X = df[available].copy()
        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)
        return X_scaled, available

    def run_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        X, cols = self._prepare_features(df)
        iso = IsolationForest(
            contamination=self.contamination, n_estimators=200,
            random_state=42, n_jobs=-1
        )
        preds = iso.fit_predict(X)
        scores = iso.score_samples(X)
        result = df.copy()
        result['if_anomaly'] = preds == -1
        result['if_score'] = scores
        return result

    def run_lof(self, df: pd.DataFrame) -> pd.DataFrame:
        X, cols = self._prepare_features(df)
        lof = LocalOutlierFactor(
            n_neighbors=20, contamination=self.contamination,
            metric='euclidean', novelty=False
        )
        preds = lof.fit_predict(X)
        scores = lof.negative_outlier_factor_
        result = df.copy()
        result['lof_anomaly'] = preds == -1
        result['lof_score'] = scores
        return result

    def run_statistical_threshold(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result['stat_anomaly'] = False
        result['stat_score'] = 0.0
        for col in self.numeric_cols:
            if col in df.columns and df[col].nunique() > 2:
                data = df[col].dropna()
                if len(data) < 10:
                    continue
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                col_outlier = (df[col] < lower) | (df[col] > upper)
                result['stat_anomaly'] = result['stat_anomaly'] | col_outlier.fillna(False)
                z_scores = np.abs((df[col] - data.mean()) / (data.std() if data.std() > 0 else 1))
                result['stat_score'] = np.maximum(result['stat_score'], z_scores.fillna(0))
        return result

    def run_combined(self, df: pd.DataFrame,
                     methods: List[str] = None) -> pd.DataFrame:
        if methods is None:
            methods = ['isolation_forest', 'lof', 'statistical']
        result = df.copy()
        vote_cols = []
        if 'isolation_forest' in methods:
            res_iso = self.run_isolation_forest(df)
            result['if_anomaly'] = res_iso['if_anomaly'].values
            result['if_score'] = res_iso['if_score'].values
            vote_cols.append('if_anomaly')
        if 'lof' in methods:
            res_lof = self.run_lof(df)
            result['lof_anomaly'] = res_lof['lof_anomaly'].values
            result['lof_score'] = res_lof['lof_score'].values
            vote_cols.append('lof_anomaly')
        if 'statistical' in methods:
            res_stat = self.run_statistical_threshold(df)
            result['stat_anomaly'] = res_stat['stat_anomaly'].values
            result['stat_score'] = res_stat['stat_score'].values
            vote_cols.append('stat_anomaly')
        if vote_cols:
            result['anomaly_votes'] = result[vote_cols].astype(int).sum(axis=1)
            threshold = max(1, len(vote_cols) // 2)
            result['is_anomaly'] = result['anomaly_votes'] >= threshold
        else:
            result['is_anomaly'] = False
            result['anomaly_votes'] = 0
        return result

    def _classify_anomaly_type(self, row: pd.Series) -> str:
        try:
            if 'length_of_stay' in row and pd.notna(row['length_of_stay']):
                if row['length_of_stay'] > 30:
                    return 'EXTREME_LOS'
            if 'treatment_cost' in row and pd.notna(row['treatment_cost']):
                if row['treatment_cost'] > 100000:
                    return 'ABNORMAL_COST'
            if 'previous_readmissions' in row and pd.notna(row['previous_readmissions']):
                if row['previous_readmissions'] >= 3:
                    return 'READMISSION_SPIKE'
            if 'num_procedures' in row and pd.notna(row['num_procedures']):
                if row['num_procedures'] > 15:
                    return 'WORKLOAD_ANOMALY'
            return 'GENERIC_POINT'
        except Exception:
            return 'GENERIC_POINT'

    def _get_severity(self, votes: int, total_methods: int) -> str:
        ratio = votes / max(total_methods, 1)
        if ratio >= 0.9:
            return 'CRITICAL'
        elif ratio >= 0.66:
            return 'HIGH'
        elif ratio >= 0.33:
            return 'MEDIUM'
        else:
            return 'LOW'

    def generate_anomaly_records(self, df_with_anomalies: pd.DataFrame,
                                 methods_used: List[str] = None) -> List[Dict[str, Any]]:
        if methods_used is None:
            methods_used = ['isolation_forest', 'lof', 'statistical']
        total_methods = len(methods_used)
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly']].copy()
        if len(anomalies) == 0:
            return []
        records = []
        for idx, row in anomalies.iterrows():
            anomaly_type = self._classify_anomaly_type(row)
            votes = int(row.get('anomaly_votes', 1))
            severity = self._get_severity(votes, total_methods)
            score = float(row.get('if_score', 0))
            if score == 0:
                score = -float(row.get('lof_score', 1))
            supporting_data = {}
            for col in self.numeric_cols:
                if col in row and pd.notna(row[col]):
                    supporting_data[col] = (float(row[col])
                                            if isinstance(row[col], (np.integer, np.floating))
                                            else row[col])
            record = {
                'anomaly_type': anomaly_type,
                'anomaly_score': round(score, 4),
                'severity': severity,
                'affected_department': row.get('department', None),
                'affected_date': row.get('admission_date', None),
                'affected_metric': anomaly_type,
                'expected_value': None,
                'actual_value': None,
                'supporting_data': supporting_data,
                'description': (f"{self.ANOMALY_TYPES.get(anomaly_type, 'Anomaly detected')} "
                                f"- {votes}/{total_methods} methods flagged. "
                                f"Patient: {row.get('patient_id', 'N/A')}")
            }
            if anomaly_type == 'EXTREME_LOS' and 'length_of_stay' in row:
                record['actual_value'] = float(row['length_of_stay'])
                record['expected_value'] = 5.0
            elif anomaly_type == 'ABNORMAL_COST' and 'treatment_cost' in row:
                record['actual_value'] = float(row['treatment_cost'])
                record['expected_value'] = 15000.0
            records.append(record)
        return records

    def detect_time_series_anomalies(self, df: pd.DataFrame,
                                     run_types: List[str] = None) -> List[Dict[str, Any]]:
        if run_types is None:
            run_types = ['admissions', 'readmissions', 'cost', 'los']
        results = []
        if 'admission_date' not in df.columns:
            return results
        df_ts = df.copy()
        df_ts['admission_date'] = pd.to_datetime(df_ts['admission_date'], errors='coerce')
        df_ts = df_ts.dropna(subset=['admission_date'])
        if len(df_ts) < 10:
            return results
        daily = df_ts.groupby(df_ts['admission_date'].dt.date).agg(
            admissions=('patient_id', 'count'),
            readmissions=('readmitted', lambda x: x.sum() if 'readmitted' in df_ts.columns else 0),
            avg_cost=('treatment_cost', 'mean'),
            avg_los=('length_of_stay', 'mean')
        ).reset_index()
        daily.columns = ['date', 'admissions', 'readmissions', 'avg_cost', 'avg_los']
        daily = daily.sort_values('date')
        metric_configs = [
            ('admissions', 'ADMISSION_SPIKE', 'Admissions'),
            ('readmissions', 'READMISSION_SPIKE', 'Readmissions'),
            ('avg_cost', 'ABNORMAL_COST', 'Average Cost'),
            ('avg_los', 'EXTREME_LOS', 'Average LOS')
        ]
        for metric, anom_type, label in metric_configs:
            if metric not in run_types:
                continue
            data = daily[metric].dropna()
            if len(data) < 7:
                continue
            mean = data.mean()
            std = data.std() if data.std() > 0 else mean * 0.1
            threshold = mean + 2.5 * std
            for _, row in daily[daily[metric] > threshold].iterrows():
                results.append({
                    'anomaly_type': anom_type,
                    'anomaly_score': round(float((row[metric] - mean) / std), 2),
                    'severity': 'HIGH' if (row[metric] - mean) / std > 3 else 'MEDIUM',
                    'affected_department': None,
                    'affected_date': pd.Timestamp(row['date']),
                    'affected_metric': label,
                    'expected_value': round(float(mean), 2),
                    'actual_value': round(float(row[metric]), 2),
                    'supporting_data': {'date': str(row['date']), metric: float(row[metric]),
                                        'mean_7d': round(float(data.tail(7).mean()), 2)},
                    'description': (f"{self.ANOMALY_TYPES.get(anom_type, anom_type)} on {row['date']}: "
                                    f"observed {row[metric]:.1f} vs expected {mean:.1f}")
                })
        return results

    def detect_department_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if 'department' not in df.columns:
            return []
        results = []
        dept_stats = df.groupby('department').agg(
            patients=('patient_id', 'nunique'),
            avg_los=('length_of_stay', 'mean'),
            avg_cost=('treatment_cost', 'mean'),
            readm_rate=('readmitted', 'mean')
        ).reset_index()
        for metric in ['avg_los', 'avg_cost', 'readm_rate']:
            data = dept_stats[metric].dropna()
            if len(data) < 3:
                continue
            mean = data.mean()
            std = data.std() if data.std() > 0 else mean * 0.1
            for _, row in dept_stats.iterrows():
                if pd.notna(row[metric]) and abs(row[metric] - mean) > 2 * std:
                    z = (row[metric] - mean) / std
                    anom_type = 'WORKLOAD_ANOMALY' if z > 0 else 'ABNORMAL_COST'
                    label_map = {'avg_los': 'Avg LOS', 'avg_cost': 'Avg Cost', 'readm_rate': 'Readmission Rate'}
                    results.append({
                        'anomaly_type': anom_type,
                        'anomaly_score': round(float(z), 2),
                        'severity': 'HIGH' if abs(z) > 3 else 'MEDIUM',
                        'affected_department': row['department'],
                        'affected_date': None,
                        'affected_metric': label_map[metric],
                        'expected_value': round(float(mean), 2),
                        'actual_value': round(float(row[metric]), 2),
                        'supporting_data': {metric: float(row[metric]), 'dept_avg': float(mean)},
                        'description': (f"Dept '{row['department']}': {label_map[metric]} = {row[metric]:.2f}, "
                                        f"deviation {z:+.1f}σ from mean {mean:.2f}")
                    })
        return results

    def run_full_detection(self, df: pd.DataFrame,
                           detection_methods: List[str] = None,
                           run_types: List[str] = None) -> List[Dict[str, Any]]:
        if detection_methods is None:
            detection_methods = ['isolation_forest', 'lof', 'statistical']
        point_df = self.run_combined(df, methods=detection_methods)
        point_records = self.generate_anomaly_records(point_df, detection_methods)
        ts_records = self.detect_time_series_anomalies(df, run_types)
        dept_records = self.detect_department_anomalies(df)
        all_records = point_records + ts_records + dept_records
        all_records.sort(key=lambda r: abs(r['anomaly_score']), reverse=True)
        return all_records
