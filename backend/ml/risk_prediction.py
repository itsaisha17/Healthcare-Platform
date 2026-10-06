import pandas as pd
import numpy as np
import os
import json
import joblib
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from backend.config import settings


class RiskPredictionPipeline:
    FEATURE_NUMERIC = [
        'age', 'num_diagnoses', 'num_procedures', 'num_medications',
        'num_lab_tests', 'length_of_stay', 'treatment_cost',
        'previous_admissions', 'previous_readmissions'
    ]

    FEATURE_CATEGORICAL = [
        'gender', 'ethnicity', 'department', 'insurance_type',
        'admission_type', 'discharge_disposition'
    ]

    TARGET_COLUMN = 'readmitted'

    def __init__(self):
        os.makedirs(settings.MODEL_STORAGE_PATH, exist_ok=True)
        self.models: Dict[str, Any] = {}
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []
        self.is_trained = False

    def _get_feature_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        numeric_cols = [c for c in self.FEATURE_NUMERIC if c in df.columns]
        categorical_cols = [c for c in self.FEATURE_CATEGORICAL if c in df.columns]
        return numeric_cols, categorical_cols

    def _build_preprocessor(self, numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

    def _get_model_configs(self, model_types: List[str]) -> Dict[str, Any]:
        configs = {}
        if 'logistic_regression' in model_types:
            configs['logistic_regression'] = LogisticRegression(
                max_iter=1000, C=1.0, class_weight='balanced', random_state=42
            )
        if 'random_forest' in model_types:
            configs['random_forest'] = RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
            )
        if 'xgboost' in model_types and XGBOOST_AVAILABLE:
            configs['xgboost'] = XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                eval_metric='logloss', random_state=42
            )
        return configs

    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: np.ndarray, model_name: str) -> Dict[str, Any]:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics = {
            'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
            'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            'recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            'f1_score': round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            'roc_auc': round(float(roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5), 4),
            'specificity': round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0, 4),
            'sensitivity': round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0, 4),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            },
            'classification_report': classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        }
        return metrics

    def train_models(self, df: pd.DataFrame,
                     model_types: List[str] = None,
                     test_size: float = 0.2,
                     random_state: int = 42) -> Dict[str, Any]:
        if model_types is None:
            model_types = ['logistic_regression', 'random_forest', 'xgboost']
        if self.TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{self.TARGET_COLUMN}' not found in dataset")

        numeric_cols, categorical_cols = self._get_feature_columns(df)
        all_feature_cols = numeric_cols + categorical_cols
        if not all_feature_cols:
            raise ValueError("No feature columns found in dataset")

        df_model = df[all_feature_cols + [self.TARGET_COLUMN]].copy()
        y = df_model[self.TARGET_COLUMN].astype(int).values
        X = df_model[all_feature_cols]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.preprocessor = self._build_preprocessor(numeric_cols, categorical_cols)
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        if hasattr(self.preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
            cat_features = list(self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                                .get_feature_names_out(categorical_cols))
        else:
            cat_features = []
        self.feature_names = numeric_cols + cat_features

        model_configs = self._get_model_configs(model_types)
        results = {'training_timestamp': datetime.now().isoformat(), 'models': {}}

        for model_name, model in model_configs.items():
            model_params = model.get_params()
            model.fit(X_train_processed, y_train)

            y_train_pred = model.predict(X_train_processed)
            y_train_prob = model.predict_proba(X_train_processed)[:, 1]
            y_test_pred = model.predict(X_test_processed)
            y_test_prob = model.predict_proba(X_test_processed)[:, 1]

            train_metrics = self._evaluate_model(y_train, y_train_pred, y_train_prob, model_name)
            test_metrics = self._evaluate_model(y_test, y_test_pred, y_test_prob, model_name)

            feature_importance = self._get_feature_importance(model, model_name)

            self.models[model_name] = {
                'model': model,
                'metrics': test_metrics,
                'train_metrics': train_metrics,
                'feature_importance': feature_importance,
                'params': {k: (v if not isinstance(v, (np.integer, np.floating)) else (int(v) if isinstance(v, np.integer) else float(v)))
                           for k, v in model_params.items()}
            }

            self._save_model(model_name, model, test_metrics, feature_importance, model_params, len(X_train))

            results['models'][model_name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance
            }

        self.is_trained = True
        best_model = max(results['models'].items(),
                         key=lambda x: x[1]['test_metrics'].get('roc_auc', 0))
        results['best_model'] = best_model[0]
        results['training_rows'] = len(X_train)
        results['test_rows'] = len(X_test)
        results['feature_count'] = len(self.feature_names)

        self._save_state()
        return results

    def _get_feature_importance(self, model: Any, model_name: str) -> List[Dict[str, Any]]:
        importance = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            for i in indices[:15]:
                if i < len(self.feature_names):
                    importance.append({
                        'feature': self.feature_names[i],
                        'importance': round(float(importances[i]), 6),
                        'rank': len(importance) + 1
                    })
        elif hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_[0])
            indices = np.argsort(coefs)[::-1]
            for i in indices[:15]:
                if i < len(self.feature_names):
                    importance.append({
                        'feature': self.feature_names[i],
                        'importance': round(float(coefs[i]), 6),
                        'coefficient': round(float(model.coef_[0][i]), 6),
                        'rank': len(importance) + 1
                    })
        return importance

    def _get_shap_explanation(self, model: Any, X_processed: np.ndarray,
                               sample_idx: int = 0, background_size: int = 100) -> List[Dict[str, Any]]:
        if not SHAP_AVAILABLE:
            return []
        try:
            bg_data = X_processed[:background_size] if len(X_processed) > background_size else X_processed
            if hasattr(model, 'predict_proba'):
                explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)[:, 1], bg_data[:min(10, len(bg_data))]
                )
                shap_values = explainer.shap_values(X_processed[sample_idx:sample_idx + 1], nsamples=50)
            else:
                return []
            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values
            sv_flat = sv.flatten() if hasattr(sv, 'flatten') else sv
            indices = np.argsort(np.abs(sv_flat))[::-1]
            contribs = []
            for i in indices[:10]:
                if i < len(self.feature_names):
                    contribs.append({
                        'feature': self.feature_names[i],
                        'shap_value': round(float(sv_flat[i]), 6),
                        'direction': 'positive' if sv_flat[i] > 0 else 'negative'
                    })
            return contribs
        except Exception:
            return []

    def predict(self, patient_data: Dict[str, Any],
                model_name: str = None) -> Dict[str, Any]:
        if not self.is_trained:
            self._load_state()
        if model_name is None:
            model_name = self._get_best_available_model()
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")

        df_input = pd.DataFrame([patient_data])
        numeric_cols, categorical_cols = self._get_feature_columns(df_input)
        all_cols = numeric_cols + categorical_cols
        for col in all_cols:
            if col not in df_input.columns:
                if col in numeric_cols:
                    df_input[col] = 0
                else:
                    df_input[col] = 'Unknown'

        X_processed = self.preprocessor.transform(df_input[all_cols])
        model_info = self.models[model_name]
        model = model_info['model']

        probability = float(model.predict_proba(X_processed)[:, 1][0])
        prediction = int(model.predict(X_processed)[0])

        risk_score = round(probability * 100, 1)
        if risk_score >= 70:
            risk_level = 'HIGH'
        elif risk_score >= 40:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        shap_contribs = self._get_shap_explanation(model, X_processed, 0)
        if not shap_contribs:
            shap_contribs = model_info.get('feature_importance', [])[:10]

        version = model_info.get('version', '1.0.0')
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'probability': round(probability, 4),
            'prediction_class': prediction,
            'top_features': shap_contribs,
            'model_name': model_name,
            'model_version': version,
            'disclaimer': 'Predictions are for demonstration/decision-support purposes only and do not constitute medical advice.'
        }

    def predict_batch(self, df: pd.DataFrame,
                      model_name: str = None,
                      patient_ids: List[str] = None) -> pd.DataFrame:
        if not self.is_trained:
            self._load_state()
        if model_name is None:
            model_name = self._get_best_available_model()

        df_work = df.copy()
        numeric_cols, categorical_cols = self._get_feature_columns(df_work)
        all_cols = numeric_cols + categorical_cols
        for col in all_cols:
            if col not in df_work.columns:
                if col in numeric_cols:
                    df_work[col] = 0
                else:
                    df_work[col] = 'Unknown'

        X = df_work[all_cols]
        if patient_ids is not None and 'patient_id' in df_work.columns:
            mask = df_work['patient_id'].isin(patient_ids)
            if mask.any():
                X = X[mask]
                df_work = df_work[mask]

        X_processed = self.preprocessor.transform(X)
        model = self.models[model_name]['model']

        probs = model.predict_proba(X_processed)[:, 1]
        df_work['risk_score'] = np.round(probs * 100, 1)
        df_work['risk_level'] = pd.cut(
            df_work['risk_score'],
            bins=[-1, 39.9, 69.9, 101],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        df_work['risk_probability'] = np.round(probs, 4)
        df_work['predicted_readmission'] = (probs >= 0.5).astype(int)
        return df_work

    def _get_best_available_model(self) -> str:
        if 'xgboost' in self.models:
            return 'xgboost'
        if 'random_forest' in self.models:
            return 'random_forest'
        if 'logistic_regression' in self.models:
            return 'logistic_regression'
        raise ValueError("No trained models available")

    def _save_model(self, model_name: str, model: Any, metrics: Dict,
                    feature_importance: List, params: Dict, training_rows: int):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"1.0.{int(timestamp[-6:]) % 10000}"
        model_filename = f"{model_name}_v{version}.joblib"
        model_path = os.path.join(settings.MODEL_STORAGE_PATH, model_filename)
        joblib.dump(model, model_path)

        self.models[model_name]['version'] = version
        self.models[model_name]['file_path'] = model_path
        self.models[model_name]['training_timestamp'] = timestamp
        self.models[model_name]['training_rows'] = training_rows

        meta = {
            'model_name': model_name,
            'model_version': version,
            'model_type': model_name,
            'parameters': {k: (int(v) if isinstance(v, np.integer) else
                               float(v) if isinstance(v, np.floating) else v)
                           for k, v in params.items()},
            'metrics': metrics,
            'feature_importance': feature_importance,
            'training_timestamp': timestamp,
            'training_rows': training_rows,
            'file_path': model_path
        }
        meta_path = os.path.join(settings.MODEL_STORAGE_PATH, f"{model_name}_v{version}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    def _save_state(self):
        state = {
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'model_names': list(self.models.keys()),
            'saved_at': datetime.now().isoformat()
        }
        for name, info in self.models.items():
            state[name] = {k: v for k, v in info.items() if k != 'model'}
        state_path = os.path.join(settings.MODEL_STORAGE_PATH, 'pipeline_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, os.path.join(settings.MODEL_STORAGE_PATH, 'preprocessor.joblib'))

    def _load_state(self):
        state_path = os.path.join(settings.MODEL_STORAGE_PATH, 'pipeline_state.json')
        preprocessor_path = os.path.join(settings.MODEL_STORAGE_PATH, 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            self.feature_names = state.get('feature_names', [])
            self.is_trained = state.get('is_trained', False)
            for name in state.get('model_names', []):
                if name in state:
                    meta = state[name]
                    fp = meta.get('file_path')
                    if fp and os.path.exists(fp):
                        self.models[name] = {'model': joblib.load(fp)}
                        for k, v in meta.items():
                            self.models[name][k] = v

    def get_model_info_all(self) -> List[Dict[str, Any]]:
        if not self.is_trained:
            self._load_state()
        info_list = []
        for name, info in self.models.items():
            info_list.append({
                'model_name': name,
                'model_version': info.get('version', '1.0.0'),
                'model_type': name,
                'parameters': info.get('params', {}),
                'metrics': info.get('metrics', {}),
                'feature_importance': info.get('feature_importance', []),
                'training_timestamp': info.get('training_timestamp'),
                'training_rows': info.get('training_rows'),
                'is_active': True
            })
        return info_list
