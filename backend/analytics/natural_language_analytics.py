import pandas as pd
import numpy as np
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class NLQToSQL:
    ALLOWED_TABLES = ['patient_records_view']
    ALLOWED_OPERATIONS = ['SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'LIMIT', 'HAVING', 'JOIN', 'LEFT', 'RIGHT',
                          'INNER', 'OUTER', 'ON', 'AS', 'AND', 'OR', 'NOT', 'IN', 'IS', 'NULL', 'BETWEEN',
                          'LIKE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'CAST', 'EXTRACT', 'DATE', 'INTERVAL',
                          'ASC', 'DESC', 'DISTINCT', 'WITH', 'UNION', 'ALL']
    ALLOWED_AGGREGATES = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'VARIANCE', 'ROUND', 'PERCENTILE',
                          'COALESCE', 'NULLIF', 'EXISTS']
    FORBIDDEN_KEYWORDS = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'REPLACE',
                          'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'SHUTDOWN', 'BACKUP', 'RESTORE', 'COPY',
                          'INTO', 'SET', 'VALUES', 'DECLARE', 'OPEN', 'FETCH', 'CLOSE', 'CURSOR',
                          '--', '/*', '*/', ';', 'xp_', 'sp_', ';--', '0x', 'char(', 'nchar(', 'varchar(',
                          'sys.', 'information_schema', 'pg_', 'sleep(', 'benchmark(']

    COLUMN_DESCRIPTIONS = {
        'patient_id': 'Unique identifier for each patient',
        'age': 'Patient age in years',
        'gender': 'Patient gender (Male/Female/Other/Unknown)',
        'ethnicity': 'Patient ethnicity group',
        'admission_id': 'Unique identifier for each admission encounter',
        'admission_date': 'Date of hospital admission (timestamp)',
        'discharge_date': 'Date of hospital discharge (timestamp)',
        'hospital': 'Hospital / facility name',
        'department': 'Medical department or unit (Cardiology, Oncology, etc.)',
        'diagnosis_code': 'ICD-style diagnosis code',
        'diagnosis_description': 'Full-text description of primary diagnosis',
        'procedure_code': 'Procedure code',
        'procedure_description': 'Full-text description of primary procedure',
        'num_diagnoses': 'Total number of diagnoses recorded for admission',
        'num_procedures': 'Number of procedures performed during admission',
        'num_medications': 'Number of distinct medications prescribed',
        'num_lab_tests': 'Number of lab tests performed',
        'length_of_stay': 'Length of hospital stay in days',
        'treatment_cost': 'Total USD cost of admission treatment',
        'previous_admissions': 'Number of prior admissions for this patient in last 12 months',
        'previous_readmissions': 'Number of prior readmissions for this patient',
        'readmitted': 'Binary flag: whether patient was readmitted within 30 days (1=Yes, 0=No)',
        'risk_score': 'Model-predicted readmission risk 0-100',
        'risk_level': 'Risk tier: LOW / MEDIUM / HIGH',
        'insurance_type': 'Insurance / payer type',
        'admission_type': 'Admission category: Elective / Emergency / Urgent / Transfer',
        'discharge_disposition': 'Discharge location: Home, SNF, Home Health, AMA, etc.'
    }

    DEPARTMENTS = ['Cardiology', 'Oncology', 'Pulmonology', 'General Medicine', 'Surgery', 'ICU', 'Pediatrics',
                   'Neurology', 'Orthopedics', 'Emergency']
    DIAGNOSES = ['Heart Failure (CHF)', 'Pneumonia', 'Acute Myocardial Infarction (AMI)', 'COPD Exacerbation',
                 'Diabetes with Complications', 'Stroke / CVA', 'Hip Fracture', 'Sepsis', 'Post-op Infection',
                 'Kidney Failure / ESRD', 'Cancer (Chemotherapy)', 'GI Bleed', 'Asthma', 'Hypertensive Crisis',
                 'Dehydration / Electrolyte Imbalance']
    RISK_LEVELS = ['LOW', 'MEDIUM', 'HIGH']
    ADMISSION_TYPES = ['Elective', 'Emergency', 'Urgent', 'Transfer']
    INSURANCE_TYPES = ['Medicare', 'Medicaid', 'Private PPO', 'Private HMO', 'Uninsured', 'VA']

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df
        self.table_name = 'patient_records_view'

    def _classify_intent(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        intent = {
            'type': 'aggregation',
            'metrics': [],
            'dimensions': [],
            'filters': [],
            'time_grain': None,
            'sort_by': None,
            'sort_order': 'DESC',
            'limit': None,
            'comparison': False,
            'chart_hint': None
        }

        metric_patterns = [
            (r'readmission rate|readmit rate|readmission ratio', 'readmission_rate', 'AVG(CAST(readmitted AS FLOAT))*100'),
            (r'readmission count|readmissions total|total readmissions', 'readmission_count', 'SUM(CAST(readmitted AS INT))'),
            (r'readmited|readmissions?\b', 'readmission_flag', 'readmitted'),
            (r'average length of stay|avg los|average los|mean length of stay', 'avg_los', 'AVG(length_of_stay)'),
            (r'length of stay|los\b', 'los', 'length_of_stay'),
            (r'total cost|overall cost|sum.*cost|total expenditure', 'total_cost', 'SUM(treatment_cost)'),
            (r'average cost|avg cost|mean cost', 'avg_cost', 'AVG(treatment_cost)'),
            (r'cost\b|treatment cost|charge', 'cost', 'treatment_cost'),
            (r'patient count|number of patients|total patients|patients? admitted|how many patients', 'patient_count', 'COUNT(DISTINCT patient_id)'),
            (r'admission count|number of admissions|total admissions|admissions total', 'admission_count', 'COUNT(admission_id)'),
            (r'admissions?\b', 'admissions', 'COUNT(admission_id)'),
            (r'high[- ]risk patients?|high risk count', 'high_risk_count', "COUNT(DISTINCT CASE WHEN risk_level = 'HIGH' THEN patient_id END)"),
            (r'risk level|risk tier', 'risk_level', 'risk_level'),
            (r'age|patient age', 'age', 'age'),
            (r'average age|mean age|avg age', 'avg_age', 'AVG(age)'),
            (r'mortality|death', None, None),
            (r'diagnose?', 'diagnosis', 'diagnosis_description'),
            (r'department|dept|ward|unit', 'department', 'department'),
            (r'hospital|facility', 'hospital', 'hospital'),
            (r'physician|doctor', None, None),
            (r'insurance|payer', 'insurance_type', 'insurance_type'),
            (r'admission type|type of admission', 'admission_type', 'admission_type'),
            (r'median los|median length of stay', 'median_los', 'PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY length_of_stay)')
        ]

        for pattern, name, expr in metric_patterns:
            if re.search(pattern, q):
                if name and name not in [m['name'] for m in intent['metrics']]:
                    intent['metrics'].append({'name': name, 'expression': expr if expr else name})

        dim_patterns = [
            (r'by department|per department|each department', 'department', 'department'),
            (r'by hospital|per hospital|each hospital', 'hospital', 'hospital'),
            (r'by diagnosis|per diagnosis|each diagnosis|top diagnosis|diagnosis distribution',
             'diagnosis_description', 'diagnosis_description'),
            (r'by month|monthly|per month|each month', 'admission_month',
             "EXTRACT(YEAR FROM admission_date) || '-' || LPAD(EXTRACT(MONTH FROM admission_date)::text,2,'0')"),
            (r'by quarter|quarterly|per quarter|each quarter|last quarter|previous quarter', 'admission_quarter',
             "EXTRACT(YEAR FROM admission_date) || '-Q' || EXTRACT(QUARTER FROM admission_date)"),
            (r'by year|yearly|annual|per year', 'admission_year', 'EXTRACT(YEAR FROM admission_date)'),
            (r'by date|daily|per day|each day|by day|over time|trend', 'admission_date',
             "DATE(admission_date)"),
            (r'by risk level|per risk level|risk level|by risk', 'risk_level', 'risk_level'),
            (r'by gender|per gender|gender', 'gender', 'gender'),
            (r'by ethnicity|per ethnicity|ethnicity', 'ethnicity', 'ethnicity'),
            (r'by age group|age group|by age', 'age_group', None),
            (r'by insurance|by payer|per insurance', 'insurance_type', 'insurance_type'),
            (r'by admission type|per admission type', 'admission_type', 'admission_type'),
            (r'compare.*(and|vs|versus).*', None, '__comparison__')
        ]

        for pattern, name, expr in dim_patterns:
            if re.search(pattern, q):
                if expr == '__comparison__':
                    intent['comparison'] = True
                elif name and name not in [d['name'] for d in intent['dimensions']]:
                    intent['dimensions'].append({'name': name, 'expression': expr if expr else name})

        time_keywords = [
            (r'last (year|12 months|twelve months)', 'year', '1 year'),
            (r'last (quarter|3 months|three months|q[0-9])', 'quarter', '3 months'),
            (r'last (month|30 days|one month)', 'month', '1 month'),
            (r'last (week|7 days|one week)', 'week', '7 days'),
            (r'past (week|7 days)', 'week', '7 days'),
            (r'this month|current month', 'this_month', None),
            (r'this year|current year', 'this_year', None),
            (r'q[1-4]\s*20[0-9]{2}|quarter\s*[1-4]', 'specific_quarter', None)
        ]
        for pat, name, _ in time_keywords:
            if re.search(pat, q):
                intent['time_filter'] = name

        if re.search(r'top (\d+)|first (\d+)|highest (\d+)', q):
            m = re.search(r'top (\d+)|first (\d+)|highest (\d+)', q)
            intent['limit'] = int(next(g for g in m.groups() if g))
        if re.search(r'bottom (\d+)|lowest (\d+)', q):
            m = re.search(r'bottom (\d+)|lowest (\d+)', q)
            intent['limit'] = int(next(g for g in m.groups() if g))
            intent['sort_order'] = 'ASC'

        if re.search(r'why.*(increase|decrease|changed|went up|went down|higher|lower|spiked|dropped)', q):
            intent['type'] = 'explanation'

        if re.search(r'compare|comparison|vs\.?|versus', q):
            intent['comparison'] = True

        if re.search(r'trend|over time|monthly|quarterly|daily|yearly', q):
            intent['chart_hint'] = 'line'
        elif re.search(r'distribution|breakdown|by .*department|by .*diagnosis|top \d+', q):
            intent['chart_hint'] = 'bar'
        elif re.search(r'compare|comparison|vs', q):
            intent['chart_hint'] = 'bar'
        elif len(intent['dimensions']) == 1:
            intent['chart_hint'] = 'bar'
        else:
            intent['chart_hint'] = 'table'

        if not intent['metrics']:
            intent['metrics'].append({'name': 'admission_count', 'expression': 'COUNT(admission_id)'})

        intent['filters_raw'] = self._extract_filters(q)
        return intent

    def _extract_filters(self, query: str) -> List[Dict]:
        q = query.lower()
        filters = []
        for dept in self.DEPARTMENTS:
            if dept.lower() in q:
                filters.append({'column': 'department', 'operator': '=', 'value': dept})
        for diag in self.DIAGNOSES:
            first_word = diag.split()[0].lower()
            if first_word in q and len(first_word) > 3:
                filters.append({'column': 'diagnosis_description', 'operator': 'LIKE', 'value': f"%{diag.split('(')[0].strip()}%"})
        for risk in self.RISK_LEVELS:
            if risk.lower() in q:
                filters.append({'column': 'risk_level', 'operator': '=', 'value': risk})
        for at in self.ADMISSION_TYPES:
            if at.lower() in q:
                filters.append({'column': 'admission_type', 'operator': '=', 'value': at})
        for ins in self.INSURANCE_TYPES:
            if ins.lower() in q:
                filters.append({'column': 'insurance_type', 'operator': '=', 'value': ins})
        for gen in ['male', 'female']:
            if re.search(rf'\b{gen}\b', q):
                filters.append({'column': 'gender', 'operator': '=', 'value': gen.title()})
        age_match = re.search(r'age\s*(over|above|under|below|>|<|>=|<=)?\s*(\d{1,3})', q)
        if age_match:
            op = age_match.group(1)
            val = int(age_match.group(2))
            op_map = {'over': '>', 'above': '>', 'under': '<', 'below': '<',
                      '>': '>', '<': '<', '>=': '>=', '<=': '<=', None: '='}
            filters.append({'column': 'age', 'operator': op_map[op], 'value': val})
        return filters

    def _validate_sql(self, sql: str) -> Tuple[bool, str]:
        clean_upper = re.sub(r"'[^']*'", '', sql).upper()
        for kw in self.FORBIDDEN_KEYWORDS:
            if kw.upper() in clean_upper:
                return False, f"Forbidden keyword or pattern detected: '{kw}'"
        stripped = clean_upper.strip().lstrip('(').strip()
        if not stripped.startswith('SELECT') and not stripped.startswith('WITH'):
            return False, "Only SELECT queries are allowed"
        if ';' in sql.rstrip(';'):
            return False, "Multiple statements not allowed"
        tokens = re.findall(r"[A-Za-z_]+", clean_upper)
        if any(t in ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE'] for t in tokens):
            return False, "Destructive operations are not permitted"
        return True, "OK"

    def generate_sql(self, query: str) -> Tuple[str, Dict]:
        intent = self._classify_intent(query)
        metrics_sql = ", ".join([f"{m['expression']} AS {m['name']}" for m in intent['metrics']])
        if intent['dimensions']:
            group_by_sql = ", ".join([str(i + 1) for i in range(len(intent['dimensions']))])
            select_dims = ", ".join([f"{d['expression']} AS {d['name']}" for d in intent['dimensions']])
            select_sql = f"{select_dims}, {metrics_sql}"
        else:
            select_sql = metrics_sql
            group_by_sql = None
        from_sql = f"FROM {self.table_name}"
        where_clauses = []
        params = {}
        for i, f in enumerate(intent.get('filters_raw', [])):
            ph = f"val_{i}"
            col = f['column']
            op = f['operator']
            if op.upper() == 'LIKE':
                where_clauses.append(f"{col} LIKE :{ph}")
            else:
                where_clauses.append(f"{col} {op} :{ph}")
            params[ph] = f['value']
        if 'time_filter' in intent:
            tf = intent['time_filter']
            if tf in ('year', 'quarter', 'month', 'week'):
                where_clauses.append(f"admission_date >= CURRENT_DATE - INTERVAL '1 {tf}'")
            elif tf == 'this_month':
                where_clauses.append("EXTRACT(YEAR FROM admission_date) = EXTRACT(YEAR FROM CURRENT_DATE)")
                where_clauses.append("EXTRACT(MONTH FROM admission_date) = EXTRACT(MONTH FROM CURRENT_DATE)")
            elif tf == 'this_year':
                where_clauses.append("EXTRACT(YEAR FROM admission_date) = EXTRACT(YEAR FROM CURRENT_DATE)")
        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)
        group_sql = f" GROUP BY {group_by_sql}" if group_by_sql else ""
        order_sql = ""
        if intent['dimensions']:
            first_metric = intent['metrics'][0]['name']
            order_sql = f" ORDER BY {first_metric} {intent['sort_order']}"
        limit_sql = f" LIMIT {intent['limit']}" if intent['limit'] else ""
        sql = f"SELECT {select_sql} {from_sql}{where_sql}{group_sql}{order_sql}{limit_sql}"
        sql = re.sub(r'\s+', ' ', sql).strip()
        valid, msg = self._validate_sql(sql)
        if not valid:
            raise ValueError(f"Generated SQL failed validation: {msg}")
        return sql, {'intent': intent, 'params': params}


class NaturalLanguageAnalytics:
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df
        self.sql_generator = NLQToSQL(df)

    def set_dataframe(self, df: pd.DataFrame):
        self.df = df
        self.sql_generator.df = df

    def _execute_pandas_query(self, intent: Dict) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No dataframe loaded for analysis")
        df = self.df.copy()
        df['admission_date'] = pd.to_datetime(df['admission_date'], errors='coerce')
        df['discharge_date'] = pd.to_datetime(df['discharge_date'], errors='coerce')
        for f in intent.get('filters_raw', []):
            col, op, val = f['column'], f['operator'], f['value']
            if col not in df.columns:
                continue
            if op.upper() == 'LIKE':
                like_val = val.strip('%')
                df = df[df[col].astype(str).str.contains(like_val, case=False, na=False)]
            elif op == '=':
                df = df[df[col].astype(str) == str(val)]
            elif op == '>':
                df = df[pd.to_numeric(df[col], errors='coerce') > val]
            elif op == '<':
                df = df[pd.to_numeric(df[col], errors='coerce') < val]
            elif op == '>=':
                df = df[pd.to_numeric(df[col], errors='coerce') >= val]
            elif op == '<=':
                df = df[pd.to_numeric(df[col], errors='coerce') <= val]
        if 'time_filter' in intent:
            tf = intent['time_filter']
            df = df.dropna(subset=['admission_date'])
            max_date = df['admission_date'].max()
            if tf == 'year':
                df = df[df['admission_date'] >= max_date - pd.DateOffset(years=1)]
            elif tf == 'quarter':
                df = df[df['admission_date'] >= max_date - pd.DateOffset(months=3)]
            elif tf == 'month':
                df = df[df['admission_date'] >= max_date - pd.DateOffset(months=1)]
            elif tf == 'week':
                df = df[df['admission_date'] >= max_date - pd.DateOffset(days=7)]

        metric_computations = {
            'patient_count': lambda d: d['patient_id'].nunique() if 'patient_id' in d.columns else 0,
            'admission_count': lambda d: len(d),
            'readmission_count': lambda d: int(d['readmitted'].sum()) if 'readmitted' in d.columns else 0,
            'readmission_rate': lambda d: round(float(d['readmitted'].mean() * 100), 2) if 'readmitted' in d.columns and len(d) > 0 else 0.0,
            'avg_los': lambda d: round(float(d['length_of_stay'].mean()), 2) if 'length_of_stay' in d.columns and len(d) > 0 else 0.0,
            'avg_cost': lambda d: round(float(d['treatment_cost'].mean()), 2) if 'treatment_cost' in d.columns and len(d) > 0 else 0.0,
            'total_cost': lambda d: round(float(d['treatment_cost'].sum()), 2) if 'treatment_cost' in d.columns else 0.0,
            'high_risk_count': lambda d: int((d['risk_level'] == 'HIGH').sum()) if 'risk_level' in d.columns else 0,
            'avg_age': lambda d: round(float(d['age'].mean()), 1) if 'age' in d.columns and len(d) > 0 else 0.0,
            'los': lambda d: d['length_of_stay'],
            'cost': lambda d: d['treatment_cost'],
            'age': lambda d: d['age'],
            'readmission_flag': lambda d: d['readmitted'],
            'risk_level': lambda d: d['risk_level'],
            'diagnosis': lambda d: d['diagnosis_description'],
            'department': lambda d: d['department'],
            'hospital': lambda d: d['hospital'],
            'insurance_type': lambda d: d['insurance_type'],
            'admission_type': lambda d: d['admission_type'],
        }

        dimension_transforms = {
            'department': lambda d: d['department'],
            'hospital': lambda d: d['hospital'],
            'diagnosis_description': lambda d: d['diagnosis_description'],
            'risk_level': lambda d: d['risk_level'],
            'gender': lambda d: d['gender'],
            'ethnicity': lambda d: d['ethnicity'],
            'insurance_type': lambda d: d['insurance_type'],
            'admission_type': lambda d: d['admission_type'],
            'admission_month': lambda d: d['admission_date'].dt.strftime('%Y-%m'),
            'admission_quarter': lambda d: d['admission_date'].dt.to_period('Q').astype(str),
            'admission_year': lambda d: d['admission_date'].dt.year,
            'admission_date': lambda d: d['admission_date'].dt.date,
            'age_group': lambda d: pd.cut(d['age'], bins=[0, 18, 45, 65, 80, 120],
                                          labels=['0-18', '19-45', '46-65', '66-80', '80+']),
        }

        if not intent['dimensions']:
            result_row = {}
            for metric in intent['metrics']:
                name = metric['name']
                if name in metric_computations:
                    val = metric_computations[name](df)
                    result_row[name] = val
                else:
                    if name in df.columns:
                        result_row[name] = df[name].iloc[0] if len(df) > 0 else None
            return pd.DataFrame([result_row])
        else:
            dims_df = pd.DataFrame()
            for dim in intent['dimensions']:
                name = dim['name']
                if name in dimension_transforms:
                    dims_df[name] = dimension_transforms[name](df)
                elif name in df.columns:
                    dims_df[name] = df[name]
                else:
                    dims_df[name] = 'N/A'
            combined = pd.concat([dims_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
            group_cols = [d['name'] for d in intent['dimensions']]
            agg_dict = {}
            for metric in intent['metrics']:
                name = metric['name']
                if name == 'patient_count':
                    agg_dict[name] = ('patient_id', 'nunique') if 'patient_id' in combined.columns else (group_cols[0], 'count')
                elif name == 'admission_count':
                    agg_dict[name] = (group_cols[0], 'count')
                elif name == 'readmission_count':
                    agg_dict[name] = ('readmitted', 'sum') if 'readmitted' in combined.columns else (group_cols[0], 'count')
                elif name == 'readmission_rate':
                    agg_dict[name] = ('readmitted', 'mean') if 'readmitted' in combined.columns else (group_cols[0], 'count')
                elif name in ('avg_los', 'avg_age', 'avg_cost'):
                    src = {'avg_los': 'length_of_stay', 'avg_age': 'age', 'avg_cost': 'treatment_cost'}[name]
                    agg_dict[name] = (src, 'mean') if src in combined.columns else (group_cols[0], 'count')
                elif name == 'total_cost':
                    agg_dict[name] = ('treatment_cost', 'sum') if 'treatment_cost' in combined.columns else (group_cols[0], 'count')
                elif name == 'high_risk_count':
                    if 'risk_level' in combined.columns:
                        combined['_hr'] = (combined['risk_level'] == 'HIGH').astype(int)
                        agg_dict[name] = ('_hr', 'sum')
                    else:
                        agg_dict[name] = (group_cols[0], 'count')
                else:
                    if name in combined.columns:
                        agg_dict[name] = (name, 'mean')
                    else:
                        agg_dict[name] = (group_cols[0], 'count')
            result = combined.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()
            for metric in intent['metrics']:
                name = metric['name']
                if name == 'readmission_rate' and name in result.columns:
                    result[name] = (result[name] * 100).round(2)
                elif name in ('avg_los', 'avg_age', 'avg_cost') and name in result.columns:
                    result[name] = result[name].round(2)
                elif name == 'total_cost' and name in result.columns:
                    result[name] = result[name].round(2)
            if intent['metrics']:
                sort_col = intent['metrics'][0]['name']
                if sort_col in result.columns:
                    result = result.sort_values(sort_col, ascending=(intent['sort_order'] == 'ASC'))
            if intent['limit']:
                result = result.head(intent['limit'])
            return result

    def _build_chart(self, result_df: pd.DataFrame, intent: Dict) -> Optional[Dict]:
        if result_df is None or len(result_df) == 0:
            return None
        dim_cols = [d['name'] for d in intent['dimensions']] if intent['dimensions'] else []
        metric_cols = [m['name'] for m in intent['metrics']]
        chart_type = intent.get('chart_hint', 'bar')
        try:
            if dim_cols and metric_cols:
                x_col = dim_cols[0]
                traces = []
                for metric in metric_cols:
                    if metric in result_df.columns:
                        y_data = pd.to_numeric(result_df[metric], errors='coerce').fillna(0).tolist()
                        traces.append({
                            'name': metric.replace('_', ' ').title(),
                            'x': result_df[x_col].astype(str).tolist(),
                            'y': y_data,
                            'type': chart_type if chart_type != 'line' else 'scatter',
                            'mode': 'lines+markers' if chart_type == 'line' else None,
                        })
                return {
                    'chart_type': chart_type,
                    'title': f"{', '.join(m.replace('_',' ').title() for m in metric_cols)} by {x_col.replace('_',' ').title()}",
                    'data': traces,
                    'layout': {
                        'xaxis_title': x_col.replace('_', ' ').title(),
                        'yaxis_title': 'Value',
                        'barmode': 'group' if len(traces) > 1 else 'relative',
                        'margin': {'l': 50, 'r': 20, 't': 50, 'b': 120},
                        'xaxis': {'tickangle': -35 if len(result_df) > 6 else 0}
                    }
                }
        except Exception:
            return None
        return None

    def _generate_explanation(self, query: str, result_df: pd.DataFrame,
                              intent: Dict) -> str:
        if result_df is None or len(result_df) == 0:
            return "No matching data was found for this query. Try broadening your filters or rephrasing."
        metric_cols = [m['name'] for m in intent['metrics']]
        dim_cols = [d['name'] for d in intent['dimensions']]
        parts = []
        if not dim_cols:
            parts.append(f"Based on the dataset:")
            for m in metric_cols:
                if m in result_df.columns:
                    val = result_df[m].iloc[0]
                    label = m.replace('_', ' ').title()
                    if m == 'readmission_rate':
                        parts.append(f"  - {label}: {val:.2f}%")
                    elif m == 'avg_cost' or m == 'total_cost':
                        parts.append(f"  - {label}: ${val:,.2f}")
                    elif 'count' in m:
                        parts.append(f"  - {label}: {int(val):,}")
                    else:
                        parts.append(f"  - {label}: {val}")
            if intent.get('filters_raw'):
                filter_desc = ", ".join([f"{f['column']} {f['operator']} {f['value']}" for f in intent['filters_raw'][:5]])
                parts.append(f"\nApplied filters: {filter_desc}")
        else:
            top_n = min(5, len(result_df))
            label_dim = dim_cols[0].replace('_', ' ').title()
            for m in metric_cols:
                label_m = m.replace('_', ' ').title()
                parts.append(f"**{label_m} by {label_dim} (Top {top_n}):**")
                tmp = result_df.head(top_n).copy()
                for i, (_, row) in enumerate(tmp.iterrows(), 1):
                    dim_val = row[dim_cols[0]]
                    m_val = row.get(m, 'N/A')
                    if isinstance(m_val, float):
                        if m == 'readmission_rate':
                            display = f"{m_val:.2f}%"
                        elif 'cost' in m:
                            display = f"${m_val:,.2f}"
                        else:
                            display = f"{m_val:.2f}"
                    else:
                        display = str(m_val)
                    parts.append(f"  {i}. {dim_val}: {display}")
            if intent['comparison']:
                parts.append("\n**Comparison summary:** Identify gaps between the categories above — the largest variance typically signals an opportunity for investigation (e.g., see AI Insights section for driver analysis).")
            if intent['type'] == 'explanation':
                parts.append("\n**Note on 'why':** This module surfaces descriptive aggregates. Root-cause and multi-variate driver analysis are provided in the 'AI Insights' and 'Patient Risk' modules of the platform (see feature importance, correlated factors, and automatic insight generation sections).")
        return "\n".join(parts)

    def analyze_query(self, query: str, generate_chart: bool = True) -> Dict[str, Any]:
        start = time.time()
        response = {
            'natural_query': query,
            'generated_sql': None,
            'query_result': None,
            'explanation': '',
            'chart_type': None,
            'chart_data': None,
            'execution_time_ms': 0,
            'success': True,
            'error_message': None
        }
        try:
            sql, meta = self.sql_generator.generate_sql(query)
            response['generated_sql'] = sql
            intent = meta['intent']
            result = self._execute_pandas_query(intent)
            response['query_result'] = {
                'columns': list(result.columns),
                'rows': result.where(pd.notnull(result), None).to_dict(orient='records')
            }
            if generate_chart:
                chart = self._build_chart(result, intent)
                if chart:
                    response['chart_data'] = chart
                    response['chart_type'] = chart.get('chart_type')
            response['explanation'] = self._generate_explanation(query, result, intent)
        except Exception as e:
            response['success'] = False
            response['error_message'] = str(e)
            response['explanation'] = f"An error occurred while processing your query: {str(e)}"
        response['execution_time_ms'] = int((time.time() - start) * 1000)
        return response
