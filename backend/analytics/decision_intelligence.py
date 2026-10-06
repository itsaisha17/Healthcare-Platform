import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class DecisionIntelligenceEngine:
    def __init__(self):
        pass

    def _build_actions_for_readmission_trend(self, insight: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        evidence = insight.get('evidence', {})
        problem = insight.get('title', 'Readmission rate changed significantly between periods')
        possible_causes = insight.get('possible_drivers', [])
        ev = {
            'period_comparison': evidence,
            'finding': insight.get('finding', '')
        }
        actions = []
        impacts = []
        actions.append("Implement targeted discharge bundle (med rec, follow-up call, PCP appt) for all high-risk patients")
        impacts.append({
            'action_idx': 0,
            'estimated_impact': '-5% to -12% readmission rate in high-risk subgroup',
            'evidence_basis': 'Q4 2023 guidelines estimate 35-40% of readmissions as preventable via standard bundles',
            'confidence': 'MEDIUM',
            'is_hypothesis': True
        })
        actions.append("Stratify readmissions by diagnosis and prioritize top 2 conditions for disease-management programs")
        impacts.append({
            'action_idx': 1,
            'estimated_impact': '-3% to -8% overall when combined with bundle',
            'evidence_basis': 'CMS/NCQA disease-management program meta-analyses',
            'confidence': 'MEDIUM',
            'is_hypothesis': True
        })
        actions.append("Audit case manager assignment compliance for patients with ≥2 prior admissions")
        impacts.append({
            'action_idx': 2,
            'estimated_impact': 'Process improvement; reduces preventable readmissions in tracked subgroup',
            'evidence_basis': 'Policy HP-2024-RISK-003 requires <90% compliance threshold',
            'confidence': 'HIGH',
            'is_hypothesis': False
        })
        actions.append("Add 48-hour post-discharge telehealth outreach pilot for highest risk decile")
        impacts.append({
            'action_idx': 3,
            'estimated_impact': '-15% to -25% readmissions in piloted subgroup (hypothesis)',
            'evidence_basis': 'Published telehealth readmission studies; local validation required',
            'confidence': 'LOW',
            'is_hypothesis': True
        })
        return {
            'problem': problem,
            'evidence': ev,
            'possible_causes': possible_causes if possible_causes else ['Multifactorial; see AI Insight evidence'],
            'recommended_actions': actions,
            'expected_impacts': impacts,
            'impact_type': 'HYPOTHESIS'
        }

    def _build_actions_for_department_performance(self, insight: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        ev = insight.get('evidence', {})
        dept = ev.get('department', 'Target department')
        problem = f"{dept} readmission/performance deviates from peer group average (insight severity: {insight.get('severity')})"
        possible_causes = insight.get('possible_drivers', [])
        actions = [
            f"Open structured case review for {dept}: pull 10-15 latest readmitted cases for peer discussion",
            f"Baseline and track {dept} discharge bundle compliance: med rec, follow-up appt, teach-back documented",
            f"Compare {dept} LOS and cost against peer department after case-mix / diagnosis standardization",
            f"Deploy standardized care path for top 3 diagnoses driving readmissions in {dept}"
        ]
        impacts = [
            {'action_idx': 0, 'estimated_impact': 'Identifies root causes in 40-70% of reviewed cases',
             'evidence_basis': 'Hospital QI methodology standard', 'confidence': 'HIGH', 'is_hypothesis': False},
            {'action_idx': 1, 'estimated_impact': 'Each +10% bundle compliance ~-1.5 to -3% readmission rate (estimated)',
             'evidence_basis': 'IHI discharge bundle meta-analysis', 'confidence': 'MEDIUM', 'is_hypothesis': True},
            {'action_idx': 2, 'estimated_impact': 'Case-mix standardization avoids false-positive performance attribution',
             'evidence_basis': 'Healthcare analytics standard practice', 'confidence': 'HIGH', 'is_hypothesis': False},
            {'action_idx': 3, 'estimated_impact': '-5% to -12% readmission for pathway-targeted conditions',
             'evidence_basis': 'Clinical guideline adherence programs', 'confidence': 'MEDIUM', 'is_hypothesis': True},
        ]
        return {
            'problem': problem,
            'evidence': insight.get('evidence', {}),
            'possible_causes': possible_causes,
            'recommended_actions': actions,
            'expected_impacts': impacts,
            'impact_type': 'HYPOTHESIS'
        }

    def _build_actions_for_risk_concentration(self, insight: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        problem = "High-risk patient segment concentrates readmission burden (preventable opportunity)"
        possible_causes = insight.get('possible_drivers', [])
        actions = [
            "100% HIGH-risk patients assigned a case manager within 12 hours (per HP-2024-RISK-003)",
            "Mandatory discharge planning meeting 48h pre-discharge for every HIGH-risk patient",
            "Telehealth follow-up within 72h post-discharge for HIGH-risk cohort",
            "Prioritize patients with >=3 prior admissions for care-transition coaching program",
            "Evaluate SNF referral pathway for HIGH-risk patients with no home support"
        ]
        impacts = [
            {'action_idx': 0, 'estimated_impact': 'Process control; ensures HIGH-risk tier is identifiable at point of care',
             'evidence_basis': 'Policy requirement', 'confidence': 'HIGH', 'is_hypothesis': False},
            {'action_idx': 1, 'estimated_impact': 'Fewer gaps at discharge; reduces communication-related readmissions',
             'evidence_basis': 'IHI / CMS care-transition playbook', 'confidence': 'MEDIUM', 'is_hypothesis': True},
            {'action_idx': 2, 'estimated_impact': '-12% to -20% readmission in HIGH-risk (hypothesis per pilot literature)',
             'evidence_basis': 'Telehealth readmission pilot studies', 'confidence': 'MEDIUM', 'is_hypothesis': True},
            {'action_idx': 3, 'estimated_impact': '47% of readmissions currently come from patients with >=3 prior admissions; highest ROI',
             'evidence_basis': 'Q4 2023 report segment analysis', 'confidence': 'HIGH', 'is_hypothesis': False},
            {'action_idx': 4, 'estimated_impact': 'Reduces unsafe discharges; impact size requires local validation',
             'evidence_basis': 'SNF placement protocol', 'confidence': 'LOW', 'is_hypothesis': True},
        ]
        return {
            'problem': problem,
            'evidence': insight.get('evidence', {}),
            'possible_causes': possible_causes,
            'recommended_actions': actions,
            'expected_impacts': impacts,
            'impact_type': 'HYBRID'
        }

    def _build_actions_for_cost_anomaly(self, insight: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        ev = insight.get('evidence', {})
        dept = ev.get('department', 'Target department')
        problem = f"{dept} cost/case significantly exceeds peer average — financial and utilization review opportunity"
        possible_causes = insight.get('possible_drivers', [])
        actions = [
            f"{dept} top-20 cost cases per month go to concurrent case management review",
            f"Utilization review of LOS outliers: automate trigger at 75th percentile for {dept}",
            f"Supply / device pricing re-bid for top 3 procedural supply categories in {dept}",
            f"Lab & imaging order-set review: reduce duplication and low-yield testing",
            f"Pharmacy: interchangeable biologic / therapeutic substitution protocol for {dept}"
        ]
        impacts = [
            {'action_idx': 0, 'estimated_impact': 'Case review typically identifies 5-15% cost savings on outlier cases',
             'evidence_basis': 'Case management benchmarks', 'confidence': 'MEDIUM', 'is_hypothesis': True},
            {'action_idx': 1, 'estimated_impact': 'LOS reduction of 0.2-0.5 day avg; direct cost and bed-turn benefit',
             'evidence_basis': 'UR program case studies', 'confidence': 'MEDIUM', 'is_hypothesis': True},
            {'action_idx': 2, 'estimated_impact': 'Supply cost 8-15% reduction in selected categories',
             'evidence_basis': 'Standard group-purchasing benchmarks', 'confidence': 'HIGH', 'is_hypothesis': True},
            {'action_idx': 3, 'estimated_impact': '5-10% lab / imaging cost reduction; quality-neutral when evidence-based',
             'evidence_basis': 'Choosing Wisely / ABIM Foundation campaigns', 'confidence': 'MEDIUM', 'is_hypothesis': True},
            {'action_idx': 4, 'estimated_impact': 'Pharmacy cost reduction; dependent on current utilization',
             'evidence_basis': 'P&T committee protocol standard', 'confidence': 'MEDIUM', 'is_hypothesis': True},
        ]
        return {
            'problem': problem,
            'evidence': ev,
            'possible_causes': possible_causes,
            'recommended_actions': actions,
            'expected_impacts': impacts,
            'impact_type': 'HYPOTHESIS'
        }

    def _build_actions_for_diagnosis_cluster(self, insight: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        ev = insight.get('evidence', {})
        dx = ev.get('top_diagnosis', 'Target diagnosis')
        problem = f"Diagnosis '{dx}' has highest readmission rate among all tracked conditions"
        possible_causes = insight.get('possible_drivers', [])
        actions = [
            f"Form multi-disciplinary improvement team for '{dx}' pathway",
            f"Implement evidence-based order set with decision-support at order entry",
            f"Patient education + teach-back protocol for '{dx}' discharge instructions",
            f"30-day care manager check-ins for every '{dx}' discharge",
            f"Track: time-to-first-follow-up and medication possession ratio"
        ]
        impacts = [
            {'action_idx': 0, 'estimated_impact': 'Governance; required to sustain change',
             'evidence_basis': 'QI standard', 'confidence': 'HIGH', 'is_hypothesis': False},
            {'action_idx': 1, 'estimated_impact': 'Reduces unwarranted variation; -5% to -10% readmission (hypothesis)',
             'evidence_basis': 'Order-set adherence literature', 'confidence': 'MEDIUM', 'is_hypothesis': True},
            {'action_idx': 2, 'estimated_impact': 'Improves adherence; indirect readmission reduction',
             'evidence_basis': 'Patient education standard of care', 'confidence': 'MEDIUM', 'is_hypothesis': True},
            {'action_idx': 3, 'estimated_impact': 'Earlier identification of deterioration; reduces ED revisits',
             'evidence_basis': 'Transitional care models', 'confidence': 'MEDIUM', 'is_hypothesis': True},
            {'action_idx': 4, 'estimated_impact': 'Provides leading indicators for the improvement program',
             'evidence_basis': 'Core metrics for chronic disease programs', 'confidence': 'HIGH', 'is_hypothesis': False},
        ]
        return {
            'problem': problem,
            'evidence': ev,
            'possible_causes': possible_causes,
            'recommended_actions': actions,
            'expected_impacts': impacts,
            'impact_type': 'HYPOTHESIS'
        }

    def _build_actions_for_los_disparity(self, insight: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        problem = "Weekend vs weekday average LOS discrepancy signals operational / capacity gap"
        possible_causes = insight.get('possible_drivers', [])
        actions = [
            "Weekend discharge planning staffing uplift (SW / CM coverage model)",
            "Weekend SNF / post-acute placement liaison role",
            "Pre-weekend (Friday) discharge huddle to identify and clear Saturday-ready patients",
            "Extended / Saturday pharmacy discharge medication processing window"
        ]
        impacts = [
            {'action_idx': 0, 'estimated_impact': '-0.3 to -0.7 day reduction in weekend LOS',
             'evidence_basis': 'Case-staffing interventions from peer hospitals', 'confidence': 'MEDIUM', 'is_hypothesis': True},
            {'action_idx': 1, 'estimated_impact': 'Reduces weekend-hold pattern; bed-turn benefit',
             'evidence_basis': 'Post-acute placement operations', 'confidence': 'MEDIUM', 'is_hypothesis': True},
            {'action_idx': 2, 'estimated_impact': 'Avoids patient boarding through weekend; fast ROI',
             'evidence_basis': 'Operational flow playbook', 'confidence': 'HIGH', 'is_hypothesis': True},
            {'action_idx': 3, 'estimated_impact': 'Removes pharmacy discharge delay bottleneck',
             'evidence_basis': 'Discharge barrier root-cause data', 'confidence': 'HIGH', 'is_hypothesis': False},
        ]
        return {
            'problem': problem,
            'evidence': insight.get('evidence', {}),
            'possible_causes': possible_causes,
            'recommended_actions': actions,
            'expected_impacts': impacts,
            'impact_type': 'HYPOTHESIS'
        }

    def build_decisions_for_insight(self, insight: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        itype = insight.get('insight_type')
        handlers = {
            'READMISSION_TREND': self._build_actions_for_readmission_trend,
            'DEPARTMENT_PERFORMANCE': self._build_actions_for_department_performance,
            'RISK_CONCENTRATION': self._build_actions_for_risk_concentration,
            'COST_ANOMALY': self._build_actions_for_cost_anomaly,
            'DIAGNOSIS_CLUSTER': self._build_actions_for_diagnosis_cluster,
            'LOS_DISPARITY': self._build_actions_for_los_disparity,
        }
        handler = handlers.get(itype)
        if handler:
            return handler(insight, df)
        return {
            'problem': insight.get('title', 'System-level insight'),
            'evidence': insight.get('evidence', {}),
            'possible_causes': insight.get('possible_drivers', []),
            'recommended_actions': [
                "Open structured investigation using the recommended investigation path from AI Insights",
                "Validate insight with a small case review sample before scaling interventions",
                "Track one leading metric and one lag metric as a control"
            ],
            'expected_impacts': [
                {'action_idx': 0, 'estimated_impact': 'Investigation determines root cause; estimate pending',
                 'evidence_basis': 'QI structured investigation', 'confidence': 'HIGH', 'is_hypothesis': True}
            ],
            'impact_type': 'HYPOTHESIS'
        }
