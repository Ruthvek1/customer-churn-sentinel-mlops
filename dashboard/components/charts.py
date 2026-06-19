"""
Reusable chart components for the Streamlit dashboard.

Uses Plotly for interactive, professional-looking visualizations
that make the dashboard impressive on GitHub screenshots.
"""

from typing import Dict, List

import plotly.graph_objects as go


# --- Color Palette ---
COLORS = {
    "primary": "#6366f1",       # Indigo
    "secondary": "#8b5cf6",     # Violet
    "success": "#10b981",       # Emerald
    "warning": "#f59e0b",       # Amber
    "danger": "#ef4444",        # Red
    "info": "#3b82f6",          # Blue
    "background": "#0f172a",    # Slate 900
    "card": "#1e293b",          # Slate 800
    "text": "#e2e8f0",          # Slate 200
    "muted": "#94a3b8",         # Slate 400
    "churn": "#ef4444",
    "no_churn": "#10b981",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
)


def create_churn_gauge(probability: float) -> go.Figure:
    """
    Create a gauge chart showing churn probability.
    
    Args:
        probability: Churn probability (0-1)
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Churn Probability", "font": {"size": 18, "color": COLORS["text"]}},
        number={"suffix": "%", "font": {"size": 36, "color": COLORS["text"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
            "bar": {"color": COLORS["primary"]},
            "bgcolor": COLORS["card"],
            "steps": [
                {"range": [0, 40], "color": "rgba(16, 185, 129, 0.2)"},
                {"range": [40, 70], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [70, 100], "color": "rgba(239, 68, 68, 0.2)"},
            ],
            "threshold": {
                "line": {"color": COLORS["danger"], "width": 4},
                "thickness": 0.75,
                "value": probability * 100,
            },
        }
    ))
    
    fig.update_layout(**PLOTLY_LAYOUT, height=300)
    return fig


def create_feature_importance_chart(features: List[Dict]) -> go.Figure:
    """
    Create a horizontal bar chart of feature importance.
    
    Args:
        features: List of {feature, importance} dicts
    """
    if not features:
        return go.Figure()
    
    features_sorted = sorted(features, key=lambda x: x["importance"])
    
    fig = go.Figure(go.Bar(
        x=[f["importance"] for f in features_sorted],
        y=[f["feature"] for f in features_sorted],
        orientation="h",
        marker=dict(
            color=[f["importance"] for f in features_sorted],
            colorscale=[[0, COLORS["info"]], [1, COLORS["primary"]]],
        ),
        text=[f"{f['importance']:.3f}" for f in features_sorted],
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="",
        height=max(300, len(features) * 30 + 100),
    )
    
    return fig


def create_drift_heatmap(feature_details: Dict) -> go.Figure:
    """
    Create a heatmap showing drift status per feature.
    
    Args:
        feature_details: Dict of feature name -> drift info
    """
    if not feature_details:
        return go.Figure()
    
    features = list(feature_details.keys())
    scores = [feature_details[f].get("drift_score", 0) for f in features]
    drifted = [feature_details[f].get("drifted", False) for f in features]
    
    colors = [COLORS["danger"] if d else COLORS["success"] for d in drifted]
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=features,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{'⚠️ DRIFT' if d else '✅ OK'} ({s:.4f})" for s, d in zip(scores, drifted)],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Feature Drift Scores",
        xaxis_title="Drift Score (p-value)",
        height=max(400, len(features) * 28 + 100),
    )
    
    return fig


def create_prediction_history_chart(predictions: List[Dict]) -> go.Figure:
    """
    Create a timeline chart of recent predictions.
    
    Args:
        predictions: List of prediction records
    """
    if not predictions:
        return go.Figure()
    
    timestamps = [p.get("timestamp", "") for p in predictions]
    probabilities = [p.get("churn_probability", 0) for p in predictions]
    labels = [p.get("risk_level", "LOW") for p in predictions]
    
    color_map = {"HIGH": COLORS["danger"], "MEDIUM": COLORS["warning"], "LOW": COLORS["success"]}
    colors = [color_map.get(lbl, COLORS["info"]) for lbl in labels]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(timestamps))),
        y=probabilities,
        mode="lines+markers",
        marker=dict(color=colors, size=8),
        line=dict(color=COLORS["primary"], width=2),
        text=[f"Risk: {lbl} | Prob: {p:.2%}" for lbl, p in zip(labels, probabilities)],
        hoverinfo="text",
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.7, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="High Risk", annotation_position="right")
    fig.add_hline(y=0.4, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="Medium Risk", annotation_position="right")
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Prediction History",
        xaxis_title="Prediction #",
        yaxis_title="Churn Probability",
        yaxis=dict(range=[0, 1]),
        height=350,
    )
    
    return fig


def create_risk_distribution_chart(stats: Dict) -> go.Figure:
    """
    Create a donut chart showing risk level distribution.
    
    Args:
        stats: Prediction statistics dict
    """
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    values = [
        stats.get("low_risk", 0),
        stats.get("medium_risk", 0),
        stats.get("high_risk", 0)
    ]
    
    if sum(values) == 0:
        values = [1, 0, 0]
    
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=[COLORS["success"], COLORS["warning"], COLORS["danger"]]),
        textinfo="label+percent",
        textfont=dict(color=COLORS["text"]),
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Risk Distribution",
        height=300,
        showlegend=False,
    )
    
    return fig


def create_confusion_matrix_chart(cm_data: Dict) -> go.Figure:
    """
    Create a confusion matrix heatmap.
    
    Args:
        cm_data: Confusion matrix data dict
    """
    matrix = cm_data.get("matrix", [[0, 0], [0, 0]])
    labels = cm_data.get("labels", ["Not Churned", "Churned"])
    
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale=[[0, COLORS["card"]], [1, COLORS["primary"]]],
        text=[[str(val) for val in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=20, color=COLORS["text"]),
        showscale=False,
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=350,
    )
    
    return fig


def create_roc_curve_chart(roc_data: Dict) -> go.Figure:
    """
    Create an ROC curve chart.
    
    Args:
        roc_data: ROC curve data dict with fpr, tpr, auc_score
    """
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=roc_data.get("fpr", [0, 1]),
        y=roc_data.get("tpr", [0, 1]),
        mode="lines",
        name=f"ROC (AUC = {roc_data.get('auc_score', 0):.3f})",
        line=dict(color=COLORS["primary"], width=3),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.1)",
    ))
    
    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(color=COLORS["muted"], width=1, dash="dash"),
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"ROC Curve (AUC = {roc_data.get('auc_score', 0):.3f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        legend=dict(x=0.6, y=0.1),
    )
    
    return fig


def create_drift_trend_chart(drift_history: List[Dict]) -> go.Figure:
    """
    Create a trend chart of drift scores over time.
    
    Args:
        drift_history: List of drift check results
    """
    if not drift_history:
        return go.Figure()
    
    timestamps = [h.get("timestamp", "")[:19] for h in drift_history]
    scores = [h.get("drift_score", 0) for h in drift_history]
    drifted = [h.get("is_drifted", False) for h in drift_history]
    
    colors = [COLORS["danger"] if d else COLORS["success"] for d in drifted]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=scores,
        mode="lines+markers",
        marker=dict(color=colors, size=10),
        line=dict(color=COLORS["primary"], width=2),
        name="Drift Score",
    ))
    
    fig.add_hline(
        y=0.5, line_dash="dash", line_color=COLORS["danger"],
        annotation_text="Drift Threshold",
    )
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Drift Score Over Time",
        xaxis_title="Time",
        yaxis_title="Drift Score",
        yaxis=dict(range=[0, 1]),
        height=350,
    )
    
    return fig
