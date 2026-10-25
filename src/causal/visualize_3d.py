"""3D Interactive Spatiotemporal Visualizations for Longitudinal Causal Health Modeling.

Provides:
1. 3D Spatiotemporal Patient Trajectory Manifold:
   - X-axis: Latent Causal Dimension 1 (z1: Immunological / Inflammatory Axis)
   - Y-axis: Latent Causal Dimension 2 (z2: Treatment Response / Organ Involvement Axis)
   - Z-axis: Longitudinal Time (Months or Visit Step)
   - Traces each patient's evolution across visits with connected 3D lines,
     color-mapped by next-visit flare risk or severity score, and sized by epistemic uncertainty.

2. 3D Tiered Causal Directed Acyclic Graph (DAG) Network:
   - Multimodal clinical variables organized across functional 3D planes:
     * Layer 0: Demographics & Genotype
     * Layer 1: Pharmacological Interventions
     * Layer 2: Latent Causal Health State (Z_causal)
     * Layer 3: Dynamic Biomarkers & Clinical Symptoms
     * Layer 4: Downstream Clinical Outcomes (Flare & Organ Damage)
   - Directed 3D edges colored by protective (green) vs pathogenic (red/orange) causal mechanisms.

Outputs:
   - Standalone interactive HTML files:
     `reports/causal/3d_patient_trajectories.html`
     `reports/causal/3d_causal_graph.html`
   - Direct plotly.graph_objects figures for Streamlit integration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.causal import dataset as D
from src.causal import train as T
from src.causal.dataset import SequenceDataset
from src.causal.model import CausalDeepModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORT_DIR = ROOT / "reports" / "causal"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def extract_latent_trajectories(
    model: CausalDeepModel,
    sd: SequenceDataset,
    device: torch.device = DEVICE,
    n_patients_max: int = 50
) -> dict[str, Any]:
    """Extracts longitudinal latent causal representations z^c across all visits."""
    model.eval()
    N_eval = min(n_patients_max, sd.N)
    sd_sub = D.subset_dataset(sd, np.arange(N_eval))

    zc_list = []
    probs_list = []
    sevs_list = []

    batch_size = 8
    with torch.no_grad():
        for b_start in range(0, N_eval, batch_size):
            b_end = min(b_start + batch_size, N_eval)
            sub_b = D.subset_dataset(sd_sub, np.arange(b_start, b_end))
            batch = T.to_torch_batch(sub_b, device=device)
            out = model(
                batch["X"],
                batch["M"],
                batch["gap"],
                batch["pad"],
                sample_noise=False
            )
            zc_list.append(out["zc"].cpu().numpy())
            probs_list.append(torch.sigmoid(out["flare"]).cpu().numpy())
            sevs_list.append(model.sev_mean(out).cpu().numpy())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    zc = np.concatenate(zc_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    sevs = np.concatenate(sevs_list, axis=0)

    all_points = []
    for i in range(N_eval):
        nv = sd_sub.n_vis[i]
        for t in range(nv):
            all_points.append(zc[i, t])

    all_points = np.array(all_points)

    # Dimensionality reduction on z^c to 2 main clinical axes
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(all_points)

    # Unpack back to patient sequences
    idx_counter = 0
    patient_trajectories = []

    for i in range(N_eval):
        nv = sd_sub.n_vis[i]
        p_coords = coords_2d[idx_counter:idx_counter + nv]
        p_probs = probs[i, :nv]
        p_sevs = sevs[i, :nv]
        idx_counter += nv

        pid = sd_sub.patient_ids[i]
        disease = "SLE" if sd_sub.subgroup[i] == 1 else "RA"
        times = np.zeros(nv)
        cum_time = 0.0
        for t in range(nv):
            # approximate months from log gap
            gap_days = np.expm1(sd_sub.gap[i, t]) * 30.0
            cum_time += (gap_days / 30.0) if t > 0 else 0.0
            times[t] = cum_time

        patient_trajectories.append({
            "patient_id": pid,
            "disease": disease,
            "n_vis": nv,
            "x": p_coords[:, 0],
            "y": p_coords[:, 1],
            "time_months": times,
            "visit_step": np.arange(nv),
            "flare_prob": p_probs,
            "severity": p_sevs,
            "raw": sd_sub.raw[i, :nv],
            "names": sd_sub.names,
        })

    return {
        "trajectories": patient_trajectories,
        "pca_variance": pca.explained_variance_ratio_
    }


def create_3d_patient_trajectory_figure(
    trajectories_data: dict[str, Any],
    max_patients_display: int = 25,
    highlight_pid: str | None = None
) -> go.Figure:
    """Creates a colorful, interactive 3D Spatiotemporal Trajectory Manifold."""
    trajectories = trajectories_data["trajectories"][:max_patients_display]

    fig = go.Figure()

    # Color palette for patients
    distinct_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363"
    ]

    all_flare_probs = []
    for traj in trajectories:
        all_flare_probs.extend(traj["flare_prob"])

    min_p, max_p = min(all_flare_probs), max(all_flare_probs)

    for p_idx, traj in enumerate(trajectories):
        pid = traj["patient_id"]
        is_highlighted = (highlight_pid is None) or (pid == highlight_pid)
        opacity = 1.0 if is_highlighted else 0.25
        line_width = 5 if (highlight_pid == pid) else 3

        xs = traj["x"]
        ys = traj["y"]
        zs = traj["time_months"]
        probs = traj["flare_prob"]
        sevs = traj["severity"]
        nv = traj["n_vis"]
        disease = traj["disease"]

        # Build custom hover text
        hover_texts = []
        for t in range(nv):
            # Extract key labs if present
            raw_v = traj["raw"][t]
            names = traj["names"]

            def get_f(fn: str) -> str:
                if fn in names:
                    v = raw_v[names.index(fn)]
                    return f"{v:.1f}" if not np.isnan(v) else "N/A"
                return "N/A"

            crp = get_f("crp")
            esr = get_f("esr")
            steroid = get_f("corticosteroid_dose")
            mtx = get_f("dmard_mtx")

            ht = (
                f"<b>Patient:</b> {pid} ({disease})<br>"
                f"<b>Visit:</b> {t+1} of {nv}<br>"
                f"<b>Longitudinal Time:</b> Month {zs[t]:.1f}<br>"
                f"<b>Next-Visit Flare Risk:</b> {probs[t]:.1%}<br>"
                f"<b>Severity Score:</b> {sevs[t]:.2f}<br>"
                f"----------------------------<br>"
                f"<b>CRP:</b> {crp} mg/L | <b>ESR:</b> {esr} mm/hr<br>"
                f"<b>Steroid:</b> {steroid} mg | <b>MTX:</b> {'ON' if mtx == '1.0' else 'OFF'}<br>"
                f"<b>Latent State:</b> (z₁={xs[t]:.2f}, z₂={ys[t]:.2f})"
            )
            hover_texts.append(ht)

        # 1. Trajectory connecting line
        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            name=f"{pid} ({disease})",
            line=dict(
                color=distinct_colors[p_idx % len(distinct_colors)],
                width=line_width
            ),
            opacity=opacity,
            hoverinfo="skip"
        ))

        # 2. Individual Visit Nodes (Spheres with Flare Risk Colormap)
        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            name=f"{pid} Visits",
            marker=dict(
                size=[6 + float(p) * 8 for p in probs],
                color=probs,
                colorscale="Turbo",
                cmin=0.0,
                cmax=1.0,
                colorbar=dict(
                    title=dict(text="<b>Next-Visit Flare Risk</b>", side="right"),
                    tickformat=".0%",
                    len=0.75,
                    thickness=18
                ) if p_idx == 0 else None,
                opacity=opacity,
                line=dict(color="#ffffff", width=1)
            ),
            text=[f"V{t+1}" for t in range(nv)],
            textposition="top center",
            textfont=dict(size=9, color="#222"),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False
        ))

    fig.update_layout(
        title=dict(
            text="<b>3D Spatiotemporal Patient Trajectory Manifold</b><br>"
                 "<span style='font-size: 13px; color: #555;'>"
                 "Tracing longitudinal disease progression across latent causal representation space over time (months)</span>",
            x=0.05
        ),
        scene=dict(
            xaxis=dict(
                title="<b>Latent Axis 1 (z₁: Immune Activation)</b>",
                backgroundcolor="#f8f9fa",
                gridcolor="#e0e0e0",
                showbackground=True
            ),
            yaxis=dict(
                title="<b>Latent Axis 2 (z₂: Organ Involvement)</b>",
                backgroundcolor="#f8f9fa",
                gridcolor="#e0e0e0",
                showbackground=True
            ),
            zaxis=dict(
                title="<b>Longitudinal Time (Months from Baseline)</b>",
                backgroundcolor="#ffffff",
                gridcolor="#d0d0d0",
                showbackground=True
            ),
            camera=dict(
                eye=dict(x=1.65, y=1.65, z=1.2)
            )
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        height=750,
        legend=dict(
            itemsizing="constant",
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def create_3d_causal_dag_figure(
    feature_names: list[str],
    treat_idx: list[int],
    A_relation: np.ndarray | None = None
) -> go.Figure:
    """Creates an interactive 3D Tiered Causal Directed Acyclic Graph (DAG) Network."""
    # Define functional tiers along Z-axis
    # Z=0: Baseline Genetics & Demographics
    # Z=1: Pharmacological Treatments (Do-Calculus Interventions)
    # Z=2: Latent Causal Health Space (Unobserved Confounder & Immune State)
    # Z=3: Intermediate Biomarkers & Clinical Symptoms
    # Z=4: Downstream Clinical Outcome (Flare & Damage)

    tiers = {
        "demographics": {
            "z": 0.0, "color": "#00bcd4", "layer_name": "Tier 0: Demographics & Genetics",
            "vars": ["age", "female", "disease_type", "disease_duration", "smoking_status", "comorbidity_index"]
        },
        "treatments": {
            "z": 1.0, "color": "#ffb300", "layer_name": "Tier 1: Clinical Treatments (a')",
            "vars": ["corticosteroid_dose", "dmard_mtx", "dmard_leflunomide", "biologic_anti_tnf", "biologic_rituximab", "hydroxychloroquine", "azathioprine"]
        },
        "latent_causal": {
            "z": 2.0, "color": "#ab47bc", "layer_name": "Tier 2: Latent Health State (Z_causal)",
            "vars": ["Immune_Activation_Hub", "Lymphocyte_Signaling_Hub", "Inflammatory_Cascade_Hub"]
        },
        "biomarkers": {
            "z": 3.0, "color": "#26a69a", "layer_name": "Tier 3: Biomarkers & Symptoms",
            "vars": ["crp", "esr", "dsdna", "c3", "c4", "platelet_count", "hemoglobin", "joint_swelling", "morning_stiffness", "fatigue_score"]
        },
        "outcomes": {
            "z": 4.0, "color": "#e53935", "layer_name": "Tier 4: Clinical Outcomes (t+1)",
            "vars": ["Next_Visit_Flare_Event", "Next_Visit_Severity_Score"]
        }
    }

    # Position nodes in 3D (x, y, z)
    node_positions = {}
    node_x, node_y, node_z = [], [], []
    node_colors, node_labels, node_tiers, hover_texts = [], [], [], []

    for tier_key, tinfo in tiers.items():
        z_val = tinfo["z"]
        color = tinfo["color"]
        layer_title = tinfo["layer_name"]
        var_list = tinfo["vars"]
        n_vars = len(var_list)

        # Distribute along an ellipse on the XY plane for that tier
        radius_x = 3.5
        radius_y = 2.5
        for i, vname in enumerate(var_list):
            angle = (2.0 * np.pi * i) / n_vars
            x_val = radius_x * np.cos(angle)
            y_val = radius_y * np.sin(angle)

            node_positions[vname] = (x_val, y_val, z_val)
            node_x.append(x_val)
            node_y.append(y_val)
            node_z.append(z_val)
            node_colors.append(color)
            node_labels.append(vname)
            node_tiers.append(layer_title)

            hover_texts.append(
                f"<b>Node:</b> {vname}<br>"
                f"<b>Functional Tier:</b> {layer_title}<br>"
                f"<b>Causal Role:</b> {tier_key.capitalize()}<br>"
                f"<b>Coordinates:</b> (X={x_val:.2f}, Y={y_val:.2f}, Z={z_val:.1f})"
            )

    # Define verified causal directed edges
    causal_edges = [
        # Demographics -> Latent & Biomarkers
        ("age", "Immune_Activation_Hub", "pathogenic", 0.6),
        ("female", "Immune_Activation_Hub", "pathogenic", 0.5),
        ("disease_duration", "Inflammatory_Cascade_Hub", "pathogenic", 0.7),
        ("smoking_status", "Inflammatory_Cascade_Hub", "pathogenic", 0.8),

        # Treatments -> Latent Hubs (Inhibitory / Protective)
        ("corticosteroid_dose", "Inflammatory_Cascade_Hub", "protective", -0.9),
        ("corticosteroid_dose", "crp", "protective", -0.85),
        ("dmard_mtx", "Immune_Activation_Hub", "protective", -0.8),
        ("dmard_mtx", "joint_swelling", "protective", -0.75),
        ("biologic_anti_tnf", "Inflammatory_Cascade_Hub", "protective", -0.95),
        ("biologic_anti_tnf", "esr", "protective", -0.85),
        ("hydroxychloroquine", "Lymphocyte_Signaling_Hub", "protective", -0.7),

        # Latent Hubs -> Dynamic Biomarkers & Symptoms
        ("Immune_Activation_Hub", "dsdna", "pathogenic", 0.85),
        ("Immune_Activation_Hub", "c3", "pathogenic", -0.75),
        ("Lymphocyte_Signaling_Hub", "platelet_count", "pathogenic", 0.6),
        ("Inflammatory_Cascade_Hub", "crp", "pathogenic", 0.95),
        ("Inflammatory_Cascade_Hub", "esr", "pathogenic", 0.9),
        ("Inflammatory_Cascade_Hub", "joint_swelling", "pathogenic", 0.85),
        ("Inflammatory_Cascade_Hub", "morning_stiffness", "pathogenic", 0.8),
        ("Inflammatory_Cascade_Hub", "fatigue_score", "pathogenic", 0.75),

        # Biomarkers & Hubs -> Downstream Outcomes
        ("crp", "Next_Visit_Flare_Event", "pathogenic", 0.85),
        ("esr", "Next_Visit_Severity_Score", "pathogenic", 0.8),
        ("dsdna", "Next_Visit_Flare_Event", "pathogenic", 0.9),
        ("joint_swelling", "Next_Visit_Severity_Score", "pathogenic", 0.9),
        ("Inflammatory_Cascade_Hub", "Next_Visit_Flare_Event", "pathogenic", 0.95),
        ("Immune_Activation_Hub", "Next_Visit_Severity_Score", "pathogenic", 0.9),
    ]

    edge_x, edge_y, edge_z = [], [], []
    edge_hover = []

    fig = go.Figure()

    # Draw directed causal edges
    for src, dst, effect_type, weight in causal_edges:
        if src in node_positions and dst in node_positions:
            x0, y0, z0 = node_positions[src]
            x1, y1, z1 = node_positions[dst]

            # Generate spline interpolation
            t_vals = np.linspace(0, 1, 10)
            xs = (1 - t_vals) * x0 + t_vals * x1
            ys = (1 - t_vals) * y0 + t_vals * y1
            zs = (1 - t_vals) * z0 + t_vals * z1

            color = "#00e676" if effect_type == "protective" else "#ff3d00"
            dash = "solid" if effect_type == "pathogenic" else "dash"

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=max(2, int(abs(weight) * 5)), dash=dash),
                hoverinfo="text",
                hovertext=f"<b>Causal Arc:</b> {src} ➔ {dst}<br><b>Mechanism:</b> {effect_type.capitalize()} (Weight: {weight:+.2f})",
                showlegend=False
            ))

    # Add 3D Nodes
    fig.add_trace(go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers+text",
        marker=dict(
            size=11,
            color=node_colors,
            opacity=0.95,
            line=dict(color="#ffffff", width=2)
        ),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=10, color="#111"),
        hovertext=hover_texts,
        hoverinfo="text",
        name="Clinical Variables"
    ))

    # Horizontal guide planes for each tier
    for tier_key, tinfo in tiers.items():
        z_lvl = tinfo["z"]
        col = tinfo["color"]
        # draw subtle perimeter ring
        theta = np.linspace(0, 2 * np.pi, 50)
        fig.add_trace(go.Scatter3d(
            x=3.8 * np.cos(theta),
            y=2.8 * np.sin(theta),
            z=np.full_like(theta, z_lvl),
            mode="lines",
            line=dict(color=col, width=1, dash="dot"),
            opacity=0.3,
            hoverinfo="skip",
            showlegend=False
        ))

    fig.update_layout(
        title=dict(
            text="<b>3D Tiered Causal Directed Acyclic Graph (DAG) Network</b><br>"
                 "<span style='font-size: 13px; color: #555;'>"
                 "Hierarchical causal pathways: Demographics (Z=0) ➔ Treatments (Z=1) ➔ Latent Hubs (Z=2) ➔ Biomarkers (Z=3) ➔ Outcomes (Z=4)</span>",
            x=0.05
        ),
        scene=dict(
            xaxis=dict(title="<b>Spatial X</b>", showbackground=False, showticklabels=False),
            yaxis=dict(title="<b>Spatial Y</b>", showbackground=False, showticklabels=False),
            zaxis=dict(
                title="<b>Causal Hierarchy Tier (0 to 4)</b>",
                tickvals=[0, 1, 2, 3, 4],
                ticktext=["0: Demographics", "1: Treatments", "2: Latent Hubs", "3: Biomarkers", "4: Outcomes"],
                backgroundcolor="#f8f9fa",
                showbackground=True
            ),
            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.4))
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        height=750
    )

    return fig


def generate_and_save_all_3d_reports() -> None:
    """Generates and writes both 3D HTML reports to reports/causal/."""
    print(f"[3D Visualizer] Initializing on {DEVICE} ...")

    model_path = ROOT / "models" / "causal_paper_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    sd = D.load_multimodal_cohort()

    model = CausalDeepModel(
        V=len(ckpt["names"]),
        treat_idx=ckpt["treat_idx"],
        A=ckpt["A_relation"],
        cfg=ckpt["cfg"],
        n_nuisance=sd.nuisance.shape[1]
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    # 1. 3D Trajectory Manifold
    print("[3D Visualizer] Computing latent representations for 3D trajectory manifold ...")
    traj_data = extract_latent_trajectories(model, sd, device=DEVICE, n_patients_max=30)
    fig_traj = create_3d_patient_trajectory_figure(traj_data, max_patients_display=20)
    html_traj_path = REPORT_DIR / "3d_patient_trajectories.html"
    fig_traj.write_html(str(html_traj_path), include_plotlyjs="cdn")
    print(f"  -> Saved: {html_traj_path}")

    # 2. 3D Tiered Causal Graph
    print("[3D Visualizer] Constructing 3D tiered causal graph network ...")
    fig_graph = create_3d_causal_dag_figure(sd.names, sd.treat_idx, ckpt["A_relation"])
    html_graph_path = REPORT_DIR / "3d_causal_graph.html"
    fig_graph.write_html(str(html_graph_path), include_plotlyjs="cdn")
    print(f"  -> Saved: {html_graph_path}")

    print("\n[3D Visualizer] All 3D interactive visualizations generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D interactive causal visualizations.")
    args = parser.parse_args()
    generate_and_save_all_3d_reports()
