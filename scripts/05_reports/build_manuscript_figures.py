from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


FIG_DIR = Path("results/manuscript/figures")
SOURCE_DIR = Path("results/manuscript/source_data")
AI_ARCHITECTURE_IMAGE = Path("docs/figures/fig1a_openai_flowchart.png")

PALETTE = {
    "signal_features": "#2f6f9f",
    "ssl_embeddings": "#b79ad7",
    "signal_plus_ssl": "#8a6bb8",
    "signal_plus_ssl_plus_signal_phenotype": "#c45a3d",
    "clinical_signal": "#2f6f9f",
    "ssl_embedding": "#8a6bb8",
    "phenotype": "#c45a3d",
    "raw": "#777777",
    "platt": "#2f6f9f",
    "isotonic": "#c45a3d",
    "event": "#c45a3d",
    "nonevent": "#9aa6b2",
}

MODEL_LABELS = {
    "signal_features": "Classical signal\nfeatures",
    "ssl_embeddings": "SSL embeddings",
    "signal_plus_ssl": "Signal + SSL",
    "signal_plus_ssl_plus_signal_phenotype": "Signal + SSL\n+ phenotype",
    "signal_phenotype_only": "Signal phenotype",
    "ssl_phenotype_only": "SSL phenotype",
    "signal_plus_signal_phenotype": "Signal + signal\nphenotype",
    "signal_plus_ssl_phenotype": "Signal + SSL\nphenotype",
}


def setup_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "legend.frameon": False,
        "figure.dpi": 150,
    })


def save_pub(fig, stem, dpi=600):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / stem
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def add_panel_label(ax, label, x=-0.08, y=1.06):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
    )


def read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)


def write_source_data(name, frames):
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    out = SOURCE_DIR / name
    rows = []
    for panel, frame in frames:
        temp = frame.copy()
        temp.insert(0, "source_panel", panel)
        rows.append(temp)
    pd.concat(rows, ignore_index=True, sort=False).to_csv(out, index=False)
    return out


def model_label(model):
    return MODEL_LABELS.get(model, model)


def model_color(model):
    return PALETTE.get(model, "#555555")


def roc_pr_points(y_true, score):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.r_[0, np.cumsum(y_sorted)]
    fp = np.r_[0, np.cumsum(1 - y_sorted)]
    positives = max(1, y.sum())
    negatives = max(1, len(y) - y.sum())
    recall = tp / positives
    fpr = fp / negatives
    tpr = recall
    precision = np.divide(tp, tp + fp, out=np.full_like(tp, y.mean(), dtype=float), where=(tp + fp) > 0)
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    pr = pd.DataFrame({"recall": recall, "precision": precision})
    return roc, pr


def curve_source(preds, models):
    frames = []
    for model in models:
        sub = preds[preds["model"] == model].copy()
        roc, pr = roc_pr_points(sub["y_true"], sub["probability"])
        roc.insert(0, "model", model)
        roc.insert(1, "curve", "roc")
        pr.insert(0, "model", model)
        pr.insert(1, "curve", "precision_recall")
        frames.extend([roc, pr])
    return pd.concat(frames, ignore_index=True, sort=False)


def crop_white(img, pad=8):
    rgb = img[..., :3]
    mask = np.any(rgb < 0.985, axis=2)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(img.shape[0], y1 + pad)
    x1 = min(img.shape[1], x1 + pad)
    return img[y0:y1, x0:x1]


def draw_workflow(ax):
    ax.axis("off")
    boxes = [
        ("Open CTG data\nCTU-UHB, CTGDL, UCI", 0.04, 0.76, 0.42, "#eff6fb"),
        ("Record split\n386 train / 166 test", 0.04, 0.57, 0.42, "#f8fbfd"),
        ("FHR + UC windows\nmasked encoder", 0.04, 0.38, 0.42, "#f8fbfd"),
        ("Dynamic phenotypes\nsignal + SSL", 0.04, 0.19, 0.42, "#f8fbfd"),
        ("Prediction\nCI, calibration, DCA", 0.58, 0.38, 0.38, "#fff3ed"),
    ]
    for text, x, y, width, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), width, 0.12, facecolor=color, edgecolor="#333333", linewidth=0.9))
        ax.text(x + width / 2, y + 0.06, text, ha="center", va="center", fontsize=6.6)
    arrows = [
        ((0.25, 0.76), (0.25, 0.69)),
        ((0.25, 0.57), (0.25, 0.50)),
        ((0.25, 0.38), (0.25, 0.31)),
        ((0.46, 0.44), (0.58, 0.44)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"))


def draw_ai_architecture(ax):
    ax.axis("off")
    if AI_ARCHITECTURE_IMAGE.exists():
        img = crop_white(plt.imread(AI_ARCHITECTURE_IMAGE), pad=4)
        ax.imshow(img)
        ax.set_anchor("C")
        return

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, color, fontsize=6.1, weight="normal"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=color,
            edgecolor="#2f3437",
            linewidth=0.85,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color="#1f2529",
        )
        return (x, y, w, h)

    def arrow(start, end, style="-|>", rad=0.0):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle=style,
                lw=0.85,
                color="#2f3437",
                shrinkA=3,
                shrinkB=3,
                connectionstyle=f"arc3,rad={rad}",
            ),
        )

    input_box = box(0.025, 0.57, 0.145, 0.27, "FHR + UC\nwindow", "#eff6fb", weight="bold")
    patch_box = box(0.225, 0.57, 0.135, 0.27, "Patch\ntokens", "#f8fbfd")
    mask_box = box(0.410, 0.57, 0.135, 0.27, "Mixed\nmasking", "#f8fbfd")
    encoder_box = box(0.595, 0.57, 0.150, 0.27, "Transformer\nencoder", "#efe8f7", weight="bold")
    embed_box = box(0.825, 0.57, 0.140, 0.27, "Record\nembedding", "#efe8f7", weight="bold")
    recon_box = box(0.495, 0.18, 0.190, 0.17, "Masked\nreconstruction\nloss", "#fff3ed")
    contrast_box = box(0.265, 0.18, 0.190, 0.17, "NT-Xent\ncontrastive\nloss", "#fff3ed")
    phenotype_box = box(0.755, 0.18, 0.100, 0.17, "Phenotype\nclustering", "#f7eee8")
    risk_box = box(0.875, 0.18, 0.100, 0.17, "Risk\nmodel", "#f7eee8")

    t = np.linspace(0, 1, 70)
    wave_x = input_box[0] + 0.018 + t * (input_box[2] - 0.036)
    ax.plot(wave_x, input_box[1] + 0.205 + 0.018 * np.sin(2 * np.pi * 2.2 * t), color="#2f6f9f", lw=0.9)
    ax.plot(wave_x, input_box[1] + 0.075 + 0.012 * np.sin(2 * np.pi * 1.3 * t), color="#c45a3d", lw=0.9)

    for i in range(5):
        ax.add_patch(plt.Rectangle((0.243 + i * 0.020, 0.602), 0.014, 0.060, facecolor="#dbe7f3", edgecolor="#62798c", lw=0.45))
    for i in range(5):
        color = "#dbe7f3" if i not in (1, 3) else "#f1b2a4"
        ax.add_patch(plt.Rectangle((0.428 + i * 0.020, 0.602), 0.014, 0.060, facecolor=color, edgecolor="#62798c", lw=0.45))
        if i in (1, 3):
            ax.text(0.435 + i * 0.020, 0.633, "M", ha="center", va="center", fontsize=4.5, color="#7a2f24")

    for i in range(3):
        y = 0.602 + i * 0.023
        ax.add_patch(plt.Rectangle((0.622, y), 0.096, 0.015, facecolor="#ffffff", edgecolor="#8a6bb8", lw=0.45))
    for i in range(4):
        ax.add_patch(plt.Circle((0.855 + i * 0.024, 0.615 + (i % 2) * 0.025), 0.008, facecolor="#8a6bb8", edgecolor="white", lw=0.3))

    arrow((0.170, 0.705), (0.225, 0.705))
    arrow((0.360, 0.705), (0.410, 0.705))
    arrow((0.545, 0.705), (0.595, 0.705))
    arrow((0.745, 0.705), (0.825, 0.705))
    arrow((0.662, 0.57), (0.590, 0.35), rad=0.05)
    arrow((0.635, 0.57), (0.360, 0.35), rad=0.12)
    arrow((0.685, 0.265), (0.825, 0.625), rad=-0.15)
    arrow((0.455, 0.265), (0.825, 0.625), rad=-0.25)
    arrow((0.895, 0.57), (0.805, 0.35))
    arrow((0.895, 0.57), (0.925, 0.35))

    ax.text(0.594, 0.405, "joint SSL objective", ha="center", va="center", fontsize=5.8, color="#555555")
    ax.text(
        0.5,
        0.055,
        "Self-supervised representation learning links masked reconstruction and contrastive consistency to downstream phenotyping and risk prediction.",
        ha="center",
        va="center",
        fontsize=6.2,
        color="#333333",
    )


def read_loss_curve(path):
    rows = []
    metrics_path = Path(path)
    if not metrics_path.exists():
        return pd.DataFrame(columns=["epoch", "total", "recon", "contrastive"])
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or not line.startswith("epoch_"):
            continue
        epoch_part, rest = line.split("_total=", 1)
        epoch = int(epoch_part.replace("epoch_", ""))
        parts = {"total": rest.split(",")[0]}
        for item in rest.split(",")[1:]:
            key, value = item.split("=", 1)
            parts[key] = value
        rows.append({
            "epoch": epoch,
            "total": float(parts.get("total", np.nan)),
            "recon": float(parts.get("recon", np.nan)),
            "contrastive": float(parts.get("contrastive", np.nan)),
        })
    return pd.DataFrame(rows)


def figure_1():
    split = read_csv("data/processed/ctu_uhb_record_split.csv")
    ctu_windows = read_csv("data/processed/ctu_uhb_window_index.csv")
    ctgdl_windows = read_csv("data/processed/ctgdl_fhrma_window_index.csv")
    feature_rows = read_csv("data/processed/ctu_uhb_record_features.csv")
    feature_split = feature_rows.merge(split[["record_id", "split"]], on="record_id", how="left")
    ctu_windows_split = ctu_windows.merge(split[["record_id", "split"]], on="record_id", how="left")

    counts = pd.DataFrame([
        {"item": "CTU-UHB records", "value": len(split), "type": "records"},
        {"item": "CTU-UHB train records", "value": int((split["split"] == "train").sum()), "type": "records"},
        {"item": "CTU-UHB test records", "value": int((split["split"] == "test").sum()), "type": "records"},
        {"item": "CTU-UHB windows", "value": len(ctu_windows), "type": "windows"},
        {"item": "CTU-UHB train windows", "value": int((ctu_windows_split["split"] == "train").sum()), "type": "windows"},
        {"item": "CTU-UHB test windows", "value": int((ctu_windows_split["split"] == "test").sum()), "type": "windows"},
        {"item": "CTGDL-FHRMA records", "value": int(ctgdl_windows["record_id"].nunique()), "type": "records"},
        {"item": "CTGDL-FHRMA windows", "value": len(ctgdl_windows), "type": "windows"},
    ])
    split_events = (
        feature_split.groupby("split")["neonatal_risk"]
        .agg(records="size", events="sum", event_rate="mean")
        .reset_index()
    )
    split_events["non_events"] = split_events["records"] - split_events["events"]
    evidence = pd.DataFrame([
        {"module": "Self-supervised pretraining", "Outcome": 0, "Interpretation": 0, "Robustness": 1, "External": 0},
        {"module": "Held-out test prediction", "Outcome": 1, "Interpretation": 0, "Robustness": 1, "External": 0},
        {"module": "Dynamic phenotypes", "Outcome": 1, "Interpretation": 1, "Robustness": 0, "External": 0},
        {"module": "Missingness sensitivity", "Outcome": 1, "Interpretation": 0, "Robustness": 1, "External": 0},
        {"module": "Recalibration + DCA", "Outcome": 1, "Interpretation": 0, "Robustness": 1, "External": 0},
        {"module": "CTGDL representation test", "Outcome": 0, "Interpretation": 0, "Robustness": 1, "External": 1},
    ])
    architecture = pd.DataFrame({
        "step_order": np.arange(1, 9),
        "step": [
            "FHR/UC window",
            "Patch tokens",
            "Mixed masking",
            "Transformer encoder",
            "Masked reconstruction",
            "NT-Xent contrastive",
            "Record embedding",
            "Phenotype/risk model",
        ],
    })
    architecture["panel_asset"] = AI_ARCHITECTURE_IMAGE.as_posix()
    write_source_data("Figure_1_source_data.csv", [
        ("A", architecture),
        ("C", counts),
        ("D", split_events),
        ("E", evidence),
    ])

    fig = plt.figure(figsize=(7.2, 8.8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.12, 1.0], height_ratios=[1.42, 1.04, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    draw_ai_architecture(ax_a)
    if not AI_ARCHITECTURE_IMAGE.exists():
        ax_a.set_title("PatchTST-style self-supervised CTG representation learning", loc="left", fontsize=8, fontweight="bold")
    add_panel_label(ax_a, "A", x=-0.02, y=1.06)

    draw_workflow(ax_b)
    ax_b.set_title("Study design and evidence chain", loc="left", fontsize=8, fontweight="bold")
    add_panel_label(ax_b, "B")

    bar_items = counts[counts["item"].isin([
        "CTU-UHB records",
        "CTU-UHB train records",
        "CTU-UHB test records",
        "CTGDL-FHRMA records",
    ])].copy()
    ax_c.barh(bar_items["item"], bar_items["value"], color="#4c78a8")
    ax_c.set_xlabel("Records")
    ax_c.set_title("Cohort scale", loc="left", fontsize=8, fontweight="bold")
    ax_c.grid(axis="x", alpha=0.25)
    add_panel_label(ax_c, "C")

    order = ["train", "test"]
    ev = split_events.set_index("split").loc[order].reset_index()
    x = np.arange(len(order))
    ax_d.bar(x, ev["non_events"], label="No event", color=PALETTE["nonevent"])
    ax_d.bar(x, ev["events"], bottom=ev["non_events"], label="Neonatal risk", color=PALETTE["event"])
    for i, row in ev.iterrows():
        ax_d.text(i, row["records"] + 8, f"{row['event_rate']:.1%}", ha="center", va="bottom", fontsize=6)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(["Train", "Test"])
    ax_d.set_ylabel("Records")
    ax_d.set_title("Outcome balance", loc="left", fontsize=8, fontweight="bold")
    ax_d.legend(frameon=False, fontsize=6, loc="upper right")
    ax_d.grid(axis="y", alpha=0.25)
    add_panel_label(ax_d, "D")

    matrix = evidence[["Outcome", "Interpretation", "Robustness", "External"]].to_numpy()
    ax_e.imshow(matrix, cmap=mpl.colors.ListedColormap(["#f2f2f2", "#4c78a8"]), aspect="auto", vmin=0, vmax=1)
    ax_e.set_xticks(np.arange(4))
    ax_e.set_xticklabels(["Outcome", "Interp.", "Robust", "External"], fontsize=5.6, rotation=20, ha="right")
    ax_e.set_yticks(np.arange(len(evidence)))
    ax_e.set_yticklabels(evidence["module"], fontsize=5.5)
    ax_e.set_title("Evidence map", loc="left", fontsize=7, fontweight="bold")
    ax_e.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax_e.set_yticks(np.arange(-0.5, len(evidence), 1), minor=True)
    ax_e.grid(which="minor", color="white", linewidth=0.8)
    ax_e.tick_params(which="minor", bottom=False, left=False)
    for spine in ax_e.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
    add_panel_label(ax_e, "E", x=-0.09, y=1.06)

    fig.tight_layout(w_pad=1.35, h_pad=1.15)
    save_pub(fig, "Figure_1_study_design_pipeline")
    plt.close(fig)


def figure_2():
    boot = read_csv("results/tables/bootstrap_validation_ci.csv")
    diffs = read_csv("results/tables/bootstrap_model_differences.csv")
    missing = read_csv("results/tables/missingness_sensitivity_metrics.csv")
    importance = read_csv("results/tables/permutation_feature_importance.csv")
    family = read_csv("results/tables/permutation_feature_family_importance.csv")
    preds = read_csv("results/tables/manuscript_test_predictions.csv")
    models = ["signal_features", "ssl_embeddings", "signal_plus_ssl", "signal_plus_ssl_plus_signal_phenotype"]
    curves = curve_source(preds, models)
    write_source_data("Figure_2_source_data.csv", [
        ("A_B", curves),
        ("C", boot[boot["model"].isin(models) & boot["metric"].isin(["auroc", "auprc"])]),
        ("D", diffs[diffs["metric"] == "auprc"]),
        ("E", missing[missing["model"].isin(models) & (missing["metric"] == "auprc")]),
        ("F_features", importance[importance["model"] == "signal_plus_ssl"].sort_values("importance_mean_auprc", ascending=False).head(12)),
        ("F_family", family[family["model"].isin(["signal_plus_ssl", "signal_plus_ssl_plus_signal_phenotype"])]),
    ])

    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.05])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    prevalence = preds.drop_duplicates(["record_id"])["y_true"].mean()
    for model in models:
        sub_pr = curves[(curves["model"] == model) & (curves["curve"] == "precision_recall")]
        ax_a.plot(sub_pr["recall"], sub_pr["precision"], color=model_color(model), label=model_label(model))
    ax_a.axhline(prevalence, color="#999999", linestyle="--", linewidth=0.8, label="Prevalence")
    ax_a.set_xlabel("Recall")
    ax_a.set_ylabel("Precision")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_title("Classical and SSL model precision-recall", loc="left", fontsize=8, fontweight="bold")
    ax_a.legend(fontsize=5.3, loc="upper right")
    ax_a.grid(alpha=0.25)
    add_panel_label(ax_a, "A")

    for model in models:
        sub_roc = curves[(curves["model"] == model) & (curves["curve"] == "roc")]
        ax_b.plot(sub_roc["fpr"], sub_roc["tpr"], color=model_color(model), label=model_label(model))
    ax_b.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=0.8)
    ax_b.set_xlabel("False-positive rate")
    ax_b.set_ylabel("True-positive rate")
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.set_title("Classical and SSL model ROC", loc="left", fontsize=8, fontweight="bold")
    ax_b.grid(alpha=0.25)
    add_panel_label(ax_b, "B")

    plot = boot[boot["model"].isin(models) & boot["metric"].isin(["auroc", "auprc"])].copy()
    y_base = np.arange(len(models))
    offsets = {"auroc": -0.13, "auprc": 0.13}
    colors = {"auroc": "#4c78a8", "auprc": "#f58518"}
    for metric in ["auroc", "auprc"]:
        sub = plot[plot["metric"] == metric].set_index("model").loc[models].reset_index()
        y = y_base + offsets[metric]
        ax_c.errorbar(
            sub["point"],
            y,
            xerr=[sub["point"] - sub["ci_2_5"], sub["ci_97_5"] - sub["point"]],
            fmt="o",
            color=colors[metric],
            capsize=2,
            label=metric.upper(),
        )
    ax_c.set_yticks(y_base)
    ax_c.set_yticklabels([model_label(m) for m in models])
    ax_c.set_xlabel("Held-out performance")
    ax_c.set_title("Bootstrap validation by AI feature set", loc="left", fontsize=8, fontweight="bold")
    ax_c.grid(axis="x", alpha=0.25)
    ax_c.legend(
        frameon=False,
        fontsize=6,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.18),
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.0,
        columnspacing=0.9,
    )
    add_panel_label(ax_c, "C")

    sub = diffs[
        (diffs["metric"] == "auprc")
        & diffs["comparison"].isin([
            "signal_plus_ssl minus signal_features",
            "signal_plus_ssl_plus_signal_phenotype minus signal_features",
        ])
    ].copy()
    labels = sub["comparison"].str.replace("signal_plus_ssl_plus_signal_phenotype", "Signal + SSL + phenotype", regex=False)
    labels = labels.str.replace("signal_plus_ssl", "Signal + SSL", regex=False)
    labels = labels.str.replace("signal_features", "Classical signal", regex=False)
    labels = labels.str.replace(" minus ", "\nminus ", regex=False)
    y = np.arange(len(sub))
    ax_d.errorbar(
        sub["bootstrap_mean_difference"],
        y,
        xerr=[sub["bootstrap_mean_difference"] - sub["ci_2_5"], sub["ci_97_5"] - sub["bootstrap_mean_difference"]],
        fmt="o",
        color="#8a6bb8",
        capsize=2,
    )
    ax_d.axvline(0, color="black", linewidth=0.8)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(labels, fontsize=6)
    ax_d.set_xlabel("AUPRC difference")
    ax_d.set_title("Incremental SSL enrichment", loc="left", fontsize=8, fontweight="bold")
    ax_d.grid(axis="x", alpha=0.25)
    add_panel_label(ax_d, "D")

    missing_models = ["signal_features", "signal_plus_ssl", "signal_plus_ssl_plus_signal_phenotype"]
    for idx, scenario in enumerate(["full_features", "no_missingness_features"]):
        sub = missing[(missing["scenario"] == scenario) & (missing["metric"] == "auprc")].set_index("model").loc[missing_models].reset_index()
        x = np.arange(len(missing_models)) + (idx - 0.5) * 0.25
        label = "Full features" if scenario == "full_features" else "No missingness"
        ax_e.bar(x, sub["point"], width=0.25, label=label, color=["#4c78a8", "#f58518"][idx])
    ax_e.set_xticks(np.arange(len(missing_models)))
    ax_e.set_xticklabels([model_label(m) for m in missing_models], rotation=25, ha="right")
    ax_e.set_ylabel("AUPRC")
    ax_e.set_title("Missingness sensitivity", loc="left", fontsize=8, fontweight="bold")
    ax_e.legend(frameon=False, fontsize=6)
    ax_e.grid(axis="y", alpha=0.25)
    add_panel_label(ax_e, "E")

    top = importance[importance["model"] == "signal_plus_ssl"].sort_values("importance_mean_auprc", ascending=False).head(8)
    top = top.sort_values("importance_mean_auprc", ascending=True)
    colors = top["feature_family"].map(PALETTE).fillna("#777777")
    ax_f.barh(top["feature_label"], top["importance_mean_auprc"], color=colors)
    ssl_sum = family[(family["model"] == "signal_plus_ssl") & (family["feature_family"] == "ssl_embedding")]["positive_importance_sum_auprc"]
    if len(ssl_sum):
        ax_f.text(
            0.98,
            0.04,
            f"SSL embedding family\nsum={float(ssl_sum.iloc[0]):.2f}",
            transform=ax_f.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#dddddd", linewidth=0.4),
        )
    ax_f.set_xlabel("AUPRC importance")
    ax_f.set_title("AI embedding contribution", loc="left", fontsize=8, fontweight="bold")
    ax_f.grid(axis="x", alpha=0.25)
    add_panel_label(ax_f, "F")

    fig.tight_layout(h_pad=1.4, w_pad=1.4)
    save_pub(fig, "Figure_2_model_validation_and_explanation")
    plt.close(fig)


def figure_3():
    signal = read_csv("results/tables/ctu_uhb_signal_phenotype_summary.csv")
    proto = read_csv("results/tables/phenotype_prototype_records.csv")
    metrics = [
        ("neonatal_risk_mean", "Risk"),
        ("cord_ph_mean", "Cord pH"),
        ("apgar5_mean", "Apgar5"),
        ("fhr_mean_mean", "FHR mean"),
        ("fhr_sd_mean", "FHR SD"),
        ("fhr_p05_mean", "FHR P05"),
        ("uc_mean_mean", "UC mean"),
        ("uc_p95_mean", "UC P95"),
        ("deceleration_proxy_count_mean", "Decel proxy"),
    ]
    heat = signal[[m[0] for m in metrics]].copy()
    heat_z = (heat - heat.mean(axis=0)) / heat.std(axis=0, ddof=0)
    heat_source = signal[["signal_phenotype"] + [m[0] for m in metrics]].copy()
    write_source_data("Figure_3_source_data.csv", [
        ("A_D", signal),
        ("E_H", proto[proto["phenotype_type"] == "signal_phenotype"]),
        ("C", heat_source),
    ])

    fig = plt.figure(figsize=(7.2, 8.8))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.82, 0.82, 1.45, 1.45])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    proto_axes = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]

    phenos = signal["signal_phenotype"].astype(str)
    ax_a.bar(phenos, signal["neonatal_risk_mean"], color="#c45a3d")
    ax_a.set_xlabel("Signal phenotype")
    ax_a.set_ylabel("Neonatal risk rate")
    ax_a.set_title("Risk gradient", loc="left", fontsize=8, fontweight="bold")
    ax_a.grid(axis="y", alpha=0.25)
    add_panel_label(ax_a, "A")

    ax_b.plot(phenos, signal["cord_ph_mean"], marker="o", color="#2f6f9f", label="Cord pH")
    ax_b.set_xlabel("Signal phenotype")
    ax_b.set_ylabel("Mean cord pH")
    ax_b2 = ax_b.twinx()
    ax_b2.plot(phenos, signal["apgar5_mean"], marker="s", color="#8a6bb8", label="Apgar5")
    ax_b2.set_ylabel("Mean Apgar5")
    ax_b.set_title("Clinical outcome profile", loc="left", fontsize=8, fontweight="bold")
    ax_b.grid(axis="y", alpha=0.25)
    ax_b.legend(loc="lower left", fontsize=6, frameon=False)
    ax_b2.legend(loc="lower right", fontsize=6, frameon=False)
    add_panel_label(ax_b, "B")

    im = ax_c.imshow(heat_z.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-1.8, vmax=1.8)
    ax_c.set_yticks(np.arange(len(signal)))
    ax_c.set_yticklabels(phenos)
    ax_c.set_xticks(np.arange(len(metrics)))
    ax_c.set_xticklabels([m[1] for m in metrics], rotation=45, ha="right", fontsize=6)
    ax_c.set_ylabel("Signal phenotype")
    ax_c.set_title("Standardized phenotype profile", loc="left", fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.02, label="z")
    add_panel_label(ax_c, "C")

    ax_d.plot(phenos, signal["fhr_sd_mean"], marker="o", color="#2f6f9f", label="FHR SD")
    ax_d.set_xlabel("Signal phenotype")
    ax_d.set_ylabel("FHR SD")
    ax_d2 = ax_d.twinx()
    ax_d2.plot(phenos, signal["deceleration_proxy_count_mean"], marker="s", color="#8a6bb8", label="Decel proxy")
    ax_d2.set_ylabel("Decel proxy")
    ax_d.set_title("Physiological signal burden", loc="left", fontsize=8, fontweight="bold")
    ax_d.grid(axis="y", alpha=0.25)
    ax_d.legend(loc="upper left", fontsize=6)
    ax_d2.legend(loc="upper right", fontsize=6)
    add_panel_label(ax_d, "D")

    proto_signal = proto[proto["phenotype_type"] == "signal_phenotype"].sort_values("phenotype")
    for ax, (_, row), label in zip(proto_axes, proto_signal.iterrows(), ["E", "F", "G", "H"]):
        image_path = Path(str(row["figure"]))
        if not image_path.exists():
            image_path = Path(str(row["figure"]).replace("\\", "/"))
        img = crop_white(plt.imread(image_path), pad=10)
        ax.imshow(img, aspect="auto")
        ax.axis("off")
        ax.set_title(f"Phenotype {row['phenotype']} prototype, record {row['record_id']}", fontsize=7)
        add_panel_label(ax, label)
    fig.tight_layout(h_pad=1.0, w_pad=1.3)
    save_pub(fig, "Figure_3_dynamic_phenotype_prototypes")
    plt.close(fig)


def figure_4():
    recal = read_csv("results/tables/recalibration_metrics_ci.csv")
    bins = read_csv("results/tables/recalibration_curve_bins.csv")
    dca = read_csv("results/tables/recalibrated_decision_curve_net_benefit.csv")
    dca_summary = read_csv("results/tables/recalibrated_decision_curve_summary.csv")
    recal_preds = read_csv("results/tables/recalibrated_test_predictions.csv")
    model = "signal_plus_ssl_plus_signal_phenotype"
    final_bins = bins[bins["model"] == model].copy()
    final_preds = recal_preds[recal_preds["model"] == model].copy()
    write_source_data("Figure_4_source_data.csv", [
        ("A", final_bins),
        ("B", recal[recal["model"] == model]),
        ("C", final_preds),
        ("D", final_bins),
        ("E", dca[dca["strategy"].isin([f"{model}_raw", f"{model}_platt", f"{model}_isotonic", "treat_all", "treat_none"])]),
        ("F", dca_summary[dca_summary["threshold_range"] == "0.10-0.30"]),
    ])

    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(3, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    for method, group in final_bins.groupby("calibration_method"):
        ax_a.plot(group["mean_predicted_risk"], group["observed_event_rate"], marker="o", label=method, color=PALETTE.get(method))
    ax_a.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax_a.set_xlabel("Mean predicted risk")
    ax_a.set_ylabel("Observed event rate")
    ax_a.set_title("Cross-fitted recalibration", loc="left", fontsize=8, fontweight="bold")
    ax_a.grid(alpha=0.25)
    ax_a.legend(frameon=False, fontsize=6)
    add_panel_label(ax_a, "A")

    methods = ["raw", "platt", "isotonic"]
    x = np.arange(len(methods))
    for offset, metric, color, label in [
        (-0.16, "brier", "#4c78a8", "Brier point"),
        (0.02, "expected_calibration_error", "#f58518", "ECE point"),
        (0.20, "expected_calibration_error", "#d95f02", "ECE bootstrap mean"),
    ]:
        rows = recal[
            (recal["model"] == model)
            & (recal["calibration_method"].isin(methods))
            & (recal["metric"] == metric)
        ].set_index("calibration_method").loc[methods].reset_index()
        if "bootstrap" in label:
            yvals = rows["bootstrap_mean"].astype(float)
            yerr = [
                yvals - rows["ci_2_5"].astype(float),
                rows["ci_97_5"].astype(float) - yvals,
            ]
            ax_b.errorbar(x + offset, yvals, yerr=yerr, fmt="o", color=color, capsize=2, label=label)
        else:
            ax_b.scatter(x + offset, rows["point"].astype(float), s=24, color=color, label=label, zorder=3)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(methods)
    ax_b.set_ylabel("Metric value")
    ax_b.set_title("Calibration metrics: point vs bootstrap", loc="left", fontsize=8, fontweight="bold")
    ax_b.legend(frameon=False, fontsize=5.5)
    ax_b.grid(axis="y", alpha=0.25)
    add_panel_label(ax_b, "B")

    box_data = []
    box_labels = []
    box_colors = []
    for method in ["raw", "platt"]:
        for y_val, label in [(0, "No event"), (1, "Risk event")]:
            vals = final_preds[(final_preds["calibration_method"] == method) & (final_preds["y_true"] == y_val)]["probability"].to_numpy()
            box_data.append(vals)
            box_labels.append(f"{method}\n{label}")
            box_colors.append(PALETTE[method] if y_val == 1 else "#b9c0c8")
    bp = ax_c.boxplot(box_data, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax_c.set_xticklabels(box_labels, fontsize=6)
    ax_c.set_ylabel("Predicted risk")
    ax_c.set_title("Risk score compression", loc="left", fontsize=8, fontweight="bold")
    ax_c.grid(axis="y", alpha=0.25)
    add_panel_label(ax_c, "C")

    width = 0.22
    for idx, method in enumerate(methods):
        sub = final_bins[final_bins["calibration_method"] == method].sort_values("bin")
        xpos = np.arange(len(sub)) + (idx - 1) * width
        ax_d.bar(xpos, sub["absolute_error"], width=width, color=PALETTE[method], label=method)
    ax_d.set_xticks(np.arange(5))
    ax_d.set_xticklabels([str(i) for i in range(1, 6)])
    ax_d.set_xlabel("Calibration bin")
    ax_d.set_ylabel("Absolute error")
    ax_d.set_title("Bin-level calibration error", loc="left", fontsize=8, fontweight="bold")
    ax_d.legend(fontsize=6, frameon=False)
    ax_d.grid(axis="y", alpha=0.25)
    add_panel_label(ax_d, "D")

    strategies = {
        f"{model}_raw": ("Raw", "#777777", "-"),
        f"{model}_platt": ("Platt", "#2f6f9f", "-"),
        f"{model}_isotonic": ("Isotonic", "#c45a3d", "-"),
        "treat_all": ("Treat all", "gray", "--"),
        "treat_none": ("Treat none", "black", ":"),
    }
    for strategy, (label, color, style) in strategies.items():
        group = dca[dca["strategy"] == strategy]
        ax_e.plot(group["threshold"], group["net_benefit"], label=label, color=color, linestyle=style)
    ax_e.axhline(0, color="black", linewidth=0.8)
    ax_e.set_xlabel("Risk threshold")
    ax_e.set_ylabel("Net benefit")
    ax_e.set_title("Decision curve", loc="left", fontsize=8, fontweight="bold")
    ax_e.legend(frameon=False, fontsize=5.8)
    ax_e.grid(alpha=0.25)
    add_panel_label(ax_e, "E")

    summary = dca_summary[
        (dca_summary["threshold_range"] == "0.10-0.30")
        & dca_summary["strategy"].isin([f"{model}_raw", f"{model}_platt", f"{model}_isotonic"])
    ].copy()
    summary["label"] = summary["strategy"].str.replace(f"{model}_", "", regex=False)
    ax_f.bar(summary["label"], summary["mean_net_benefit"], color=[PALETTE.get(x) for x in summary["label"]])
    ax_f.set_ylabel("Mean net benefit")
    ax_f.set_title("Threshold range 0.10-0.30", loc="left", fontsize=8, fontweight="bold")
    ax_f.grid(axis="y", alpha=0.25)
    add_panel_label(ax_f, "F")

    fig.tight_layout(h_pad=1.4, w_pad=1.4)
    save_pub(fig, "Figure_4_recalibration_and_clinical_utility")
    plt.close(fig)


def figure_s1():
    external = read_csv("results/tables/ctgdl_external_embedding_validation.csv")
    assigned = read_csv("results/tables/ctgdl_fhrma_assigned_ssl_phenotypes.csv")
    ssl_summary = read_csv("results/tables/ctu_uhb_ssl_phenotype_summary.csv")
    counts = assigned["assigned_ctu_ssl_phenotype"].value_counts().sort_index().reset_index()
    counts.columns = ["assigned_ctu_ssl_phenotype", "n_records"]
    counts["proportion"] = counts["n_records"] / counts["n_records"].sum()
    ext_map = dict(zip(external["metric"], external["value"]))
    write_source_data("Figure_S1_source_data.csv", [("A_B", external), ("C", counts), ("D", ssl_summary)])

    fig = plt.figure(figsize=(7.2, 6.45))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    img = plt.imread("results/figures/ctgdl_external_embedding_pca.png")
    ax_a.imshow(crop_white(img, pad=5), aspect="auto")
    ax_a.axis("off")
    ax_a.set_title("External representation space", fontsize=8, fontweight="bold")
    add_panel_label(ax_a, "A")

    domain_auroc = float(ext_map["domain_classifier_auroc_ctgdl_vs_ctu"])
    ax_b.hlines(0, 0.5, domain_auroc, color="#4c78a8", linewidth=2.0)
    ax_b.scatter([domain_auroc], [0], s=48, color="#4c78a8", edgecolor="white", linewidth=0.6, zorder=3)
    ax_b.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax_b.text(0.5, -0.20, "chance", ha="center", va="top", fontsize=6)
    ax_b.text(domain_auroc, 0.15, f"{domain_auroc:.3f}", ha="center", va="bottom", fontsize=7)
    ax_b.set_xlim(0.45, 1.0)
    ax_b.set_ylim(-0.32, 0.32)
    ax_b.set_yticks([])
    ax_b.set_xlabel("Domain-classifier AUROC")
    ax_b.set_title("Domain separability", loc="left", fontsize=8, fontweight="bold")
    ax_b.grid(axis="x", alpha=0.25)
    add_panel_label(ax_b, "B")

    ax_c.bar(counts["assigned_ctu_ssl_phenotype"].astype(str), counts["n_records"], color="#8a6bb8")
    ax_c.set_xlabel("Assigned CTU SSL phenotype")
    ax_c.set_ylabel("CTGDL-FHRMA records")
    ax_c.set_title("External phenotype assignment", loc="left", fontsize=8, fontweight="bold")
    ax_c.grid(axis="y", alpha=0.25)
    add_panel_label(ax_c, "C")

    ax_d.plot(ssl_summary["ssl_phenotype"].astype(str), ssl_summary["neonatal_risk_mean"], marker="o", color="#c45a3d")
    ax_d.set_title("CTU SSL risk gradient", loc="left", fontsize=8, fontweight="bold")
    ax_d.set_xlabel("SSL phenotype")
    ax_d.set_ylabel("Risk rate")
    ax_d.grid(axis="y", alpha=0.25)
    add_panel_label(ax_d, "D")

    fig.tight_layout(w_pad=1.4, h_pad=1.4)
    save_pub(fig, "Figure_S1_external_domain_shift")
    plt.close(fig)


def figure_s2():
    sweep = read_csv("results/tables/transformer_sweep_summary_ranked.csv")
    family = read_csv("results/tables/permutation_feature_family_importance.csv")
    losses = read_loss_curve("results/models/transformer_ssl_encoder_metrics.txt")
    write_source_data("Figure_S2_source_data.csv", [
        ("A", sweep),
        ("B", sweep),
        ("C", losses),
        ("D", family),
    ])

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    top = sweep.sort_values("auprc", ascending=False).head(8).copy()
    top["label"] = top["config"] + "\n" + top["model"].map(model_label).fillna(top["model"])
    ax_a.barh(np.arange(len(top)), top["auprc"], color=top["model"].map(PALETTE).fillna("#777777"))
    ax_a.set_yticks(np.arange(len(top)))
    ax_a.set_yticklabels(top["label"], fontsize=5.5)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("AUPRC")
    ax_a.set_title("Transformer sweep ranking", loc="left", fontsize=8, fontweight="bold")
    ax_a.grid(axis="x", alpha=0.25)
    add_panel_label(ax_a, "A")

    sweep_num = sweep.copy()
    for col in ["patch_len", "d_model", "layers", "contrastive_weight", "auprc"]:
        sweep_num[col] = pd.to_numeric(sweep_num[col], errors="coerce")
    summary_rows = []
    for variable, label in [
        ("mask_strategy", "Mask"),
        ("patch_len", "Patch length"),
        ("d_model", "d_model"),
        ("contrastive_weight", "Contrastive weight"),
    ]:
        temp = sweep_num.groupby(variable, dropna=False)["auprc"].max().reset_index()
        temp["variable"] = label
        temp["level"] = temp[variable].astype(str)
        summary_rows.append(temp[["variable", "level", "auprc"]])
    sweep_summary = pd.concat(summary_rows, ignore_index=True)
    y = np.arange(len(sweep_summary))
    colors = sweep_summary["variable"].map({
        "Mask": "#8a6bb8",
        "Patch length": "#4c78a8",
        "d_model": "#72b7b2",
        "Contrastive weight": "#f58518",
    }).fillna("#777777")
    ax_b.barh(y, sweep_summary["auprc"], color=colors)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(sweep_summary["variable"] + ": " + sweep_summary["level"], fontsize=5.6)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Best AUPRC within setting")
    ax_b.set_title("Mask/patch/model-size sensitivity", loc="left", fontsize=8, fontweight="bold")
    ax_b.grid(axis="x", alpha=0.25)
    add_panel_label(ax_b, "B")

    if len(losses):
        ax_c.plot(losses["epoch"], losses["recon"], marker="o", color="#4c78a8", label="Reconstruction")
        ax_c2 = ax_c.twinx()
        ax_c2.plot(losses["epoch"], losses["contrastive"], marker="s", color="#f58518", label="Contrastive")
        ax_c.set_ylabel("Masked reconstruction loss")
        ax_c2.set_ylabel("NT-Xent loss")
        ax_c.legend(frameon=False, fontsize=6, loc="upper left")
        ax_c2.legend(frameon=False, fontsize=6, loc="upper right")
    ax_c.set_xlabel("Epoch")
    ax_c.set_title("Reconstruction vs contrastive objective", loc="left", fontsize=8, fontweight="bold")
    ax_c.grid(alpha=0.25)
    add_panel_label(ax_c, "C")

    sub = family[family["model"].isin(["signal_plus_ssl", "signal_plus_ssl_plus_signal_phenotype"])].copy()
    sub["model_label"] = sub["model"].map(model_label)
    sub["family_label"] = sub["feature_family"].replace({
        "clinical_signal": "Signal",
        "ssl_embedding": "SSL",
        "phenotype": "Phenotype",
    })
    pivot = sub.pivot(index="family_label", columns="model_label", values="positive_importance_mean_auprc").fillna(0)
    x = np.arange(len(pivot.index))
    width = 0.34
    for idx, col in enumerate(pivot.columns):
        ax_d.bar(x + (idx - 0.5) * width, pivot[col], width=width, label=col)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(pivot.index)
    ax_d.set_ylabel("Mean positive AUPRC importance")
    ax_d.set_title("Feature-family contribution", loc="left", fontsize=8, fontweight="bold")
    ax_d.set_ylim(0, float(pivot.to_numpy().max()) * 1.55)
    ax_d.legend(fontsize=5.4, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=1)
    ax_d.grid(axis="y", alpha=0.25)
    add_panel_label(ax_d, "D")

    fig.tight_layout(h_pad=1.4, w_pad=1.6)
    save_pub(fig, "Figure_S2_model_selection_and_sensitivity")
    plt.close(fig)


def figure_manifest():
    rows = [
        {
            "figure": "Figure 1",
            "file_stem": "Figure_1_study_design_pipeline",
            "claim": "Open CTG data support a PatchTST-style SSL architecture linked to dynamic phenotyping and validation.",
            "source_data": "Figure_1_source_data.csv",
            "panels": "A-E",
        },
        {
            "figure": "Figure 2",
            "file_stem": "Figure_2_model_validation_and_explanation",
            "claim": "SSL representations improve risk enrichment and remain interpretable through feature and sensitivity analyses.",
            "source_data": "Figure_2_source_data.csv",
            "panels": "A-F",
        },
        {
            "figure": "Figure 3",
            "file_stem": "Figure_3_dynamic_phenotype_prototypes",
            "claim": "Signal phenotypes provide clinically interpretable dynamic CTG prototypes.",
            "source_data": "Figure_3_source_data.csv",
            "panels": "A-H",
        },
        {
            "figure": "Figure 4",
            "file_stem": "Figure_4_recalibration_and_clinical_utility",
            "claim": "Cross-fitted Platt recalibration improves absolute risk calibration and threshold behavior.",
            "source_data": "Figure_4_source_data.csv",
            "panels": "A-F",
        },
        {
            "figure": "Figure S1",
            "file_stem": "Figure_S1_external_domain_shift",
            "claim": "CTGDL-FHRMA supports external representation/domain-shift validation.",
            "source_data": "Figure_S1_source_data.csv",
            "panels": "A-D",
        },
        {
            "figure": "Figure S2",
            "file_stem": "Figure_S2_model_selection_and_sensitivity",
            "claim": "Transformer configuration, masking, objective loss, and feature-family sensitivity support the AI method layer.",
            "source_data": "Figure_S2_source_data.csv",
            "panels": "A-D",
        },
    ]
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SOURCE_DIR / "manuscript_figure_manifest.csv", index=False)


def main():
    setup_style()
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_s1()
    figure_s2()
    figure_manifest()
    print(f"Wrote manuscript figures to {FIG_DIR}")
    print(f"Wrote source-data tables to {SOURCE_DIR}")


if __name__ == "__main__":
    main()
