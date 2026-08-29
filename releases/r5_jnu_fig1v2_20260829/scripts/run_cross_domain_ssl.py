#!/usr/bin/env python3
"""Run the frozen JNU-to-CTU representation-transfer analysis for Reviewer 5."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


SCRIPT_PATH = Path(__file__).resolve()
BATCH_ROOT = SCRIPT_PATH.parents[2]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
CODE_ROOT = PROJECT_ROOT / "CTG_Foundation_Trajectory"
sys.path.insert(0, str(CODE_ROOT / "scripts" / "04_models"))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "05_reports"))

from train_transformer_ssl_encoder import (  # noqa: E402
    PatchTransformerAutoencoder,
    make_mask,
    masked_mse,
    nt_xent_loss,
)
import run_r5_targeted_reanalysis as r5  # noqa: E402


CTU_WINDOWS = PROJECT_ROOT / "R1_QA_20260706" / "window_rerun_20260706" / "ctu_uhb_windows.npz"
CTU_INDEX = PROJECT_ROOT / "R1_QA_20260706" / "window_rerun_20260706" / "ctu_uhb_window_index.csv"
CTU_FEATURES = PROJECT_ROOT / "R1_QA_20260706" / "real_raw_rerun_20260706" / "ctu_uhb_record_features.csv"
CTU_SPLIT = PROJECT_ROOT / "R1_QA_20260706" / "real_raw_rerun_20260706" / "ctu_uhb_record_split.csv"
CTU_MORPHOLOGY = PROJECT_ROOT / "R1_QA_20260706" / "ctu_chb_annotation_record_event_summary_20260706.csv"
JNU_WINDOWS = BATCH_ROOT / "04_data" / "jnu_processed" / "jnu_ctg_windows_f16.npy"
JNU_INDEX = BATCH_ROOT / "04_data" / "jnu_processed" / "jnu_window_index.csv"

SEEDS = [20260521, 20260522, 20260523, 20260524, 20260525]
SIGNAL_FEATURES = list(r5.SIGNAL_FEATURES)
MORPHOLOGY_TASKS = [
    "any_tachycardia",
    "any_acceleration",
    "any_deceleration",
    "any_deceleration_early",
    "any_deceleration_late",
    "any_deceleration_variable",
    "any_deceleration_prolonged",
]
INPUT_VARIANTS = ["full", "fhr_only", "uc_only", "one_hz"]


@dataclass(frozen=True)
class FrozenConfig:
    channels: int = 2
    length: int = 2400
    patch_len: int = 80
    d_model: int = 48
    nhead: int = 4
    layers: int = 2
    dropout: float = 0.1
    projection_dim: int = 64
    mask_strategy: str = "mixed"
    mask_fraction: float = 0.25
    reconstruction_weight: float = 1.0
    contrastive_weight: float = 0.1
    temperature: float = 0.2
    jitter_std: float = 0.02
    epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001


CONFIG = FrozenConfig()
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WindowDataset(Dataset):
    def __init__(self, array: np.ndarray, indices: np.ndarray | None = None):
        self.array = array
        self.indices = np.arange(len(array), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> torch.Tensor:
        values = np.asarray(self.array[self.indices[item]], dtype=np.float32)
        return torch.from_numpy(values.copy())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def model_from_config(seed: int) -> PatchTransformerAutoencoder:
    seed_everything(seed)
    return PatchTransformerAutoencoder(
        channels=CONFIG.channels,
        length=CONFIG.length,
        patch_len=CONFIG.patch_len,
        d_model=CONFIG.d_model,
        nhead=CONFIG.nhead,
        layers=CONFIG.layers,
        dropout=CONFIG.dropout,
        projection_dim=CONFIG.projection_dim,
    )


def asymmetric_fhr_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(
        batch_size,
        CONFIG.channels,
        CONFIG.length // CONFIG.patch_len,
        dtype=torch.bool,
        device=device,
    )
    n_patches = mask.shape[-1]
    random_mask = torch.rand(batch_size, n_patches, device=device) < (CONFIG.mask_fraction * 0.5)
    mask[:, 0, :] = random_mask
    span = max(1, int(round(n_patches * CONFIG.mask_fraction * 0.75)))
    for row in range(batch_size):
        start = int(torch.randint(0, max(1, n_patches - span + 1), (1,), device=device).item())
        mask[row, 0, start : start + span] = True
    empty = ~mask.reshape(batch_size, -1).any(dim=1)
    if empty.any():
        mask[empty, 0, 0] = True
    return mask.reshape(batch_size, -1)


def training_mask(batch_size: int, strategy: str, device: torch.device) -> torch.Tensor:
    if strategy == "asymmetric_fhr":
        return asymmetric_fhr_mask(batch_size, device)
    return make_mask(
        batch_size,
        CONFIG.channels,
        CONFIG.length // CONFIG.patch_len,
        strategy,
        CONFIG.mask_fraction,
        device,
    )


def train_encoder(
    array: np.ndarray,
    indices: np.ndarray | None,
    run_dir: Path,
    seed: int,
    init_weights: Path | None = None,
    mask_strategy: str = "mixed",
    epochs: int | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / "encoder_final.pt"
    state_path = run_dir / "training_state.json"
    if final_path.exists() and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "COMPLETE":
            print(f"[resume] {final_path}", flush=True)
            return final_path

    epochs = CONFIG.epochs if epochs is None else epochs
    device = DEVICE
    model = model_from_config(seed)
    if init_weights is not None:
        model.load_state_dict(torch.load(init_weights, map_location="cpu", weights_only=True))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )
    start_epoch = 0
    history: list[dict[str, float]] = []
    epoch_checkpoint = run_dir / "epoch_checkpoint.pt"
    if epoch_checkpoint.exists():
        checkpoint = torch.load(epoch_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))

    dataset = WindowDataset(array, indices)
    started = time.time()
    for epoch in range(start_epoch, epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=CONFIG.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=False,
        )
        model.train()
        total_values: list[float] = []
        recon_values: list[float] = []
        contrast_values: list[float] = []
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            mask1 = training_mask(len(batch), mask_strategy, device)
            mask2 = training_mask(len(batch), mask_strategy, device)
            view1 = batch + torch.randn_like(batch) * CONFIG.jitter_std
            view2 = batch + torch.randn_like(batch) * CONFIG.jitter_std
            pred1, target1, encoded1 = model(view1, mask1)
            pred2, target2, encoded2 = model(view2, mask2)
            reconstruction = 0.5 * (
                masked_mse(pred1, target1, mask1) + masked_mse(pred2, target2, mask2)
            )
            contrastive = nt_xent_loss(model.project(encoded1), model.project(encoded2), CONFIG.temperature)
            total = CONFIG.reconstruction_weight * reconstruction + CONFIG.contrastive_weight * contrastive
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_values.append(float(total.detach()))
            recon_values.append(float(reconstruction.detach()))
            contrast_values.append(float(contrastive.detach()))

        row = {
            "epoch": epoch + 1,
            "total_loss": float(np.mean(total_values)),
            "reconstruction_loss": float(np.mean(recon_values)),
            "contrastive_loss": float(np.mean(contrast_values)),
        }
        history.append(row)
        torch.save(
            {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history},
            epoch_checkpoint,
        )
        state_path.write_text(json.dumps({
            "status": "RUNNING",
            "seed": seed,
            "mask_strategy": mask_strategy,
            "epochs_complete": epoch + 1,
            "epochs_total": epochs,
            "n_windows": len(dataset),
            "history": history,
        }, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{run_dir.name}] seed={seed} epoch={epoch + 1}/{epochs} "
            f"loss={row['total_loss']:.5f} elapsed_min={(time.time() - started) / 60:.1f}",
            flush=True,
        )

    torch.save(model.cpu().state_dict(), final_path)
    state_path.write_text(json.dumps({
        "status": "COMPLETE",
        "seed": seed,
        "mask_strategy": mask_strategy,
        "epochs_complete": epochs,
        "epochs_total": epochs,
        "n_windows": len(dataset),
        "initial_weights": str(init_weights) if init_weights else None,
        "config": asdict(CONFIG),
        "history": history,
    }, indent=2) + "\n", encoding="utf-8")
    return final_path


def load_model(weights: Path, seed: int) -> PatchTransformerAutoencoder:
    model = model_from_config(seed)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.to(DEVICE).eval()
    return model


def apply_input_variant(batch: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "full":
        return batch
    output = batch.clone()
    if variant == "fhr_only":
        output[:, 1, :] = 0.0
    elif variant == "uc_only":
        output[:, 0, :] = 0.0
    elif variant == "one_hz":
        output = output[:, :, ::4].repeat_interleave(4, dim=-1)
    else:
        raise ValueError(f"unknown input variant {variant}")
    return output


def extract_record_embeddings(
    model: PatchTransformerAutoencoder,
    array: np.ndarray,
    index: pd.DataFrame,
    record_ids: set[str],
    variant: str = "full",
) -> pd.DataFrame:
    keep = index["record_id"].astype(str).isin(record_ids).to_numpy()
    positions = np.flatnonzero(keep)
    records = index.loc[keep, "record_id"].astype(str).to_numpy()
    embeddings: list[np.ndarray] = []
    model.to(DEVICE).eval()
    with torch.no_grad():
        for start in range(0, len(positions), 128):
            values = np.asarray(array[positions[start : start + 128]], dtype=np.float32)
            batch = apply_input_variant(torch.from_numpy(values.copy()), variant).to(DEVICE)
            embeddings.append(model.encode_clean(batch).cpu().numpy())
    matrix = np.concatenate(embeddings, axis=0)
    frame = pd.DataFrame(matrix, columns=[f"emb_{column:02d}" for column in range(matrix.shape[1])])
    frame.insert(0, "record_id", records)
    return frame.groupby("record_id", sort=True, as_index=False).mean(numeric_only=True)


def save_embeddings(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def merge_embeddings(base: pd.DataFrame, embeddings: pd.DataFrame) -> pd.DataFrame:
    return base.merge(embeddings, on="record_id", how="inner", validate="one_to_one")


def prediction_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    metrics = r5.score_predictions(y, probability)
    metrics["ece"] = r5.expected_calibration_error(y, probability)
    intercept, slope = r5.calibration_intercept_slope(y, probability)
    metrics["calibration_intercept"] = intercept
    metrics["calibration_slope"] = slope
    return metrics


def evaluate_risk_fold(
    base: pd.DataFrame,
    embeddings: pd.DataFrame,
    train_ids: set[str],
    validation_ids: set[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    merged = merge_embeddings(base, embeddings)
    train = merged[merged["record_id"].isin(train_ids)].copy()
    validation = merged[merged["record_id"].isin(validation_ids)].copy()
    candidate = r5.RiskModel(pca_dim=5).fit(train, train["neonatal_risk"])
    baseline = r5.RiskModel(pca_dim=0).fit(train, train["neonatal_risk"])
    candidate_probability = candidate.predict_proba(validation)
    baseline_probability = baseline.predict_proba(validation)
    metrics = prediction_metrics(validation["neonatal_risk"].to_numpy(int), candidate_probability)
    metrics.update({
        "n_train": len(train),
        "n_validation": len(validation),
        "events_train": int(train["neonatal_risk"].sum()),
        "events_validation": int(validation["neonatal_risk"].sum()),
        "n_predictors": candidate.n_predictors_,
        "events_per_predictor": float(train["neonatal_risk"].sum() / candidate.n_predictors_),
    })
    probabilities = validation[["record_id", "neonatal_risk"]].copy()
    probabilities["probability"] = candidate_probability
    probabilities["signal_probability"] = baseline_probability
    return metrics, probabilities


def binary_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int).to_numpy()
    return series.astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).to_numpy()


def linear_probe(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    outcome: str,
    seed: int,
) -> dict[str, float] | None:
    y_train = binary_values(train[outcome])
    y_validation = binary_values(validation[outcome])
    if len(np.unique(y_train)) < 2 or len(np.unique(y_validation)) < 2:
        return None
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[columns]))
    x_validation = scaler.transform(imputer.transform(validation[columns]))
    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
    ).fit(x_train, y_train)
    probability = model.predict_proba(x_validation)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "n_train": len(train),
        "n_validation": len(validation),
        "positive_train": int(y_train.sum()),
        "positive_validation": int(y_validation.sum()),
        "auroc": float(roc_auc_score(y_validation, probability)),
        "average_precision": float(average_precision_score(y_validation, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_validation, prediction)),
        "f1": float(f1_score(y_validation, prediction, zero_division=0)),
    }


def evaluate_morphology_fold(
    base: pd.DataFrame,
    morphology: pd.DataFrame,
    embeddings: pd.DataFrame,
    train_ids: set[str],
    validation_ids: set[str],
    seed: int,
    route: str,
    fold: int,
) -> list[dict[str, object]]:
    merged = merge_embeddings(base, embeddings).merge(
        morphology[["record_id", *MORPHOLOGY_TASKS]],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )
    train = merged[merged["record_id"].isin(train_ids)].copy()
    validation = merged[merged["record_id"].isin(validation_ids)].copy()
    embedding_columns = [column for column in merged.columns if column.startswith("emb_")]
    rows: list[dict[str, object]] = []
    for task in MORPHOLOGY_TASKS:
        for feature_set, columns in [("encoder_embedding", embedding_columns), ("classical_signal", SIGNAL_FEATURES)]:
            result = linear_probe(train, validation, columns, task, seed)
            if result is None:
                continue
            rows.append({
                "seed": seed,
                "fold": fold,
                "encoder_route": route,
                "feature_set": feature_set,
                "task": task,
                **result,
            })
    return rows


def make_folds(base: pd.DataFrame) -> list[tuple[set[str], set[str]]]:
    development = base[base["split"] == "train"].reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=20260521)
    folds: list[tuple[set[str], set[str]]] = []
    routing_rows: list[dict[str, object]] = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(development, development["neonatal_risk"]), start=1
    ):
        train_ids = set(development.iloc[train_index]["record_id"])
        validation_ids = set(development.iloc[validation_index]["record_id"])
        if train_ids & validation_ids:
            raise RuntimeError("development fold leakage detected")
        folds.append((train_ids, validation_ids))
        for record_id in train_ids:
            routing_rows.append({"record_id": record_id, "fold": fold, "role": "development_train"})
        for record_id in validation_ids:
            routing_rows.append({"record_id": record_id, "fold": fold, "role": "development_validation"})
    path = BATCH_ROOT / "09_reproducibility" / "ctu_development_folds.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(routing_rows).sort_values(["fold", "role", "record_id"]).to_csv(path, index=False)
    return folds


def ctu_window_positions(index: pd.DataFrame, record_ids: set[str]) -> np.ndarray:
    return np.flatnonzero(index["record_id"].astype(str).isin(record_ids).to_numpy())


def append_risk_result(
    metrics_rows: list[dict[str, object]],
    probability_rows: list[pd.DataFrame],
    base: pd.DataFrame,
    embeddings: pd.DataFrame,
    train_ids: set[str],
    validation_ids: set[str],
    seed: int,
    fold: int,
    route: str,
    variant: str,
    evaluation_set: str,
) -> None:
    metrics, probabilities = evaluate_risk_fold(base, embeddings, train_ids, validation_ids)
    metrics_rows.append({
        "seed": seed,
        "fold": fold,
        "encoder_route": route,
        "input_variant": variant,
        "evaluation_set": evaluation_set,
        **metrics,
    })
    probabilities.insert(0, "evaluation_set", evaluation_set)
    probabilities.insert(0, "input_variant", variant)
    probabilities.insert(0, "encoder_route", route)
    probabilities.insert(0, "fold", fold)
    probabilities.insert(0, "seed", seed)
    probability_rows.append(probabilities)


def write_outputs(
    stem: str,
    metrics_rows: list[dict[str, object]],
    probability_rows: list[pd.DataFrame],
    morphology_rows: list[dict[str, object]],
) -> None:
    table_dir = BATCH_ROOT / "04_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(table_dir / f"{stem}_risk_metrics.csv", index=False)
    pd.concat(probability_rows, ignore_index=True).to_csv(table_dir / f"{stem}_risk_probabilities.csv", index=False)
    if morphology_rows:
        pd.DataFrame(morphology_rows).to_csv(table_dir / f"{stem}_morphology_probe.csv", index=False)


def run_development(force: bool = False) -> None:
    del force  # outputs are resumed by checkpoint presence; no destructive overwrite route
    ctu_array = np.load(CTU_WINDOWS)["x"].astype(np.float32)
    jnu_array = np.load(JNU_WINDOWS, mmap_mode="r")
    ctu_index = pd.read_csv(CTU_INDEX, dtype={"record_id": str})
    jnu_index = pd.read_csv(JNU_INDEX, dtype={"record_id": str, "patient_id": str})
    if jnu_index["patient_id"].nunique() != 12_606 or len(jnu_index) != 62_307:
        raise RuntimeError("JNU grouped index contract failed")
    base = r5.load_base_data()
    morphology = pd.read_csv(CTU_MORPHOLOGY, dtype={"record_id": str})
    folds = make_folds(base)
    development_ids = set(base.loc[base["split"] == "train", "record_id"])
    fixed_test_ids = set(base.loc[base["split"] == "test", "record_id"])
    if development_ids & fixed_test_ids:
        raise RuntimeError("fixed split leakage detected")

    metrics_rows: list[dict[str, object]] = []
    probability_rows: list[pd.DataFrame] = []
    morphology_rows: list[dict[str, object]] = []
    model_root = BATCH_ROOT / "03_analysis" / "models"
    embedding_root = BATCH_ROOT / "03_analysis" / "embeddings"

    for seed in SEEDS:
        print(f"=== DEVELOPMENT SEED {seed} ===", flush=True)
        seed_root = model_root / f"seed_{seed}"
        random_model = model_from_config(seed)
        random_embeddings = extract_record_embeddings(
            random_model, ctu_array, ctu_index, development_ids, "full"
        )
        save_embeddings(random_embeddings, embedding_root / f"seed_{seed}" / "random_development.csv")

        jnu_weights = train_encoder(jnu_array, None, seed_root / "jnu_mixed", seed, mask_strategy="mixed")
        jnu_model = load_model(jnu_weights, seed)
        jnu_embeddings = extract_record_embeddings(jnu_model, ctu_array, ctu_index, development_ids, "full")
        save_embeddings(jnu_embeddings, embedding_root / f"seed_{seed}" / "jnu_only_development.csv")

        asymmetric_weights = train_encoder(
            jnu_array,
            None,
            seed_root / "jnu_asymmetric_fhr",
            seed,
            mask_strategy="asymmetric_fhr",
        )
        asymmetric_model = load_model(asymmetric_weights, seed)
        asymmetric_embeddings = extract_record_embeddings(
            asymmetric_model, ctu_array, ctu_index, development_ids, "full"
        )
        save_embeddings(
            asymmetric_embeddings,
            embedding_root / f"seed_{seed}" / "jnu_asymmetric_development.csv",
        )

        shared_routes = {
            "random": random_embeddings,
            "jnu_only": jnu_embeddings,
            "jnu_asymmetric": asymmetric_embeddings,
        }
        for fold, (train_ids, validation_ids) in enumerate(folds, start=1):
            fold_root = seed_root / f"fold_{fold}"
            train_positions = ctu_window_positions(ctu_index, train_ids)
            ctu_weights = train_encoder(
                ctu_array,
                train_positions,
                fold_root / "ctu_only",
                seed,
                mask_strategy="mixed",
            )
            sequential_weights = train_encoder(
                ctu_array,
                train_positions,
                fold_root / "jnu_to_ctu",
                seed,
                init_weights=jnu_weights,
                mask_strategy="mixed",
            )
            ctu_model = load_model(ctu_weights, seed)
            sequential_model = load_model(sequential_weights, seed)
            fold_models = {"ctu_only": ctu_model, "jnu_to_ctu": sequential_model}
            fold_embeddings: dict[str, pd.DataFrame] = dict(shared_routes)
            for route, model in fold_models.items():
                embeddings = extract_record_embeddings(model, ctu_array, ctu_index, development_ids, "full")
                fold_embeddings[route] = embeddings
                save_embeddings(
                    embeddings,
                    embedding_root / f"seed_{seed}" / f"fold_{fold}_{route}_development.csv",
                )

            for route, embeddings in fold_embeddings.items():
                append_risk_result(
                    metrics_rows,
                    probability_rows,
                    base,
                    embeddings,
                    train_ids,
                    validation_ids,
                    seed,
                    fold,
                    route,
                    "full",
                    "development_oof",
                )
                morphology_rows.extend(evaluate_morphology_fold(
                    base,
                    morphology,
                    embeddings,
                    train_ids,
                    validation_ids,
                    seed,
                    route,
                    fold,
                ))

            for route, model in fold_models.items():
                for variant in INPUT_VARIANTS[1:]:
                    embeddings = extract_record_embeddings(
                        model, ctu_array, ctu_index, development_ids, variant
                    )
                    append_risk_result(
                        metrics_rows,
                        probability_rows,
                        base,
                        embeddings,
                        train_ids,
                        validation_ids,
                        seed,
                        fold,
                        route,
                        variant,
                        "development_sensitivity",
                    )

        write_outputs("development", metrics_rows, probability_rows, morphology_rows)

    flag = BATCH_ROOT / "03_analysis" / "DEVELOPMENT_COMPLETE.flag"
    flag.write_text(
        "PASS\nFixed CTU test records were not used for model selection or development analysis.\n",
        encoding="utf-8",
    )


def run_fixed_test() -> None:
    flag = BATCH_ROOT / "03_analysis" / "DEVELOPMENT_COMPLETE.flag"
    if not flag.exists() or not flag.read_text(encoding="utf-8").startswith("PASS"):
        raise SystemExit("fixed-test phase is blocked until development phase has completed")

    ctu_array = np.load(CTU_WINDOWS)["x"].astype(np.float32)
    jnu_array = np.load(JNU_WINDOWS, mmap_mode="r")
    ctu_index = pd.read_csv(CTU_INDEX, dtype={"record_id": str})
    base = r5.load_base_data()
    development_ids = set(base.loc[base["split"] == "train", "record_id"])
    fixed_test_ids = set(base.loc[base["split"] == "test", "record_id"])
    all_ids = development_ids | fixed_test_ids
    development_positions = ctu_window_positions(ctu_index, development_ids)
    metrics_rows: list[dict[str, object]] = []
    probability_rows: list[pd.DataFrame] = []
    model_root = BATCH_ROOT / "03_analysis" / "models"
    embedding_root = BATCH_ROOT / "03_analysis" / "embeddings"

    for seed in SEEDS:
        print(f"=== FIXED TEST SEED {seed} ===", flush=True)
        seed_root = model_root / f"seed_{seed}"
        jnu_weights = seed_root / "jnu_mixed" / "encoder_final.pt"
        if not jnu_weights.exists():
            raise RuntimeError(f"missing development-stage JNU model: {jnu_weights}")
        random_model = model_from_config(seed)
        jnu_model = load_model(jnu_weights, seed)
        ctu_weights = train_encoder(
            ctu_array,
            development_positions,
            seed_root / "all_development" / "ctu_only",
            seed,
            mask_strategy="mixed",
        )
        sequential_weights = train_encoder(
            ctu_array,
            development_positions,
            seed_root / "all_development" / "jnu_to_ctu",
            seed,
            init_weights=jnu_weights,
            mask_strategy="mixed",
        )
        routes = {
            "random": random_model,
            "jnu_only": jnu_model,
            "ctu_only": load_model(ctu_weights, seed),
            "jnu_to_ctu": load_model(sequential_weights, seed),
        }
        for route, model in routes.items():
            embeddings = extract_record_embeddings(model, ctu_array, ctu_index, all_ids, "full")
            save_embeddings(
                embeddings,
                embedding_root / f"seed_{seed}" / f"{route}_all_records.csv",
            )
            append_risk_result(
                metrics_rows,
                probability_rows,
                base,
                embeddings,
                development_ids,
                fixed_test_ids,
                seed,
                0,
                route,
                "full",
                "fixed_test_exploratory",
            )
            if route in {"ctu_only", "jnu_to_ctu"}:
                for variant in INPUT_VARIANTS[1:]:
                    variant_embeddings = extract_record_embeddings(
                        model, ctu_array, ctu_index, all_ids, variant
                    )
                    append_risk_result(
                        metrics_rows,
                        probability_rows,
                        base,
                        variant_embeddings,
                        development_ids,
                        fixed_test_ids,
                        seed,
                        0,
                        route,
                        variant,
                        "fixed_test_sensitivity_exploratory",
                    )

        write_outputs("fixed_test", metrics_rows, probability_rows, [])

    (BATCH_ROOT / "03_analysis" / "FIXED_TEST_COMPLETE.flag").write_text(
        "PASS\nExploratory only; historical test exposure may induce optimism.\n",
        encoding="utf-8",
    )


def validate_inputs() -> None:
    required = [
        CTU_WINDOWS,
        CTU_INDEX,
        CTU_FEATURES,
        CTU_SPLIT,
        CTU_MORPHOLOGY,
        JNU_WINDOWS,
        JNU_INDEX,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing frozen input(s): " + "; ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "fixed-test", "all"], default="development")
    parser.add_argument("--threads", type=int, default=min(20, os.cpu_count() or 1))
    args = parser.parse_args()
    validate_inputs()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(min(4, args.threads))
    except RuntimeError:
        pass
    (BATCH_ROOT / "10_logs").mkdir(parents=True, exist_ok=True)
    (BATCH_ROOT / "09_reproducibility").mkdir(parents=True, exist_ok=True)
    (BATCH_ROOT / "09_reproducibility" / "FROZEN_MODEL_CONFIG.json").write_text(
        json.dumps({
            **asdict(CONFIG),
            "runtime_device": str(DEVICE),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }, indent=2) + "\n", encoding="utf-8"
    )
    if args.phase in {"development", "all"}:
        run_development()
    if args.phase in {"fixed-test", "all"}:
        run_fixed_test()


if __name__ == "__main__":
    main()
