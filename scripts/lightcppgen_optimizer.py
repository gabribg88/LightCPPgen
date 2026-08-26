#!/usr/bin/env python3
import argparse
import csv
import pickle
import random
import sys
from pathlib import Path

import numpy as np


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    records = []
    current_id = None
    current = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current)))
                current_id = line[1:].strip() or f"seq_{len(records) + 1}"
                current = []
            else:
                current.append(line)
        if current_id is not None:
            records.append((current_id, "".join(current)))
    return records


def read_csv_sequences(path, sequence_column, id_column):
    records = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header: {path}")
        if sequence_column not in reader.fieldnames:
            raise ValueError(f"CSV file must contain a '{sequence_column}' column: {path}")
        for idx, row in enumerate(reader, start=1):
            seq_id = row.get(id_column) if id_column and id_column in row else None
            records.append((seq_id or f"seq_{idx}", row[sequence_column]))
    return records


def read_text_sequences(path):
    records = []
    with path.open() as handle:
        for idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line:
                records.append((f"seq_{idx}", line))
    return records


def load_input_records(args):
    records = []
    if args.sequences:
        records.extend((f"seq_{idx}", seq) for idx, seq in enumerate(args.sequences, start=1))
    if args.input:
        path = Path(args.input)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        suffix = path.suffix.lower()
        if suffix in {".fasta", ".fa", ".faa"}:
            records.extend(read_fasta(path))
        elif suffix == ".csv":
            records.extend(read_csv_sequences(path, args.sequence_column, args.id_column))
        else:
            records.extend(read_text_sequences(path))
    if not records:
        raise ValueError("Provide sequences with --sequences or --input.")
    return [(seq_id, clean_sequence(seq_id, seq)) for seq_id, seq in records]


def clean_sequence(seq_id, sequence):
    seq = "".join(str(sequence).split()).upper()
    invalid = sorted(set(seq) - VALID_AA)
    if invalid:
        raise ValueError(f"{seq_id}: invalid amino-acid symbols: {''.join(invalid)}")
    if len(seq) < 6:
        raise ValueError(f"{seq_id}: sequence length must be at least 6 for the 20-feature featurizer")
    return seq


def write_rows(path, rows, fieldnames):
    if path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", newline="")
    else:
        handle = sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if path:
            handle.close()


def load_optimizer_module(repo_root):
    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))
    try:
        import optimization
    except ModuleNotFoundError as exc:
        package_hint = dependency_hint(exc.name)
        raise RuntimeError(
            f"Missing dependency while importing the optimizer: {exc.name}. "
            f"Install the optimizer environment with `pip install -r requirements-optimizer.txt` "
            f"or the full paper environment with `pip install -r requirements.txt`."
            f"{package_hint}"
        ) from exc
    return optimization


def dependency_hint(module_name):
    hints = {
        "lightgbm": " Required package: lightgbm.",
        "Bio": " Required package: biopython.",
        "peptides": " Required package: peptides.",
        "sklearn": " Required package: scikit-learn.",
    }
    return hints.get(module_name, "")


def load_model(model_path):
    with model_path.open("rb") as handle:
        model_obj = pickle.load(handle)
    if isinstance(model_obj, dict) and "refit_model" in model_obj:
        return model_obj["refit_model"], model_obj.get("best_iteration"), model_obj
    return model_obj, None, model_obj


def validate_model_features(model, featurizer):
    if hasattr(model, "feature_name"):
        model_features = model.feature_name()
        if model_features and list(model_features) != list(featurizer.feature_names):
            raise ValueError(
                "Model/featurizer feature mismatch. "
                f"Model has {len(model_features)} features; featurizer has {len(featurizer.feature_names)}."
            )


def make_context(args):
    repo_root = Path(args.root).resolve()
    optimization = load_optimizer_module(repo_root)
    model_path = Path(args.model) if args.model else repo_root / "models" / "LightCPP_20.pickle"
    data_folder = Path(args.data_folder) if args.data_folder else repo_root / "data"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_folder.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")
    model, best_iteration, model_obj = load_model(model_path)
    featurizer = optimization.Featurizer(data_folder=str(data_folder))
    validate_model_features(model, featurizer)
    return {
        "repo_root": repo_root,
        "optimization": optimization,
        "model": model,
        "model_obj": model_obj,
        "best_iteration": best_iteration,
        "featurizer": featurizer,
        "data_folder": data_folder,
    }


def compute_feature_matrix(featurizer, records):
    values = []
    for _, sequence in records:
        values.append(featurizer.compute_features(sequence).iloc[0].to_numpy(dtype=float))
    return np.asarray(values)


def fit_anomaly_detector(ctx, training_fasta):
    from sklearn.neighbors import LocalOutlierFactor

    path = Path(training_fasta) if training_fasta else ctx["data_folder"] / "MLCPP2_Training.fasta"
    if not path.is_file():
        raise FileNotFoundError(f"Training FASTA for anomaly detection not found: {path}")
    records = [(seq_id, clean_sequence(seq_id, seq)) for seq_id, seq in read_fasta(path)]
    x_train = compute_feature_matrix(ctx["featurizer"], records)
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
    clf.fit(x_train)
    return clf


def predict_sequence(ctx, sequence):
    optimization = ctx["optimization"]
    features = ctx["featurizer"].compute_features(sequence)
    score = optimization.pred_penetration(features, ctx["model"], ctx["best_iteration"], 1)
    return float(score), features


def score_command(args):
    ctx = make_context(args)
    records = load_input_records(args)
    clf_anomaly = None if args.no_anomaly else fit_anomaly_detector(ctx, args.training_fasta)
    rows = []
    for seq_id, sequence in records:
        penetration, features = predict_sequence(ctx, sequence)
        anomaly = ""
        if clf_anomaly is not None:
            anomaly = float(ctx["optimization"].anomaly_score(features, clf_anomaly)[0])
        rows.append({
            "id": seq_id,
            "sequence": sequence,
            "length": len(sequence),
            "penetration_score": penetration,
            "anomaly_score": anomaly,
        })
    write_rows(args.output, rows, ["id", "sequence", "length", "penetration_score", "anomaly_score"])


def setup_aligner():
    from Bio import Align
    from Bio.Align import substitution_matrices

    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.target_internal_open_gap_score = -20
    aligner.query_internal_open_gap_score = -20
    return aligner


def optimize_one(ctx, sequence, clf_anomaly, args):
    optimization = ctx["optimization"]
    random.seed(args.seed)
    np.random.seed(args.seed)

    aligner = setup_aligner()
    params_fit = {
        "obj": optimization.my_fitness,
        "target_ligand": sequence,
        "aligner": aligner,
        "model": ctx["model"],
        "clf_anomaly": clf_anomaly,
        "featurizer": ctx["featurizer"],
        "best_iteration": ctx["best_iteration"],
        "n_models": 1,
    }

    genes = "ARNDCEQGHILKMFPSTWYV"
    population = []
    for _ in range(args.population_size):
        gnome = optimization.Individual.create_gnome(len(sequence), genes, sequence, args.max_diff_pct)
        population.append(optimization.Individual(gnome, params_fit))

    initial_penetration, initial_features = predict_sequence(ctx, sequence)
    initial_anomaly = float(optimization.anomaly_score(initial_features, clf_anomaly)[0])
    best = None
    stop_reason = "max_iter"
    generation_reached = 0

    for generation in range(1, args.max_iter + 1):
        generation_reached = generation
        population = sorted(population, key=lambda x: x.fitness)
        best = population[0]
        best_sequence = "".join(best.chromosome)
        best_penetration, best_features = predict_sequence(ctx, best_sequence)
        best_anomaly = float(optimization.anomaly_score(best_features, clf_anomaly)[0])

        if best_penetration >= args.penetration_threshold and best_anomaly <= args.anomaly_threshold:
            stop_reason = "thresholds_met"
            break
        if best.num_diff_residues == 0:
            stop_reason = "returned_to_input"
            break

        elite_size = max(1, int(args.elite_fraction * args.population_size))
        parent_pool_size = max(2, min(args.parent_pool_size, len(population)))
        new_generation = population[:elite_size]
        while len(new_generation) < args.population_size:
            parent1 = random.choice(population[:parent_pool_size])
            parent2 = random.choice(population[:parent_pool_size])
            new_generation.append(parent1.mate(parent2, genes))
        population = new_generation

    best_sequence = "".join(best.chromosome)
    best_penetration, best_features = predict_sequence(ctx, best_sequence)
    best_anomaly = float(optimization.anomaly_score(best_features, clf_anomaly)[0])
    return {
        "input_sequence": sequence,
        "input_length": len(sequence),
        "initial_penetration_score": initial_penetration,
        "initial_anomaly_score": initial_anomaly,
        "optimized_sequence": best_sequence,
        "optimized_length": len(best_sequence),
        "optimized_penetration_score": best_penetration,
        "optimized_anomaly_score": best_anomaly,
        "best_fitness": float(best.fitness),
        "different_letters": int(best.num_diff_residues),
        "generations": generation_reached,
        "stop_reason": stop_reason,
    }


def optimize_command(args):
    ctx = make_context(args)
    records = load_input_records(args)
    clf_anomaly = fit_anomaly_detector(ctx, args.training_fasta)
    rows = []
    for seq_id, sequence in records:
        row = optimize_one(ctx, sequence, clf_anomaly, args)
        row = {"id": seq_id, **row}
        rows.append(row)
    fieldnames = [
        "id",
        "input_sequence",
        "input_length",
        "initial_penetration_score",
        "initial_anomaly_score",
        "optimized_sequence",
        "optimized_length",
        "optimized_penetration_score",
        "optimized_anomaly_score",
        "best_fitness",
        "different_letters",
        "generations",
        "stop_reason",
    ]
    write_rows(args.output, rows, fieldnames)


def add_common_args(parser):
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], help="Repository root.")
    parser.add_argument("--model", default=None, help="Path to LightCPPgen pickle model.")
    parser.add_argument("--data-folder", default=None, help="Folder containing PAAC.txt and Grantham.txt.")
    parser.add_argument("--input", default=None, help="Input FASTA, CSV, or text file.")
    parser.add_argument("--sequences", nargs="*", default=None, help="One or more peptide sequences.")
    parser.add_argument("--sequence-column", default="Sequence", help="CSV column containing peptide sequences.")
    parser.add_argument("--id-column", default="ID", help="Optional CSV column containing sequence IDs.")
    parser.add_argument("--output", default=None, help="Output CSV path. Defaults to stdout.")
    parser.add_argument("--training-fasta", default=None, help="Training FASTA used to fit the anomaly detector.")


def build_parser():
    parser = argparse.ArgumentParser(description="Use the pretrained 20-feature LightCPPgen model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Score input sequences with the pretrained model.")
    add_common_args(score_parser)
    score_parser.add_argument("--no-anomaly", action="store_true", help="Do not compute anomaly scores.")
    score_parser.set_defaults(func=score_command)

    optimize_parser = subparsers.add_parser("optimize", help="Optimize input sequences with the pretrained model.")
    add_common_args(optimize_parser)
    optimize_parser.add_argument("--max-iter", type=int, default=50)
    optimize_parser.add_argument("--population-size", type=int, default=500)
    optimize_parser.add_argument("--max-diff-pct", type=float, default=50.0)
    optimize_parser.add_argument("--penetration-threshold", type=float, default=0.95)
    optimize_parser.add_argument("--anomaly-threshold", type=float, default=1.3)
    optimize_parser.add_argument("--elite-fraction", type=float, default=0.10)
    optimize_parser.add_argument("--parent-pool-size", type=int, default=50)
    optimize_parser.add_argument("--seed", type=int, default=42)
    optimize_parser.set_defaults(func=optimize_command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
