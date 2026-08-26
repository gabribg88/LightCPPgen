#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import sys
from pathlib import Path


EXPECTED_FASTA_COUNTS = {
    "data/MLCPP2_Training.fasta": 1146,
    "data/MLCPP2_Independent.fasta": 2341,
    "data/MLCPP2_Training_equal_length.fasta": 1146,
    "data/MLCPP2_Independent_equal_length.fasta": 2341,
    "data/MLCPP2_Test_optimized.fasta": 30,
    "data/MLCPP2_Test_optimized_equal_length.fasta": 30,
}

EXPECTED_LABEL_COUNTS = {
    "data/MLCPP2_TrainingCPPvalues.csv": 1146,
    "data/MLCPP2_IndependentCPPvalues.csv": 2341,
}

REQUIRED_FILES = [
    "data/Grantham.txt",
    "data/PAAC.txt",
    "data/Protein_parameters_setting.json",
    "data/PeptidesForShap.csv",
    "data/PeptidesForShap2.csv",
    "models/LightCPP_20.pickle",
    "notebooks/0_Feature_engineering.ipynb",
    "notebooks/1_Modeling.ipynb",
    "notebooks/2_Global_model_interpretation.ipynb",
    "notebooks/3_Optimization.ipynb",
    "notebooks/4_Optimization_results_analysis.ipynb",
    "notebooks/5_Local_analysis_optimized_sequences.ipynb",
    "notebooks/APPENDIX_Fasta_creation.ipynb",
    "notebooks/APPENDIX_HPO_Model_Selection.ipynb",
]

REQUIRED_DIRS = ["data", "features", "figures", "models", "notebooks", "results", "src"]

GENERATED_ARTIFACT_PATTERNS = [
    "features/comb_*.pickle",
    "results/shap_values.pickle",
    "results/optimization_results.pickle",
]

IMPORT_CHECKS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "shap": "shap",
    "xgboost": "xgboost",
    "Bio": "biopython",
    "rdkit": "rdkit",
    "peptides": "peptides",
    "iFeatureOmegaCLI": "iFeatureOmegaCLI",
}


def ok(message):
    print(f"[OK]   {message}")


def warn(message):
    print(f"[WARN] {message}")


def fail(message):
    print(f"[FAIL] {message}")


def count_fasta(path):
    count = 0
    lengths = []
    current = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    lengths.append(len("".join(current)))
                    current = []
                count += 1
            else:
                current.append(line)
        if current:
            lengths.append(len("".join(current)))
    return count, lengths


def count_csv_rows(path):
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    return max(len(rows) - 1, 0)


def check_required_dirs(root):
    errors = []
    for rel in REQUIRED_DIRS:
        path = root / rel
        if path.is_dir():
            ok(f"directory exists: {rel}")
        else:
            fail(f"missing directory: {rel}")
            errors.append(rel)
    return errors


def check_required_files(root):
    errors = []
    for rel in REQUIRED_FILES:
        path = root / rel
        if path.is_file():
            ok(f"file exists: {rel}")
        else:
            fail(f"missing file: {rel}")
            errors.append(rel)
    return errors


def check_fasta_counts(root):
    errors = []
    for rel, expected in EXPECTED_FASTA_COUNTS.items():
        path = root / rel
        if not path.is_file():
            continue
        count, lengths = count_fasta(path)
        if count == expected:
            if lengths:
                ok(f"{rel}: {count} sequences, length range {min(lengths)}-{max(lengths)}")
            else:
                ok(f"{rel}: {count} sequences")
        else:
            fail(f"{rel}: expected {expected} sequences, found {count}")
            errors.append(rel)
    return errors


def check_label_counts(root):
    errors = []
    for rel, expected in EXPECTED_LABEL_COUNTS.items():
        path = root / rel
        if not path.is_file():
            continue
        count = count_csv_rows(path)
        if count == expected:
            ok(f"{rel}: {count} label rows")
        else:
            fail(f"{rel}: expected {expected} label rows, found {count}")
            errors.append(rel)
    return errors


def check_generated_artifacts(root, strict):
    errors = []
    for pattern in GENERATED_ARTIFACT_PATTERNS:
        matches = sorted(root.glob(pattern))
        if matches:
            ok(f"generated artifact present: {pattern} ({len(matches)} match(es))")
        elif strict:
            fail(f"generated artifact missing: {pattern}")
            errors.append(pattern)
        else:
            warn(f"generated artifact not found yet: {pattern}")
    return errors


def check_imports():
    errors = []
    for module, package in IMPORT_CHECKS.items():
        if importlib.util.find_spec(module) is None:
            fail(f"package import failed or unavailable: {package} ({module})")
            errors.append(package)
        else:
            ok(f"package import available: {package}")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Check LightCPPgen paper-reproduction inputs.")
    parser.add_argument("--root", default=".", help="Repository root. Defaults to the current directory.")
    parser.add_argument("--strict-generated", action="store_true", help="Fail if generated feature/result artifacts are missing.")
    parser.add_argument("--check-imports", action="store_true", help="Also check whether major Python packages can be imported.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"Checking LightCPPgen repository at: {root}")

    if not (root / "README.md").is_file() or not (root / "src").is_dir():
        fail("the selected root does not look like the LightCPPgen repository root")
        return 1

    errors = []
    errors.extend(check_required_dirs(root))
    errors.extend(check_required_files(root))
    errors.extend(check_fasta_counts(root))
    errors.extend(check_label_counts(root))
    errors.extend(check_generated_artifacts(root, args.strict_generated))

    if args.check_imports:
        errors.extend(check_imports())

    if errors:
        print(f"\nPreflight completed with {len(errors)} error(s).")
        return 1

    print("\nPreflight completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
