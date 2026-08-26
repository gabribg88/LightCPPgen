# Pretrained Optimizer Mode

This mode uses the published 20-feature LightCPPgen model in `models/LightCPP_20.pickle` to score or optimize peptide sequences without recomputing the full descriptor set and without retraining the model.

Use this mode when you already have one or more peptide sequences and want to estimate their CPP penetration score or generate optimized variants.

## What This Mode Uses

```text
models/LightCPP_20.pickle
src/optimization.py
scripts/lightcppgen_optimizer.py
data/PAAC.txt
data/Grantham.txt
data/MLCPP2_Training.fasta
```

The script computes the same 20 features used by the published pretrained model through the fast `Featurizer` class in `src/optimization.py`.

The full feature-engineering pipeline is not required.

## Installation

Create a lightweight environment:

```bash
conda create -n lightcppgen-optimizer python=3.9
conda activate lightcppgen-optimizer
pip install -r requirements-optimizer.txt
```

The default optimizer requirements are intentionally flexible. If you need the exact tested versions, use:

```bash
pip install -r requirements-optimizer-pinned.txt
```

## Quick Test

From the repository root:

```bash
python scripts/lightcppgen_optimizer.py score \
  --sequences GEPWKVCVN \
  --output results/scored_sequences.csv
```

The output file should contain:

```text
id,sequence,length,penetration_score,anomaly_score
```

## Input Formats

### Command-Line Sequences

```bash
python scripts/lightcppgen_optimizer.py score \
  --sequences GEPWKVCVN LDPIVAKRVRHILTENARTVEA \
  --output results/scored_sequences.csv
```

### FASTA

```bash
python scripts/lightcppgen_optimizer.py score \
  --input peptides.fasta \
  --output results/scored_sequences.csv
```

Example FASTA:

```text
>peptide_1
GEPWKVCVN
>peptide_2
LDPIVAKRVRHILTENARTVEA
```

### CSV

```bash
python scripts/lightcppgen_optimizer.py score \
  --input peptides.csv \
  --sequence-column Sequence \
  --id-column ID \
  --output results/scored_sequences.csv
```

Example CSV:

```text
ID,Sequence
peptide_1,GEPWKVCVN
peptide_2,LDPIVAKRVRHILTENARTVEA
```

### Plain Text

Plain text input is also accepted, with one sequence per line:

```bash
python scripts/lightcppgen_optimizer.py score \
  --input peptides.txt \
  --output results/scored_sequences.csv
```

## Scoring Sequences

The `score` command computes the CPP penetration score for each input sequence.

```bash
python scripts/lightcppgen_optimizer.py score \
  --input peptides.fasta \
  --output results/scored_sequences.csv
```

Output columns:

```text
id
sequence
length
penetration_score
anomaly_score
```

The `penetration_score` is the pretrained LightGBM model output. Higher values indicate stronger predicted CPP-like behavior.

The `anomaly_score` is computed with a `LocalOutlierFactor` model fitted on the bundled MLCPP2.0 training sequences after applying the same 20-feature featurizer. Lower values indicate that the sequence is closer to the training-domain feature distribution. This score is mainly useful as a conservative warning signal, not as a biological property.

To skip anomaly scoring:

```bash
python scripts/lightcppgen_optimizer.py score \
  --input peptides.fasta \
  --no-anomaly \
  --output results/scored_sequences.csv
```

## Optimizing Sequences

The `optimize` command runs the genetic algorithm used by LightCPPgen around the pretrained model.

```bash
python scripts/lightcppgen_optimizer.py optimize \
  --sequences GEPWKVCVN \
  --population-size 500 \
  --max-iter 50 \
  --output results/optimized_sequences.csv
```

For a quick functional test:

```bash
python scripts/lightcppgen_optimizer.py optimize \
  --sequences GEPWKVCVN \
  --population-size 50 \
  --max-iter 5 \
  --output results/optimized_sequences_quick_test.csv
```

Output columns:

```text
id
input_sequence
input_length
initial_penetration_score
initial_anomaly_score
optimized_sequence
optimized_length
optimized_penetration_score
optimized_anomaly_score
best_fitness
different_letters
generations
stop_reason
```

## Main Optimization Parameters

```text
--population-size
```

Number of candidate sequences in each generation. Larger values explore more variants but increase runtime.

```text
--max-iter
```

Maximum number of generations.

```text
--max-diff-pct
```

Maximum percentage of residues initially mutated relative to the input sequence.

```text
--penetration-threshold
```

Target CPP penetration score used as a stopping criterion. Default: `0.95`.

```text
--anomaly-threshold
```

Maximum accepted anomaly score used as a stopping criterion. Default: `1.3`.

```text
--seed
```

Random seed for the genetic algorithm. Default: `42`.

## Reproducibility

The scoring mode is deterministic for a fixed model and package behavior.

The optimization mode is stochastic. Use `--seed` to make repeated runs comparable:

```bash
python scripts/lightcppgen_optimizer.py optimize \
  --sequences GEPWKVCVN \
  --seed 42 \
  --output results/optimized_seed42.csv
```

Small numerical differences may still occur across LightGBM, NumPy, scikit-learn or operating-system versions.

## Sequence Validation

Input sequences are converted to uppercase and whitespace is removed.

Accepted amino-acid symbols:

```text
ACDEFGHIKLMNPQRSTVWY
```

Sequences must have length at least 6, matching the minimum length used by the 20-feature featurizer.

Ambiguous or non-standard symbols such as `X`, `B`, `Z`, `U` and `O` are rejected.

## Notes and Limitations

- This mode uses the pretrained published model. It does not retrain LightGBM.
- This mode only supports the 20 features implemented in the fast `Featurizer`.
- The optimizer is intended to propose computational candidates, not experimentally validated CPPs.
- The anomaly score should be used as a warning signal for out-of-distribution candidates.
- For large optimization campaigns, use full-size settings and save outputs to CSV.

## Troubleshooting

If dependencies are missing:

```bash
pip install -r requirements-optimizer.txt
```

If you need the exact tested lightweight environment:

```bash
pip install -r requirements-optimizer-pinned.txt
```

If the script reports a model/featurizer mismatch, the selected model is not compatible with the current 20-feature featurizer. Use `models/LightCPP_20.pickle` or provide a model trained on the same 20 features in the same order.
