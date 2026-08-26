# Custom Dataset Pipeline Mode

This mode reuses the LightCPPgen pipeline on a CPP/non-CPP dataset different from MLCPP2.0.

Use this mode when you want to train a new model from your own labeled peptide sequences while keeping the same general workflow used in the paper: FASTA creation, descriptor calculation, LightGBM training, feature selection, SHAP interpretation and, when compatible, sequence optimization.

## Expected Dataset

The minimum required information is:

```text
ID
Sequence
CPP
```

where:

```text
ID        unique peptide identifier
Sequence  amino-acid sequence
CPP       binary label; 1 for CPP, 0 for non-CPP
```

Example:

```text
ID,Sequence,CPP
pep_001,GEPWKVCVN,0
pep_002,RQIKIWFQNRRMKWKK,1
```

Sequences should use standard amino-acid symbols:

```text
ACDEFGHIKLMNPQRSTVWY
```

Ambiguous or non-standard symbols should be removed, corrected or handled explicitly before feature engineering.

## Recommended Workflow

### 1. Prepare Input Data

Prepare a clean CSV file with one row per peptide and at least the columns:

```text
ID,Sequence,CPP
```

Decide whether you have:

- only a training dataset;
- a training dataset and an independent test dataset;
- multiple datasets that should remain separated.

### 2. Create FASTA and Label Files

The original pipeline expects FASTA files and separate label CSV files similar to:

```text
data/MLCPP2_Training.fasta
data/MLCPP2_TrainingCPPvalues.csv
data/MLCPP2_Independent.fasta
data/MLCPP2_IndependentCPPvalues.csv
```

For a custom dataset, create analogous files, for example:

```text
data/custom_training.fasta
data/custom_training_CPPvalues.csv
data/custom_independent.fasta
data/custom_independent_CPPvalues.csv
```

The appendix notebook `notebooks/APPENDIX_Fasta_creation.ipynb` shows the historical FASTA creation logic. A cleaner command-line helper can be added later for custom datasets.

### 3. Run Feature Engineering

Adapt `notebooks/0_Feature_engineering.ipynb` to point to the custom FASTA and label files.

This step computes:

- physicochemical descriptors;
- sequence-based descriptors through `iFeatureOmegaCLI`;
- combined feature matrices saved to `features/`.

This is the most dependency-sensitive part of the pipeline.

### 4. Train LightGBM Models

Adapt `notebooks/1_Modeling.ipynb` to load the custom generated features.

The notebook can be reused for:

- cross-validation;
- model training;
- MDI-based feature filtering;
- forward feature selection;
- final model refitting.

The final trained model should be saved with enough metadata to identify:

- selected feature names;
- feature order;
- model parameters;
- best iteration;
- training data version;
- package environment.

### 5. Interpret the Model

Adapt `notebooks/2_Global_model_interpretation.ipynb` for global SHAP interpretation.

For local interpretation of selected peptides, adapt:

```text
notebooks/5_Local_analysis_optimized_sequences.ipynb
```

### 6. Decide Whether the Optimizer Is Compatible

This is the critical difference between the paper model and a custom model.

The current fast optimizer can compute only the 20 features selected in the published LightCPPgen model. Therefore:

- if the custom model uses the same 20 features in the same order, the current optimizer can be reused;
- if the custom model uses a different feature set, the current optimizer is not automatically valid;
- if the custom model uses arbitrary full-pipeline descriptors, a matching candidate-sequence featurizer must be implemented before optimization.

## Optimizer Compatibility Options

### Option A: Train an Optimizer-Compatible Model

Restrict training to the 20 features implemented by the current `Featurizer`.

Advantages:

- direct use of the existing optimizer;
- fast candidate scoring;
- simple deployment.

Disadvantages:

- may not be the best feature subset for the new dataset.

### Option B: Train the Best Custom Model, Then Add a Matching Featurizer

Allow feature selection to choose any descriptor, then implement a custom featurizer able to compute exactly those selected features for newly generated sequences.

Advantages:

- best match to the new dataset;
- more flexible scientifically.

Disadvantages:

- more engineering work;
- optimization may become slower if features depend on external descriptor tools.

### Option C: Use the Full Descriptor Pipeline During Optimization

In principle, every candidate sequence could be passed through the full descriptor pipeline.

This is usually not recommended for genetic optimization because thousands of candidates must be scored repeatedly. It is more appropriate for batch scoring than for iterative optimization.