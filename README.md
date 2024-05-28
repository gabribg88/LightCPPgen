# LightCPPgen: An Explainable Machine Learning Pipeline for Rational Design of Cell Penetrating Peptides

This github repository contains the Python code to reproduce the results of the paper LightCPPgen: An Explainable Machine Learning Pipeline for Rational Design of Cell Penetrating Peptides by Gabriele Maroni, Filip Stojceski, Lorenzo Pallante, Marco A. Deriu, Dario Piga, Gianvito Grasso.

This paper has been submitted for publication in *Nature communications*.

<img src="figures/Mutation_matrix.pdf" width="500">

![License Badge](https://img.shields.io/badge/license-MIT-blue)

## Abstract
Common regularization algorithms for linear regression, such as LASSO and Ridge regression, rely on a regularization hyperparameter that balances the tradeoff between minimizing the fitting error and the norm of the model coefficients. As this hyperparameter is scalar, it can be easily  selected via random or grid search optimizing a cross-validation criterion. However, using a scalar hyperparameter limits the algorithm's degrees of freedom. In this paper, we address the problem of linear regression with  $\ell_2$-regularization, where a different regularization hyperparameter is associated with each input variable. We optimize these hyperparameters using a gradient-based approach, wherein the gradient of a cross-validation criterion with respect to the regularization hyperparameters is computed analytically through matrix differential calculus. Additionally, we introduce two strategies tailored for sparse model learning problems aiming at reducing the risk of overfitting to the validation data. Numerical examples demonstrate that our multi-hyperparameter regularization approach outperforms LASSO, Ridge, and Elastic Net regression. Moreover, the analytical computation of the gradient proves to be more efficient in terms of computational time compared to automatic differentiation, especially when handling a large number of input variables. Application to the identification of over-parameterized Linear Parameter-Varying models is also presented.

## Software implementation
All the source code used to generate the results and figures in the paper are in the `src` and `notebooks` folders. Computations and figure generation are all run inside [Jupyter notebooks](http://jupyter.org/). Results generated by the code are saved in `results` folder.

## Getting the code
You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/gabribg88/Multiridge.git

or [download a zip archive](https://github.com/gabribg88/Multiridge/archive/refs/heads/master.zip).

## Requirements
You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `requirements.txt`.

We recommend to use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install the dependencies without causing conflicts, with your
setup (even with different Python versions), with the pip package-management system.

Run the following command in the repository folder (where `requirements.txt`
is located) to create a separate environment and install all required
dependencies in it:

    conda create --name <env_name>
    source activate <env_name>
    pip install -r requirements.txt

## Reproducing the results
Before running any code you must activate the conda environment:

    source activate <env_name>

This will enable the environment for your current terminal session. Any subsequent commands will use software that is installed in the environment.
To reproduce the results of the paper we recommend to execute the Jupyter notebooks individually. To do this, you must first start the notebook server by going into the repository top level and running:

    jupyter lab

This will start the server and open your default web browser to the Jupyter interface. In the page, go into the notebooks folder and select the notebook that you wish to view/run.
The notebook is divided into cells (some have text while other have code). Each cell can be executed using Shift + Enter. Executing text cells does nothing and executing code cells runs the code and produces it's output. To execute the whole notebook, run all cells in order.

To reproduce the results depicted in Figure 1 of the paper you have to run the following notebooks:

    1_Massive_experiments_baselines.ipynb : Reproduces the results for the baseline models 
    2_Massive_experiments_multiridge.ipynb : Reproduce the results for the MultiRidge model
    3_Results_visualization.ipynb : Reproduce Figure 1

Note that, for every single experiment, the notebook 1_Massive_experiments_baselines.ipynb will produce a monitoring dashboard of this type:
  
![Image](results/baselines/Experiments_MASSIVE/images/MASSIVE010.png)

The notebook 2_Massive_experiments_multiridge.ipynb will produce a monitoring dashboard of this type:

![Image](results/multiridge/Experiments_MASSIVE/images/MASSIVE010.png)

You can choose to print and/or save these dashboard at every iteration modyfing the parameters of `plot_monitoring_dashboard` function.

To reproduce the results depicted in Figure 2 of the paper you have to run the following notebooks:

    4_Time_analysis_float64.ipynb :  Reproduce panel (a) in Figure 2
    4_Time_analysis_float32.ipynb : Reproduce panel (b) in Figure 2

Finally, to reproduce the results depicted in Figure 3 of the paper you have to run the following notebook:

    5_LPV_intensive.ipynb : Reproduce Figure 3

## Citation

If you use this code or our findings in your research, please cite:

```
@article{Maroni2023,
  title={Gradient-based bilevel optimization for multi-penalty ridge regression through matrix differential calculus},
  author={Gabriele Maroni, Loris Cannelli and Dario Piga},
  journal={Submitted to Automatica},
  year={2023},
}
```

## License

This project is licensed under the terms of the MIT license.

## Acknowledgments





