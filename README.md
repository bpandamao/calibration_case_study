# calibration_case_study


For the purpose of showing the calibration process of the Bayesian inference in gravitational wave astronomy.

Here, we demonstrate the process of "correcting" the Bayesian credible interval for a 1d toy model in the paper[paper_link]. Starting from the simulation for the training set, going through the training process, and testing the proposed model at the test signal d_{o} (the test signal is generated without noise), the calibrated region at a nominal level of 80% is obtained and displayed in the plot. The results differ slightly from the paper as different seed is used here.

We chose a toy model that describes a chirping waveform.

$h(t;a,f,\dot{f},\epsilon) = a \sin (2\pi t[f + \dot{f}t](1 - \epsilon))$

Here $\epsilon \ll 1$ is used as a tuneable parameter allowing deviations from an exact model $h_{\text{e}}(t;\boldsymbol{\theta},\epsilon = 0)$ given by an approximate model $h_{\text{m}}(t;\boldsymbol{\theta}, \epsilon \neq 0)$. We only consider a single data stream and use the approximate Lisa-like PSD.

Set $\epsilon = 10^{-6}$ as the approximate waveform model $h_{\text{m}}$ while $\epsilon = 0$ for the exact waveform model $h_{\text{e}}$.

We treat $\dot{f}$ as an unknown parameter and set the rest at the default values. So, this is a 1d parameter inference problem.

| parameter | default_value | prior_distribution | prior_range|
|-----------|------------|--------------------|------------|
| $\dot{f}$ | $10^{-8}$  | uniform|$10^{-13}$|
| $a$ | $5\cdot 10^{-21}$  |-|-|
| $f$ | $10^{-3}$  | -|-|

## Code structure
**The steps are as follows:**
Run the notebook step by step. There will be data generated through the process.  

1. [Realised operational coverage estimation](https://github.com/bpandamao/calibration_case_study/blob/main/1d_toy_model_realised_operational_coverage_estimation.ipynb)  
To estimate the operational coverage, we generate samples from the approximate posterior and exact posterior distribution and compute the ratio, which is not practical since the exact posterior distributions are unknown.  

2. Operational coverage estimator  
Without simulations from the exact posterior distribution, we apply a logistic regression to estimate the operational coverage and it is an unbiased estimator.   
2.1 [Sample preparation](https://github.com/bpandamao/calibration_case_study/blob/main/1d_toy_model_operational_estimator_sample_simulation.ipynb)  
2.2 [Dimension reduction and classifier training](https://github.com/bpandamao/calibration_case_study/blob/main/1d_toy_model_estimator_training%20and%20evaluation.ipynb)  

3. [Calibration curve](https://github.com/bpandamao/calibration_case_study/blob/main/1d_toy_model_calibration_curve_and_application.ipynb)  
For a test signal, here $d_O$, a calibration curve could be established to output the "correct" nominal level from the desired nominal level.

## Get started
1. Install Anaconda if you do not have it.
2. Create a virtual environment using:
```
conda create -n mcmc_tutorial -c conda-forge numpy scipy matplotlib corner tqdm jupyter tensorflow keras sklearn statsmodels pandas seaborn   
conda activate mcmc_tutorial
```
