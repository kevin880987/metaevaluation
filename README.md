# Metaevaluation: Real Options for Prognostics and Health Management

This repository contains the implementation for the paper: **"Metaevaluation: A Comprehensive Evaluation of Health Indicator on Real Options-Based Maintenance Scheduling and Health Prognostics of Bearing Degradation."**

This project proposes an end-to-end framework that bridges the gap between traditional Predictive Maintenance (PdM) and economically optimized maintenance decision-making. Instead of just predicting *if* a component will fail, this framework determines the *optimal time* to perform maintenance to maximize profitability by balancing production revenue against failure risk.

## Framework Overview

The core of this work is a three-phase framework that transforms raw sensor data into an optimal, profit-maximizing maintenance schedule.

<img width="879" height="515" alt="image" src="https://github.com/user-attachments/assets/d439f9c1-57dd-4a67-b7f2-008a5dc0d541" />

1.  **Phase 1: Health Indicator (HI) Construction & Metaevaluation**
    * **Construction**: We build a portfolio of diverse Health Indicators (HIs) from raw sensor data , using techniques like Principal Component Analysis (PCA), Independent Component Analysis (ICA), and a Convolutional Variational Autoencoder (CVAE).
    * **Metaevaluation**: We introduce a "metaevaluation" (evaluation of evaluation) process to score each HI. We use five critical metrics: **Monotonicity, Prognosability, Trendability, Reliability, and Internal Robustness**. This allows for an informed selection of HIs based on their quality.

2.  **Phase 2: Stochastic Deterioration Modeling**
    * The framework uses the **Exponential Wiener Process (EWP)** to model the degradation paths of the HIs.
    * We implement two advanced EWP variants to capture complex, non-linear degradation patterns:
        * **EWVOI**: Exponential Wiener Process with **Variance of Increments**, using a Markov chain to model different variance states (low, normal, high).
        * **EWCD**: Exponential Wiener Process with **Concept Drift**, which segments the HI data using KSWIN drift detection to model shifts in statistical properties.
    * This phase runs **10,000 Monte Carlo simulations** to generate a probabilistic forecast of the degradation, which is essential for quantifying risk.

3.  **Phase 3: Real Options-Based Decision-Making**
    * We apply a **"Hold-to-Sustain" Real Options** valuation, a technique from finance, to maintenance.
    * This model calculates the economic value (`Ot`) of delaying maintenance by weighing the **Expected Revenue (`Rt`)** from extended operation against the **Expected Costs (`Ct`)** of potential failure (including repairs, penalties, and downtime).
    * The framework identifies the **Optimal Maintenance Time (t\*)** that maximizes the total expected profit.

## Key Features

This repository includes implementations for:

* **HI Construction**: PCA, ICA, and CVAE-based HIs (e.g., reconstruction error, latent space distance) using Mahalanobis Distance and Hotelling's T-squared.
* **Metaevaluation**: The five-metric evaluation framework (Monotonicity, Prognosability, Trendability, Reliability, Internal Robustness).
* **Deterioration Modeling**: EWVOI and EWCD models for probabilistic forecasting.
* **Decision-Making**: The "Hold-to-Sustain" real options valuation model.


## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/kevin880987/metaevaluation.git](https://github.com/kevin880987/metaevaluation.git)
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

To run the full framework pipeline (HI construction, metaevaluation, modeling, and decision-making), execute the main script:

```bash
python src/main.py

You can configure the parameters, select the dataset, and choose which models to run by modifying the variables in src/configuration.py.


Key Results
The "Metaevaluation" framework provides a comprehensive view of HI quality and leads to superior economic decisions.

1. Metaevaluation Radar Charts
The 5-metric evaluation reveals that no single HI is perfect. For example, in Dataset A, PC1 has high Trendability and Internal Robustness, while IC_MD is stronger in Prognosability. Our framework uses a weighted ensemble of these HIs to make a balanced decision.


2. Deterioration Modeling
The EWP-based models generate 10,000 simulated degradation paths (grey area) to forecast the probability of failure over time, capturing the system's uncertainty.


3. Optimal Maintenance Decision
The real options model identifies the exact time to perform maintenance that maximizes total profit (the peak of the green line). This approach consistently outperforms a simple "run-to-failure" strategy (profit drops at the end) or a fixed-threshold policy.


Our results show the proposed framework achieves the highest total profit in 4 out of 6 runs on Dataset A and 3 out of 5 runs on Dataset B , with profit improvements of at least 29% compared to running until failure.


Citation
If you use this code or framework in your research, please cite our paper:
@article{chen2025metaevaluation,
  title={Metaevaluation: A Comprehensive Evaluation of Health Indicator on Real Options-Based Maintenance Scheduling and Health Prognostics of Bearing Degradation},
  author={Chen, Yen-Wen and Lee, Chia-Yen and Chu, Pin-Chi and Han, Te},
  journal={International Journal of Production Research},
  year={2025}
}
