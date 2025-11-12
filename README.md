# Metaevaluation: Real Options for Prognostics and Health Management

[cite_start]This repository contains the implementation for the paper: **"Metaevaluation: A Comprehensive Evaluation of Health Indicator on Real Options-Based Maintenance Scheduling and Health Prognostics of Bearing Degradation."** [cite: 1]

[cite_start]This project proposes an end-to-end framework that bridges the gap between traditional Predictive Maintenance (PdM) and economically optimized maintenance decision-making[cite: 47, 555]. [cite_start]Instead of just predicting *if* a component will fail, this framework determines the *optimal time* to perform maintenance to maximize profitability by balancing production revenue against failure risk[cite: 46, 49].

## Framework Overview

[cite_start]The core of this work is a three-phase framework that transforms raw sensor data into an optimal, profit-maximizing maintenance schedule[cite: 185].

![Framework Diagram](figures/fig1.png)

1.  **Phase 1: Health Indicator (HI) Construction & Metaevaluation**
    * [cite_start]**Construction**: We build a portfolio of diverse Health Indicators (HIs) from raw sensor data [cite: 186][cite_start], using techniques like Principal Component Analysis (PCA), Independent Component Analysis (ICA), and a Convolutional Variational Autoencoder (CVAE)[cite: 10, 215].
    * [cite_start]**Metaevaluation**: We introduce a "metaevaluation" (evaluation of evaluation) process to score each HI[cite: 9, 29]. [cite_start]We use five critical metrics: **Monotonicity, Prognosability, Trendability, Reliability, and Internal Robustness**[cite: 9, 38, 187]. [cite_start]This allows for an informed selection of HIs based on their quality[cite: 188].

2.  **Phase 2: Stochastic Deterioration Modeling**
    * [cite_start]The framework uses the **Exponential Wiener Process (EWP)** to model the degradation paths of the HIs[cite: 11, 41, 293].
    * We implement two advanced EWP variants to capture complex, non-linear degradation patterns:
        * [cite_start]**EWVOI**: Exponential Wiener Process with **Variance of Increments**, using a Markov chain to model different variance states (low, normal, high)[cite: 307, 309].
        * [cite_start]**EWCD**: Exponential Wiener Process with **Concept Drift**, which segments the HI data using KSWIN drift detection to model shifts in statistical properties[cite: 160, 333, 335, 336].
    * [cite_start]This phase runs **10,000 Monte Carlo simulations** to generate a probabilistic forecast of the degradation, which is essential for quantifying risk[cite: 191, 348].

3.  **Phase 3: Real Options-Based Decision-Making**
    * [cite_start]We apply a **"Hold-to-Sustain" Real Options** valuation, a technique from finance, to maintenance[cite: 44, 190, 362].
    * [cite_start]This model calculates the economic value (`Ot`) of delaying maintenance by weighing the **Expected Revenue (`Rt`)** from extended operation against the **Expected Costs (`Ct`)** of potential failure (including repairs, penalties, and downtime)[cite: 195, 371].
    * [cite_start]The framework identifies the **Optimal Maintenance Time (t\*)** that maximizes the total expected profit[cite: 196, 377].

## Key Features

This repository includes implementations for:

* [cite_start]**HI Construction**: PCA [cite: 216][cite_start], ICA [cite: 218][cite_start], and CVAE-based HIs [cite: 217] (e.g., reconstruction error, latent space distance) using Mahalanobis Distance and Hotelling's T-squared[cite: 248, 256].
* [cite_start]**Metaevaluation**: The five-metric evaluation framework (Monotonicity, Prognosability, Trendability, Reliability, Internal Robustness)[cite: 260, 266].
* [cite_start]**Deterioration Modeling**: EWVOI [cite: 307] [cite_start]and EWCD [cite: 333] models for probabilistic forecasting.
* [cite_start]**Decision-Making**: The "Hold-to-Sustain" real options valuation model[cite: 362].
* **Benchmark Models**: Implementations of **CAE-M** [cite: 484] and **VQ-VAE** [cite: 486] for comparison.
* **Datasets**: Full processing pipelines for:
    * **Dataset A**: IEEE PHM 2012 Data Challenge (PRONOSTIA)[cite: 398].
    * [cite_start]**Dataset B**: Electrical Fatigue Destruction Dataset[cite: 399].

## Repository Structure
. ├── Dataset_A_IEEE_PHM_2012_Data_Challenge/ # Data files for Dataset A ├── Dataset_B_Electrical_Fatigue_Destruction_Dataset/ # Data files for Dataset B ├── figures/ # Figures from the paper ├── src/ │ ├── HI_Construction.py # Implements PCA, ICA, CVAE HIs │ ├── Metaevaluation.py # Implements the 5-metric evaluation │ ├── CVAE.py # CVAE model architecture │ ├── EWCD.py # EWCD deterioration model │ ├── EWVOI.py # EWVOI deterioration model │ ├── real_option.py # Hold-to-Sustain option valuation │ ├── CAEM.py # Benchmark: CAE-M model │ ├── VQVAE.py # Benchmark: VQ-VAE model │ ├── dataset_A.py # Data loading/processing for Dataset A │ ├── dataset_B.py # Data loading/processing for Dataset B │ ├── main.py # Main script to run the framework │ └── utils.py # Utility functions (e.g., KSWIN) ├── requirements.txt # Python dependencies └── README.md # This file

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/kevin880987/metaevaluation.git](https://github.com/kevin880987/metaevaluation.git)
    cd metaevaluation
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  Download the datasets and place them in the respective `Dataset_A_...` and `Dataset_B_...` folders.
    * **Dataset A**: [IEEE PHM 2012 Data Challenge](https://phm-ieee.org/competition/2012-phm-ieee-data-challenge) [cite: 696]
    * **Dataset B**: [Electrical Fatigue Destruction Dataset](https://www.mdpi.com/1996-1073/12/5/801) [cite: 698]

## How to Run

To run the full framework pipeline (HI construction, metaevaluation, modeling, and decision-making), execute the main script:

```bash
python src/main.py

You can configure the parameters, select the dataset, and choose which models to run by modifying the variables at the beginning of src/main.py.


Key Results
The "Metaevaluation" framework provides a comprehensive view of HI quality and leads to superior economic decisions.

1. Metaevaluation Radar Charts
The 5-metric evaluation reveals that no single HI is perfect. For example, in Dataset A, PC1 has high Trendability and Internal Robustness, while IC_MD is stronger in Prognosability. Our framework uses a weighted ensemble of these HIs to make a balanced decision.




2. Deterioration Modeling
The EWP-based models generate 10,000 simulated degradation paths (grey area) to forecast the probability of failure over time, capturing the system's uncertainty.


3. Optimal Maintenance Decision
The real options model identifies the exact time to perform maintenance that maximizes total profit (the peak of the green line). This approach consistently outperforms a simple "run-to-failure" strategy (profit drops at the end) or a fixed-threshold policy.


Our results show the proposed framework achieves the highest total profit in 4 out of 6 runs on Dataset A and 3 out of 5 runs on Dataset B , with profit improvements of at least 29% compared to running until failure.

這裏是您可以直接複製並貼到 README.md 檔案中的原始 Markdown 文本：

Markdown

# Metaevaluation: Real Options for Prognostics and Health Management

[cite_start]This repository contains the official implementation for the paper: **"Metaevaluation: A Comprehensive Evaluation of Health Indicator on Real Options-Based Maintenance Scheduling and Health Prognostics of Bearing Degradation."** [cite: 1]

[cite_start]This project proposes an end-to-end framework that bridges the gap between traditional Predictive Maintenance (PdM) and economically optimized maintenance decision-making[cite: 47, 555]. [cite_start]Instead of just predicting *if* a component will fail, this framework determines the *optimal time* to perform maintenance to maximize profitability by balancing production revenue against failure risk[cite: 46, 49].

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

## Framework Overview

[cite_start]The core of this work is a three-phase framework that transforms raw sensor data into an optimal, profit-maximizing maintenance schedule[cite: 185].

![Framework Diagram](figures/fig1.png)

1.  **Phase 1: Health Indicator (HI) Construction & Metaevaluation**
    * [cite_start]**Construction**: We build a portfolio of diverse Health Indicators (HIs) from raw sensor data [cite: 186][cite_start], using techniques like Principal Component Analysis (PCA), Independent Component Analysis (ICA), and a Convolutional Variational Autoencoder (CVAE)[cite: 10, 215].
    * [cite_start]**Metaevaluation**: We introduce a "metaevaluation" (evaluation of evaluation) process to score each HI[cite: 9, 29]. [cite_start]We use five critical metrics: **Monotonicity, Prognosability, Trendability, Reliability, and Internal Robustness**[cite: 9, 38, 187]. [cite_start]This allows for an informed selection of HIs based on their quality[cite: 188].

2.  **Phase 2: Stochastic Deterioration Modeling**
    * [cite_start]The framework uses the **Exponential Wiener Process (EWP)** to model the degradation paths of the HIs[cite: 11, 41, 293].
    * We implement two advanced EWP variants to capture complex, non-linear degradation patterns:
        * [cite_start]**EWVOI**: Exponential Wiener Process with **Variance of Increments**, using a Markov chain to model different variance states (low, normal, high)[cite: 307, 309].
        * [cite_start]**EWCD**: Exponential Wiener Process with **Concept Drift**, which segments the HI data using KSWIN drift detection to model shifts in statistical properties[cite: 160, 333, 335, 336].
    * [cite_start]This phase runs **10,000 Monte Carlo simulations** to generate a probabilistic forecast of the degradation, which is essential for quantifying risk[cite: 191, 348].

3.  **Phase 3: Real Options-Based Decision-Making**
    * [cite_start]We apply a **"Hold-to-Sustain" Real Options** valuation, a technique from finance, to maintenance[cite: 44, 190, 362].
    * [cite_start]This model calculates the economic value (`Ot`) of delaying maintenance by weighing the **Expected Revenue (`Rt`)** from extended operation against the **Expected Costs (`Ct`)** of potential failure (including repairs, penalties, and downtime)[cite: 195, 371].
    * [cite_start]The framework identifies the **Optimal Maintenance Time (t\*)** that maximizes the total expected profit[cite: 196, 377].

## Key Features

This repository includes implementations for:

* [cite_start]**HI Construction**: PCA [cite: 216][cite_start], ICA [cite: 218][cite_start], and CVAE-based HIs [cite: 217] (e.g., reconstruction error, latent space distance) using Mahalanobis Distance and Hotelling's T-squared[cite: 248, 256].
* [cite_start]**Metaevaluation**: The five-metric evaluation framework (Monotonicity, Prognosability, Trendability, Reliability, Internal Robustness)[cite: 260, 266].
* [cite_start]**Deterioration Modeling**: EWVOI [cite: 307] [cite_start]and EWCD [cite: 333] models for probabilistic forecasting.
* [cite_start]**Decision-Making**: The "Hold-to-Sustain" real options valuation model[cite: 362].
* **Benchmark Models**: Implementations of **CAE-M** [cite: 484] and **VQ-VAE** [cite: 486] for comparison.
* **Datasets**: Full processing pipelines for:
    * **Dataset A**: IEEE PHM 2012 Data Challenge (PRONOSTIA)[cite: 398].
    * [cite_start]**Dataset B**: Electrical Fatigue Destruction Dataset[cite: 399].

## Repository Structure

. ├── Dataset_A_IEEE_PHM_2012_Data_Challenge/ # Data files for Dataset A ├── Dataset_B_Electrical_Fatigue_Destruction_Dataset/ # Data files for Dataset B ├── figures/ # Figures from the paper ├── src/ │ ├── HI_Construction.py # Implements PCA, ICA, CVAE HIs │ ├── Metaevaluation.py # Implements the 5-metric evaluation │ ├── CVAE.py # CVAE model architecture │ ├── EWCD.py # EWCD deterioration model │ ├── EWVOI.py # EWVOI deterioration model │ ├── real_option.py # Hold-to-Sustain option valuation │ ├── CAEM.py # Benchmark: CAE-M model │ ├── VQVAE.py # Benchmark: VQ-VAE model │ ├── dataset_A.py # Data loading/processing for Dataset A │ ├── dataset_B.py # Data loading/processing for Dataset B │ ├── main.py # Main script to run the framework │ └── utils.py # Utility functions (e.g., KSWIN) ├── requirements.txt # Python dependencies └── README.md # This file


## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/kevin880987/metaevaluation.git](https://github.com/kevin880987/metaevaluation.git)
    cd metaevaluation
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  Download the datasets and place them in the respective `Dataset_A_...` and `Dataset_B_...` folders.
    * **Dataset A**: [IEEE PHM 2012 Data Challenge](https://phm-ieee.org/competition/2012-phm-ieee-data-challenge) [cite: 696]
    * **Dataset B**: [Electrical Fatigue Destruction Dataset](https://www.mdpi.com/1996-1073/12/5/801) [cite: 698]

## How to Run

To run the full framework pipeline (HI construction, metaevaluation, modeling, and decision-making), execute the main script:

```bash
python src/main.py
You can configure the parameters, select the dataset, and choose which models to run by modifying the variables at the beginning of src/main.py.

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
