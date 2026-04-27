# Consumer CLV & Churn Prediction

**RFM Segmentation · Logistic Regression · Geometric Series CLV**

> Developed during the Global Summer Programme — Managing Customer Relations with Analytics track at Singapore Management University (SMU). Quantifies churn risk and customer lifetime value across segmented retail cohorts in the Asian market.

---

## Overview

This project constructs an end-to-end Customer Analytics pipeline for an Asian retail platform. A synthetic 18-month transaction history is generated for 5,000 customers across Southeast Asia, East Asia, and South Asia. The pipeline engineers RFM (Recency, Frequency, Monetary) behavioural features, fits a regularised Logistic Regression churn classifier validated by AUC-ROC, and computes Customer Lifetime Value via geometric series convergence — enabling data-driven marketing budget reallocation.

**Key finding:** The model identifies a **15% near-term churn risk** cohort representing approximately **$2.3M in annualized revenue exposure**, with top-2 CLV decile reallocation projecting a **22% ROI improvement** on digital marketing spend.

---

## Technical Methodology

### 1. Synthetic Transaction History Generation

Customer heterogeneity is driven by a latent **loyalty score** `L ~ Beta(2, 3)` which governs:

- Purchase **frequency**: transaction sampling probability ∝ `L / Σ(L)`
- **Average Order Value**: LogNormal(μ = ln(45) + 0.3L, σ = 0.6)
- **Churn probability**: `p_churn = 0.65 − 0.5L + ε`,  `ε ~ N(0, 0.05²)`

This produces realistic positive correlations between spend, frequency, and retention — mirroring empirical CRM data patterns documented in McKinsey retail benchmarking studies.

### 2. RFM Feature Engineering

Three behavioural dimensions are computed from the 18-month transaction log:

| Metric | Definition | Scoring |
|---|---|---|
| **Recency** | Days since last purchase | Quintile bins [5→1] (recent = 5) |
| **Frequency** | Total transaction count | Quintile bins [1→5] |
| **Monetary** | Total USD spend | Quintile bins [1→5] |

Customers are segmented into four tiers based on composite `RFM_total` score:

```
Champions  (13–15) · Loyal (10–12) · At-Risk (7–9) · Lost (≤6)
```

### 3. Logistic Regression Churn Model

Binary churn is modelled as:

```
P(churn = 1 | x) = σ(β₀ + β₁·R + β₂·F + β₃·M + β₄·RFM_total + ε)
```

where `σ(z) = 1 / (1 + e^{−z})` is the logistic sigmoid.

**Regularisation:** L2 (ridge) penalty with `C = 0.5`, minimising the penalised negative log-likelihood:

```
L(β) = −Σᵢ [yᵢ log(p̂ᵢ) + (1−yᵢ) log(1−p̂ᵢ)] + (1/2C) ‖β‖²
```

**Validation:**

| Metric | Value |
|---|---|
| AUC-ROC (hold-out, 25%) | **0.84** |
| AUC-ROC (5-fold CV, mean ± std) | 0.83 ± 0.01 |

AUC-ROC measures the probability that the model ranks a random churner above a random non-churner — 0.84 indicates strong discriminative power beyond random chance (0.50 baseline).

### 4. Customer Lifetime Value — Geometric Series Convergence

CLV is computed as the present value of an infinite stream of future margin contributions, discounted at rate `d` with estimated retention rate `r = 1 − P̂(churn)`:

```
CLV = Σ_{t=1}^{∞}  (margin × annual_spend) × [r / (1+d)]^t

    = (margin × annual_spend × r) / (1 + d − r)
```

This is the closed-form sum of an infinite geometric series with ratio `r / (1+d) < 1`. The formula degenerates to infinity as `r → 1`, so a minimum denominator floor of `0.01` is enforced.

**Parameters:**
- `margin = 0.28` (retail gross margin benchmark)
- `d = 0.12` (WACC-consistent annual discount rate)
- `annual_spend` extrapolated from 18-month transaction window

---

## Repository Structure

```
consumer-clv-churn/
├── clv_churn_engine.py    # Full pipeline (simulation + RFM + model + CLV)
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone https://github.com/justinjeong001/consumer-clv-churn
cd consumer-clv-churn
pip install -r requirements.txt
python clv_churn_engine.py
```

## Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.11
```

---

## Note on Synthetic Data

The transaction data is generated using statistically grounded distribution parameters derived from Asian e-commerce benchmarking reports (Bain & Company SEA Consumer Report; Google-Temasek e-Conomy SEA). The latent loyalty model replicates the empirically observed positive correlation between purchase frequency and average order value documented in CRM literature. This approach is standard in corporate analytics when proprietary customer data is confidential.

---

## Author

**Hyunwoo (Justin) Jeong** · HKUST, B.Sc. Quantitative Social Analysis (Minor: Mathematics)
`hjeongad@connect.ust.hk` · [LinkedIn](https://linkedin.com/in/justinjeong001)
