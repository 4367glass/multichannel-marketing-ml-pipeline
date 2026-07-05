# 📈 Multi-Channel Marketing Optimization Pipeline: Classical ML vs. Deep Learning Benchmarking

## 📊 Business Context & Objective
In high-velocity enterprise corporate environments, marketing campaign budgets must be optimized across diverse channels (**Website Ads, Radio Spots, Streaming Commercials, and TV Commercials**) to maximize customer acquisition yields. 

This repository establishes a production-grade machine learning optimization pipeline. It ingests 50,000 raw multi-channel consumer transaction records, enforces strict upstream data governance rules, and benchmarks three distinct algorithmic architectures to predict business conversion. 

By restructuring the pipeline into a Model Championship Platform, this project provides a clear blueprint for deployment to downstream BI tools (Power BI Cloud and Tableau Server), allowing executive stakeholders to make data-driven budget reallocations based on the highest-performing predictive models.

> 🔍 **Responsible AI Cross-Reference:** This repository focus exclusively on architectural data pipeline scaling and multi-model benchmarking. For our framework on algorithmic ethical auditing, model bias validation, and fairness performance metrics, please visit our companion project: **[Responsible AI Governance Repository](PASTE_THE_LINK_TO_YOUR_FIRST_REPOS_HERE)**.

---

## 🛠️ Tech Stack & Enterprise Conventions
* **Core Languages:** Python 3 (Google Colab / Compute Environment).
* **Data Engineering & Preparation:** Pandas, NumPy, Scikit-Learn (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`).
* **Deep Learning Framework:** TensorFlow 2.x / Keras v3 (Sequential Graph Node API).
* **Target BI Environments:** Power BI Cloud (Streaming API), Tableau Server (Compressed Local `.hyper` Data Engine).

---

## 📂 Repository Structure
```text
multichannel-marketing-ml-pipeline/
├── .gitignore              # Watertight isolation filter protecting cloud credentials & raw data
├── README.md               # Executive project documentation & recruiter summary
├── datasets/
│   └── master_marketing_model_predictions.csv # Unified 11-column Master BI dashboard dataset
└── models/
    └── champion_marketing_deep_learning_model.keras # Serialized multi-layer neural network
```

---

## 🔒 Upstream Data Governance Strategy

Standard flat-file data science pipelines frequently introduce artificial training bias by preprocessing metrics purely inside application scripts. This pipeline enforces watertight data governance rules at the ingestion layer before records ever touch the training layers:

1. **Forceful Elimination of Upstream Data Leakage:** Explicitly stripped out `COUPON_CODE_ACTIVATED`, `LAST_CLICK_TIMESTAMP`, and `OUTBOUND_FOLLOWUP_COUNT`. These features are only generated *after* a purchase is finalized. Including them artificially inflates validation metrics while rendering the model useless for predictive pre-campaign targeting.
2. **Distance-Metric Preservation:** Replaced uncontacted flags (`-1`) with proper SQL/Database `NaN/null` properties via `.replace(-1, np.nan)`, preventing distance-based kernel vectors from warping geometric coordinate scales.
3. **Deterministic Deduplication:** Condensed 50,000 raw transactional logs down to 1,110 unique, high-integrity consumer baseline profiles to eliminate duplicate record streaming bias.

---

## 📊 The Model Championship Leaderboard (K=4 Stratified Cross-Validation)

To ensure maximum structural stability across all segments of the marketing data, every model was evaluated using a **4-Fold Stratified Round-Robin Cross-Validation** framework:

### 🥉 1. Baseline: Support Vector Machine (SVM)
* **Architecture:** Radial Basis Function (RBF) Kernel, `class_weight='balanced'`.
* **Mean CV Accuracy:** **56.31%**
* **Diagnostic Analysis:** Failed in production. The SVM struggled to map complex, non-linear text column interactions (`MEDIA_CHANNEL` vs. `INCOME_BRACKET`) when forced to draw flat, multi-dimensional geometric hyperplanes.

### 🥈 2. Challenger: Random Forest Classifier
* **Architecture:** 200 Ensemble Decision Trees, `max_depth=6`.
* **Mean CV Accuracy:** **79.28%**
* **Diagnostic Analysis:** A massive 23% performance improvement over the SVM baseline. The ensemble model effectively handled categorical feature noise by building automated branching rule thresholds.

### 🏆 3. The Champion: Multi-Layer Perceptron (MLP) Neural Network
* **Architecture:** Deep Feedforward Neural Network built in TensorFlow. Formulated with an `Input(shape=(13,))` layer, a dense hidden layer of 32 nodes (ReLU), a secondary interaction hidden layer of 16 nodes (ReLU), and an output node utilizing a Sigmoid activation layer for detailed probability output. 
* **Mean CV Accuracy:** **84.42%**
* **Variance Stability:** **+/- 0.67%** (Exceptional stability, zero evidence of data overfitting).
* **Diagnostic Analysis:** The clear pipeline winner. Layer-by-layer neural weight backpropagation successfully functioned as an automated feature extractor, uncovering deep, non-linear relationships that classical ML methods completely missed. Overfitting was forcefully mitigated by injecting `Dropout(0.2)` layers.

---

## 💻 Final Production Pipeline Execution Preview
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# Unified Deep Learning Graph node definition
dl_final = Sequential([
    Input(shape=(13,)), 
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

dl_final.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_final.fit(X_transformed, y_marketing, epochs=50, batch_size=32, verbose=0)
```

---
## 🎯 BI Dashboard Integration & Executive Delivery
The final consolidated data file, `master_marketing_model_predictions.csv`, tracks predictions concurrently. When loaded into **Power BI**, a dynamic model slicer allows users to toggle decision thresholds on the fly. When connected to **Tableau**, the granular customer acquisition cost can be segmented by region, converting high-level deep learning probabilities into an actionable corporate sales strategy.# multichannel-marketing-ml-pipeline
