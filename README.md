# CogniSense: Cognitive Fatigue Detection via Keystroke Dynamics

A machine learning project that analyzes keystroke dynamics to discover latent cognitive-behavioral states using unsupervised learning techniques.

## Project Overview

CogniSense uses keystroke timing data from password typing sessions as a proxy for smartphone interaction patterns. The project applies dimensionality reduction and clustering techniques to identify distinct behavioral modes that may correspond to different cognitive states (e.g., fatigue levels, typing confidence, hesitation patterns).

### Key Objectives
- Analyze keystroke dynamics to understand user behavior patterns
- Apply unsupervised learning to discover latent cognitive-behavioral clusters
- Compare multiple clustering approaches (KMeans, GMM, HDBSCAN)

## Dataset

**DSL-StrongPasswordData.csv** - Carnegie Mellon University Keystroke Dynamics Dataset

**Download Link**: [DSL-StrongPasswordData.csv](https://drive.google.com/file/d/1HMTp-Rn2dIfv-tNtfoc_EuZrTHxsPU5p/view?usp=sharing)

| Attribute | Value |
|-----------|-------|
| Total Samples | 20,400 rows |
| Users | 51 subjects |
| Sessions per User | 8 sessions |
| Features | 34 keystroke timing features |
| Feature Types | Hold time (H), Down-Down (DD), Up-Down (UD) timing intervals |

### Feature Description
- **H (Hold Time)**: Duration a key is held down
- **DD (Down-Down)**: Time between pressing two consecutive keys
- **UD (Up-Down)**: Time between releasing one key and pressing the next

## Project Structure

```
CongiSense/
├── CogniSense_AML.ipynb    # Main Jupyter notebook with complete analysis
├── README.md               # Project documentation
├── DSL-StrongPasswordData.csv  # Dataset (upload required)
└── Output files (generated):
    ├── pca_transformed_data.csv    # PCA-transformed features
    └── pca_loadings.csv            # PCA component loadings
```

## Methodology

The analysis follows a structured pipeline:

### Process 1: Data Exploration (EDA)
- Dataset overview and structure analysis
- Missing value detection
- Statistical summary and distribution analysis
- User typing behavior profiling (fast vs slow typists)

### Process 2: Data Preprocessing
- Session-level aggregation (mean of timing features)
- Robust scaling using RobustScaler (handles outliers)
- Winsorization at 1st and 99th percentiles
- Feature selection (31 final features)

### Process 3: Dimensionality Reduction (PCA)
- Principal Component Analysis for feature reduction
- Kaiser criterion and scree plot analysis
- 9 principal components retained (~88% variance explained)
- PC1: Overall typing speed
- PC2: Hold time patterns
- PC3+: Transition-specific behaviors

### Process 4: Clustering (Latent Behavior Discovery)

#### 4.1 KMeans Clustering (Baseline)
- Elbow method and silhouette analysis
- Optimal k=2 clusters identified
- Limitation: Dominated by PC1 (speed), misses nuanced patterns

#### 4.2 Gaussian Mixture Models (GMM)
- Probabilistic clustering with full covariance
- BIC-based model selection
- Optimal k=4 clusters
- Captures overlapping behavioral states

#### 4.3 UMAP + HDBSCAN (Advanced)
- UMAP for nonlinear dimensionality reduction
- HDBSCAN for density-based clustering
- Identifies outliers and noise points

### Discovered Behavioral Clusters (GMM Results)

| Cluster | Behavioral State | Characteristics |
|---------|-----------------|-----------------|
| 0 | Relaxed–Fluent | Low hesitation, smooth transitions |
| 1 | Slow–Hesitant | High latency, careful typing |
| 2 | Fast–Confident | Rapid, consistent patterns |
| 3 | Transitioning | Mixed/intermediate behavior |

## Dependencies

```
pandas>=1.0.0
numpy>=1.18.0
matplotlib>=3.2.0
seaborn>=0.10.0
scikit-learn>=0.24.0
umap-learn>=0.5.0
hdbscan>=0.8.0
```

## How to Run

### Option 1: Google Colab (Recommended)

1. **Open in Colab**: Click the "Open in Colab" badge at the top of the notebook or visit:
   ```
   https://colab.research.google.com/github/hemangtk/CongiSense/blob/main/CogniSense_AML.ipynb
   ```

2. **Download & Upload Dataset**: 
   - Download the dataset from [Google Drive](https://drive.google.com/file/d/1HMTp-Rn2dIfv-tNtfoc_EuZrTHxsPU5p/view?usp=sharing)
   - When prompted in the notebook, upload the `DSL-StrongPasswordData.csv` file

3. **Run All Cells**: Execute cells sequentially (Runtime → Run all)

### Option 2: Local Environment

1. **Clone the Repository**
   ```bash
   git clone https://github.com/hemangtk/CongiSense.git
   cd CongiSense
   ```

2. **Create Virtual Environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn umap-learn hdbscan jupyter
   ```

4. **Download Dataset**
   - Download `DSL-StrongPasswordData.csv` from [Google Drive](https://drive.google.com/file/d/1HMTp-Rn2dIfv-tNtfoc_EuZrTHxsPU5p/view?usp=sharing)
   - Place it in the project root directory

5. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook CogniSense_AML.ipynb
   ```

6. **Run the Notebook**
   - Execute cells in order
   - Modify the data loading cell to read from local file instead of upload

## Results Summary

- **PCA Transformation**: Reduced 31 features to 9 principal components
- **GMM Clustering**: Identified 4 distinct behavioral clusters
- **Key Insight**: Typing behavior reflects underlying cognitive states that can be discovered through unsupervised learning

## Future Work

- Integrate with real-time keystroke monitoring
- Extend to full smartphone interaction dynamics
- Validate cluster labels with ground-truth fatigue annotations
- Develop predictive model for cognitive state classification

## References

- Killourhy, K.S. and Maxion, R.A. (2009). "Comparing Anomaly-Detection Algorithms for Keystroke Dynamics"
- CMU Keystroke Dynamics Dataset

## License

This project is for educational and research purposes.

## Author

Hemang TK
