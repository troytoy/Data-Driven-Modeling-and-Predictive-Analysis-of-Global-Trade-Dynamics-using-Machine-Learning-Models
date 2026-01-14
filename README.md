# Data-Driven Modeling and Predictive Analysis of Global Trade Dynamics

**Version:** 2026.01.14-0915
**Last Updated:** 09:15 AM (CET)
**Status:** Active

## Overview
This project leverages Machine Learning and Econometric models to analyze and predict trade dynamics between Thailand and GCC countries. By integrating multi-source data (UN COMTRADE, World Bank, CEPII), the system provides insights into trade potential and economic determinants.

## Key Features
- **Automated Data Pipeline**: Seamless fetching and processing of trade, GDP, and demographic data.
- **Hybrid Modeling**: Utilizes both traditional Gravity Models (PPML) and advanced ML algorithms (Random Forest, XGBoost).
- **Interactive Visualization**: Generates insightful charts and summary statistics automatically.
- **Modular Architecture**: Clean, maintainable code structure ready for scalability.

## Project Structure
```
├── data/                  # Raw input data storage
├── output/                # Generated Analysis Results (Figures, Tables)
├── src/                   # Source Code
│   ├── Data-Driven-Modeling...py  # Main entry point
│   ├── config.py          # Configuration settings
│   ├── models.py          # ML & Statistical models
│   └── ...                # Utility modules
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis**:
   ```bash
   python src/Data-Driven-Modeling-and-Predictive-Analysis.py
   ```

3. **Launch Dashboard** (New!):
   Explore the results interactively:
   ```bash
   streamlit run src/dashboard.py
   ```

4. **View Results**:
   Check the `output/` directory for generated CSV reports and PNG visualizations.

## Technologies
- **Python**: Pandas, NumPy, Scikit-learn, Statsmodels
- **Data Sources**: UN COMTRADE API, World Bank WDI
