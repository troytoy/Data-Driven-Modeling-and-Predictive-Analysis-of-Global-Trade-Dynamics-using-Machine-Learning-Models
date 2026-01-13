# Thailand-GCC Trade Analysis Thesis Project

**Version:** 6.7 ULTIMATE (Refactored)
**Date:** January 13, 2026

## Overview
This project performs a comprehensive Gravity Model analysis of trade between Thailand and GCC countries (Bahrain, Kuwait, Oman, Qatar, Saudi Arabia, UAE). It integrates data from UN COMTRADE, World Bank WDI, CEPII, and Harvard Atlas to estimate trade potential using PPML and Machine Learning models (Random Forest, XGBoost).

## Features
- **Automated Data Collection**: Fetches trade, GDP, and Population data automatically via APIs.
- **Smart Caching**: Caches API responses to disk to speed up subsequent runs.
- **Advanced Modeling**: Compares PPML (Econometric) vs. Machine Learning approaches.
- **Full Reporting**: Generates LaTeX tables, PNG figures, and Markdown summaries automatically.

## Project Structure
```
thesis_project/
├── data/                  # Raw input data (if manually placed)
├── output/                # Generated results (Figures, Tables, Summary)
├── src/                   # Source code
│   └── main.py            # Main execution script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

1. **Clone or Download** this repository.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Analysis**:
   ```bash
   python src/main.py
   ```
2. **Check Results**:
   - Go to the `Thesis_v6_7_ULTIMATE/` directory (created in your current folder or user home).
   - `summary.json` & `SUMMARY_ENHANCED.md`: Executive summary.
   - `figures/`: Visualizations of trade flows and model performance.
   - `tables/`: CSV files of coefficients and metrics.

## Configuration
Key settings (API Keys, Countries, Years) can be modified in `src/main.py` within the `ThesisConfig` class.

**Note**: For production usage, it is recommended to move API Keys to environment variables.

## Requirements
- Python 3.8+
- Internet connection (for initial data fetch)
