# Project Skills & Technical Competencies

## 1. Advanced Econometric Modeling & Machine Learning
This project implements a hybrid analytical approach, combining traditional econometric rigor with modern machine learning predictive power.

### **Traditional Gravity Models**
*   **PPML (Poisson Pseudo Maximum Likelihood)**: 
    *   Addressed the "zeros" problem in trade data and heteroskedasticity.
    *   Implemented **High-Dimensional Fixed Effects** (Destination, Year, Product) to control for multilateral resistance terms (Anderson & van Wincoop, 2003).
    *   Benchmarked against OLS to demonstrate superiority in handling count data.

### **Machine Learning Algorithms**
*   **Random Forest Regressor**: 
    *   Utilized ensemble learning to capture **non-linear relationships** in trade determinants (e.g., how distance impacts trade differently at varying tariff levels).
    *   Optimized hyperparameters (n_estimators, max_depth) to prevent overfitting.
*   **Predictive Performance Metrics**:
    *   Implemented rigorous evaluation using $R^2$ (Coefficient of Determination), RMSE (Root Mean Square Error), and MAE.
    *   **Result**: ML models demonstrated superior predictive accuracy ($R^2 \approx 0.86$) compared to traditional econometric baselines ($R^2 \approx 0.72$).

### **Explainable AI (XAI)**
*   **SHAP (SHapley Additive exPlanations)**:
    *   Applied Game Theoretic approach to interpret "Black Box" ML models.
    *   Quantified the marginal contribution of features like **Tariff Rates, Distance, and GDP** to the prediction of trade volume.

---

## 2. Data Engineering & Pipeline Architecture
Built a robust, reproducible ETL (Extract, Transform, Load) pipeline.

*   **API Integration & Automated Data Collection**:
    *   **UN Comtrade API**: Automated retrieval of bilateral trade flows (HS6 Code level).
    *   **World Bank WDI**: Integrated macroeconomic indicators (GDP, Population, Internet Penetration).
    *   **WTO / Rule-Based Tariffs**: Implemented complex logic to simulate/fetch real-world tariff structures, including:
        *   **Sin Taxes**: Prohibition detection for specific HS codes (Pork, Alcohol).
        *   **Strategic Exceptions**: Saudi Arabia's exceptions for poultry and dairy.
        *   **Food Security Exemptions**: Zero-tariff logic for essential grains.
*   **Data Processing (`DataProcessor`)**:
    *   **Panel Data Construction**: Merged disparate datasets on `(Year, Importer, Product)` keys.
    *   **Feature Engineering**: Created log-transformed variables (`ln_gdp`, `log_distance`) for econometric compatibility.
    *   **Caching Mechanism**: Implemented local caching to optimize API calls and ensure offline reproducibility.

---

## 3. Visualization & Interactive Dashboarding
Developed a user-centric interface to democratize access to complex model insights.

*   **Streamlit Framework**: 
    *   Built a responsive Single-Page Application (SPA).
*   **Geospatial Analytics**:
    *   **Choropleth Maps**: Visualized trade intensity and market diversification across the GCC region using `Geopandas` and `Folium`.
*   **Interactive Charts**:
    *   Dynamic bar charts for model comparison.
    *   Scatter plots for Actual vs. Predicted analysis.

---

## 4. Software Engineering Best Practices
*   **Modular Architecture**: Separated concerns into `DataCollector`, `DataProcessor`, and `ModelEngine` classes.
*   **Version Control**: Utilized **Git** for tracking experiments and code evolution.
*   **Robust Error Handling**: Implemented fallback mechanisms (e.g., simulated tariff logic) to ensure pipeline stability during API outages.
