import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path

# Optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from utils import logger

class ModelEngine:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results = []

    def run_ppml(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        logger.info("[6/11] Running PPML Estimation...")
        
        formula = (
            "trade_value ~ ln_gdp_ex + ln_gdp_im + ln_pop_ex + ln_pop_im + "
            "log_distance + pci + tariff + internet_im + same_legal + "
            "C(year) + C(importer) + C(hs6)"
        )
        
        try:
            model = smf.glm(formula=formula, data=train_df, family=sm.families.Poisson()).fit()
            
            # Predictions
            test_pred = model.predict(test_df)
            r2 = r2_score(test_df['trade_value'], test_pred)
            rmse = np.sqrt(mean_squared_error(test_df['trade_value'], test_pred))
            mae = mean_absolute_error(test_df['trade_value'], test_pred)
            
            self.results.append({
                'Model': 'PPML', 'R2': r2, 'RMSE': rmse, 'MAE': mae
            })
            logger.info(f"PPML Success. R2: {r2:.4f}")
            
            # Save Coefficients
            coefs = pd.DataFrame({'Coef': model.params, 'P-Value': model.pvalues})
            coefs.to_csv(self.output_dir / 'tables' / 'ppml_coefficients.csv')
            
            return model
        except Exception as e:
            logger.error(f"PPML Failed: {e}")
            return None

    def run_ml_models(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        logger.info("[7/11] Running ML Models...")
        
        features = ['ln_gdp_ex', 'ln_gdp_im', 'ln_pop_ex', 'ln_pop_im',
                    'log_distance', 'pci', 'tariff', 'internet_im', 'same_legal']
        
        X_train, y_train = train_df[features], train_df['trade_value']
        X_test, y_test = test_df[features], test_df['trade_value']
        
        scaler = StandardScaler()
        # Scale only features
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # RandomForest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        self._evaluate(rf, X_test, y_test, 'Random Forest')
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            xgb_mod = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            xgb_mod.fit(X_train, y_train)
            self._evaluate(xgb_mod, X_test, y_test, 'XGBoost')
            self._run_shap(xgb_mod, X_test, features)

    def _run_shap(self, model, X, feature_names):
        try:
            import shap
            import matplotlib.pyplot as plt
            logger.info("Running SHAP Explainer...")
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, show=False, feature_names=feature_names)
            plt.savefig(self.output_dir / 'figures' / 'shap_analysis.png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"SHAP Analysis failed: {e}")
            
    def _evaluate(self, model, X, y_true, name):
        pred = model.predict(X)
        res = {
            'Model': name,
            'R2': r2_score(y_true, pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, pred)),
            'MAE': mean_absolute_error(y_true, pred)
        }
        self.results.append(res)
        logger.info(f"{name} Results: R2={res['R2']:.4f}")

    def save_comparison(self):
        df = pd.DataFrame(self.results).sort_values('R2', ascending=False)
        df.to_csv(self.output_dir / 'tables' / 'model_comparison.csv', index=False)
        logger.info("Model comparison saved.")
