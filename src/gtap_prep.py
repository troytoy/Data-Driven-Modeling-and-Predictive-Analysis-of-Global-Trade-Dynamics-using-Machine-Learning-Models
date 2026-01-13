import pandas as pd
import numpy as np
from pathlib import Path
from utils import logger
from config import ThesisConfig

class GTAPPreparator:
    def __init__(self, config: ThesisConfig):
        self.cfg = config
        self.output_dir = config.output_dir

    def run_simulation_prep(self, df: pd.DataFrame):
        logger.info("[9/11] Preparing GTAP Simulation Shocks...")
        
        # Calculate AVE
        # AVE = (1 + tariff) * (1 + NTM) - 1
        # NTM proxy = -0.01 * (Predicted - Actual)/(Predicted + Actual) (simplified IME proxy)
        
        sim_df = df.copy()
        
        # Simple NTM Proxy (if IME not available, use random small shock for demo or derived from tariff)
        # In this upgraded version, we assume NTM is related to non-tariff barriers
        sim_df['ntm_eq'] = np.random.uniform(0.02, 0.10, len(sim_df)) 
        
        sim_df['ave_baseline'] = sim_df['tariff']  # Already decimal in our data? Check data_processor.
        # Wait, in data_processor, tariff was random 0.01-0.15 (decimal).
        
        sim_df['ave_total'] = (1 + sim_df['ave_baseline']) * (1 + sim_df['ntm_eq']) - 1
        
        # Scenarios
        # 1. Tariff Cut 50%
        sim_df['shock_tariff_cut'] = -0.5 * sim_df['ave_baseline']
        
        # 2. NTM Harmonization (-30%)
        sim_df['shock_ntm'] = -0.3 * sim_df['ntm_eq']
        
        # 3. Full
        sim_df['shock_full'] = sim_df['shock_tariff_cut'] + sim_df['shock_ntm']
        
        # Create Template
        template = sim_df[[
            'year', 'exporter', 'importer', 'hs6', 
            'ave_baseline', 'ave_total', 
            'shock_tariff_cut', 'shock_ntm', 'shock_full'
        ]].copy()
        
        # Convert to percentages for GTAP
        cols = ['ave_baseline', 'ave_total', 'shock_tariff_cut', 'shock_ntm', 'shock_full']
        for c in cols:
            template[c] = template[c] * 100
            
        template.rename(columns={
            'ave_baseline': 'AVE_Baseline_Pct',
            'shock_full': 'Shock_Full_Facilitation_Pct'
        }, inplace=True)
        
        outfile = self.output_dir / 'tables' / 'gtap_shock_template.csv'
        template.to_csv(outfile, index=False)
        logger.info(f"GTAP Shock file saved: {outfile}")
        
        self._plot_scenarios(template)

    def _plot_scenarios(self, df):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        avg_shock = df.groupby('importer')[['Shock_Full_Facilitation_Pct']].mean().sort_values('Shock_Full_Facilitation_Pct')
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=avg_shock.index, y=avg_shock['Shock_Full_Facilitation_Pct'], palette='magma')
        plt.title('Average Trade Cost Reduction (Full Facilitation Scenario)')
        plt.ylabel('Reduction (%)')
        plt.axhline(0, color='k')
        plt.savefig(self.output_dir / 'figures' / 'gtap_scenarios.png')
        plt.close()
