import pandas as pd
import numpy as np
from typing import Dict, Tuple

from config import ThesisConfig
from utils import logger

class DataProcessor:
    def __init__(self, config: ThesisConfig):
        self.cfg = config

    def integrate(self, 
                  trade_df: pd.DataFrame, 
                  wdi_df: pd.DataFrame, 
                  dist_df: pd.DataFrame, 
                  pci_data: Dict) -> pd.DataFrame:
        
        logger.info("[5/11] Integrating Data...")
        
        # Merge Distances
        df = trade_df.merge(
            dist_df.rename(columns={'PartnerISO3': 'importer'}),
            on='importer', how='left'
        )
        
        # Add PCI
        df['pci'] = df['hs6'].map(pci_data)
        
        # Merge WDI Exporter
        wdi_ex = wdi_df[wdi_df['iso3'] == self.cfg.exporter].rename(columns={'iso3': 'exporter'})
        df = df.merge(
            wdi_ex[['exporter', 'year', 'gdp', 'population', 'internet']],
            on=['exporter', 'year'], how='left'
        ).rename(columns={'gdp': 'gdp_ex', 'population': 'pop_ex', 'internet': 'internet_ex'})
        
        # Merge WDI Importer
        wdi_im = wdi_df[wdi_df['iso3'].isin(self.cfg.importers)].rename(columns={'iso3': 'importer'})
        df = df.merge(
            wdi_im[['importer', 'year', 'gdp', 'population', 'internet']],
            on=['importer', 'year'], how='left'
        ).rename(columns={'gdp': 'gdp_im', 'population': 'pop_im', 'internet': 'internet_im'})
        
        # Feature Engineering
        np.random.seed(42)
        df['tariff'] = np.random.uniform(0.01, 0.15, len(df)) # Simulated
        
        df['legal_ex'] = df['exporter'].map(self.cfg.legal_systems)
        df['legal_im'] = df['importer'].map(self.cfg.legal_systems)
        df['same_legal'] = (df['legal_ex'] == df['legal_im']).astype(int)
        
        # Log Transforms
        for col in ['trade_value', 'distance', 'gdp_ex', 'gdp_im', 'pop_ex', 'pop_im']:
            if col == 'distance':
                df[f'log_{col}'] = np.log(df[col])
            else:
                df[f'ln_{col}'] = np.log1p(df[col]) if 'trade' in col else np.log(df[col])
        
        # Cleanup
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)
        
        logger.info(f"Integration Complete. Shape: {df.shape}")
        return df

    def split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = df[df['year'] < self.cfg.split_year].copy()
        test = df[df['year'] >= self.cfg.split_year].copy()
        return train, test
