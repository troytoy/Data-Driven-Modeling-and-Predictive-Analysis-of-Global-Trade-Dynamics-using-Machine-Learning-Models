import pandas as pd
import numpy as np
from typing import Dict, Tuple

from config import AnalysisConfig
from utils import logger

class DataProcessor:
    def __init__(self, config: AnalysisConfig):
        self.cfg = config

    def integrate(self, 
                  trade_df: pd.DataFrame, 
                  wdi_df: pd.DataFrame, 
                  dist_df: pd.DataFrame, 
                  pci_data: Dict) -> pd.DataFrame:
        
        logger.info("[5/11] Integrating Data...")
        
        # 1. Align Time Periods (Intersection of available years)
        trade_years = set(trade_df['year'])
        wdi_years = set(wdi_df['year'])
        common_years = trade_years.intersection(wdi_years)
        
        if not common_years:
            logger.error("No overlapping years found between Trade and WDI data!")
            return pd.DataFrame()
            
        start_y, end_y = min(common_years), max(common_years)
        logger.info(f"Aligning data to common period: {start_y}-{end_y}")
        
        trade_df = trade_df[trade_df['year'].isin(common_years)]
        wdi_df = wdi_df[wdi_df['year'].isin(common_years)]
        
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
        
        # --- Real Tariff Logic (Rule-Based) ---
        # Map HS Codes to Product Names for logic
        hs_map = {
            "020714": "Frozen Chicken Cuts",
            "160100": "Sausages",
            "160232": "Prepared Chicken",
            "040210": "Milk Powder",
            "190531": "Cookies/Biscuits",
            "200599": "Vegetables",
            "210390": "Sauces",
            "210690": "Food Preparations"
        }
        
        def get_real_tariff(row):
            product_code = row['hs6']
            product_name = hs_map.get(product_code, "").lower()
            country = str(row['importer']).lower()
            
            # 1. Sin Tax (Pork/Alcohol) - Simulated check (assuming 160100 might be pork in some contexts, 
            # but for Thailand export to GCC it's mostly Halal Chicken. Let's keep 100% logic available)
            if 'pork' in product_name or 'alcohol' in product_name:
                return 1.00 # 100%
                
            # 2. Saudi Arabia Exceptions
            if 'sau' in country or 'saudi' in country:
                if 'chicken' in product_name or '020714' in product_code or '160232' in product_code:
                    return 0.20 # 20% for Poultry protection
                if 'milk' in product_name or 'dairy' in product_name:
                    return 0.10 # 10%
                if 'date' in product_name:
                    return 0.40 # 40%
            
            # 3. Basic Food Security (Rice, etc.)
            if any(x in product_name for x in ['rice', 'wheat', 'corn']):
                return 0.00
                
            # 4. Standard GCC Tariff
            return 0.05 # 5%

        # Apply the logic
        logger.info("Applying Rule-Based Tariff Logic...")
        df['tariff'] = df.apply(get_real_tariff, axis=1)
        
        # Feature Engineering
        np.random.seed(42) # Keep seed for other random ops if any
        # df['tariff'] = np.random.uniform(0.01, 0.15, len(df)) # REMOVED: Old Simulation
        
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
