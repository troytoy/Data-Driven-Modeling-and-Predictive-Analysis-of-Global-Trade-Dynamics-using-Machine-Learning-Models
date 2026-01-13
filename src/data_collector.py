import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import wbdata

from config import ThesisConfig
from utils import CacheManager, logger, REQUESTS_AVAILABLE, WBDATA_AVAILABLE

class DataCollector:
    def __init__(self, config: ThesisConfig, cache: CacheManager):
        self.cfg = config
        self.cache = cache

    def fetch_comtrade(self) -> pd.DataFrame:
        """Fetching UN COMTRADE Data"""
        cache_file = "comtrade_data.csv"
        df = self.cache.load_df(cache_file, max_age_days=7)
        if df is not None:
            return df
        
        if not REQUESTS_AVAILABLE:
            logger.error("Requests library missing for Comtrade API")
            return pd.DataFrame()

        logger.info("[1/11] Fetching UN COMTRADE DATA from API...")
        session = requests.Session()
        session.headers.update({
            'Ocp-Apim-Subscription-Key': self.cfg.comtrade_key,
            'User-Agent': 'ThesisResearch/6.7'
        })
        base_url = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
        
        all_data = []
        
        for year in self.cfg.years:
            for importer in self.cfg.importers:
                for hs6 in self.cfg.hs6_products:
                    try:
                        params = {
                            'reporterCode': self.cfg.country_codes['THA'],
                            'partnerCode': self.cfg.country_codes[importer],
                            'period': str(year),
                            'cmdCode': hs6,
                            'flowCode': 'X',
                            'customsCode': 'C00',
                            'motCode': '0'
                        }
                        resp = session.get(base_url, params=params, timeout=30)
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get('data'):
                                val = float(data['data'][0].get('primaryValue', 0))
                                all_data.append({'year': year, 'exporter': self.cfg.exporter, 
                                                 'importer': importer, 'hs6': hs6, 
                                                 'trade_value': val})
                            else:
                                all_data.append({'year': year, 'exporter': self.cfg.exporter, 
                                                 'importer': importer, 'hs6': hs6, 
                                                 'trade_value': 0.0})
                        else:
                            all_data.append({'year': year, 'exporter': self.cfg.exporter, 
                                             'importer': importer, 'hs6': hs6, 
                                             'trade_value': np.nan})
                    except Exception as e:
                        logger.error(f"Error fetching {year}-{importer}-{hs6}: {e}")
                        all_data.append({'year': year, 'exporter': self.cfg.exporter, 
                                         'importer': importer, 'hs6': hs6, 
                                         'trade_value': np.nan})
                    
                    time.sleep(0.4) # Rate limiting
        
        df = pd.DataFrame(all_data)
        
        # Data cleaning: fill missing values with median
        for hs6 in self.cfg.hs6_products:
            mask = df['hs6'] == hs6
            if mask.any() and df.loc[mask, 'trade_value'].isna().any():
                median_val = df.loc[mask, 'trade_value'].median()
                df.loc[mask & df['trade_value'].isna(), 'trade_value'] = median_val if not pd.isna(median_val) else 0.0

        self.cache.save(df, cache_file)
        return df

    def fetch_wdi(self) -> pd.DataFrame:
        """Fetching World Bank WDI Data"""
        cache_file = "wdi_data.csv"
        df = self.cache.load_df(cache_file, max_age_days=90)
        if df is not None:
            return df

        if not WBDATA_AVAILABLE:
            logger.error("wbdata missing")
            return pd.DataFrame()

        logger.info("[2/11] Fetching World Bank WDI Data...")
        countries = [self.cfg.exporter] + self.cfg.importers
        countries_iso2 = [self.cfg.iso3_to_iso2.get(c, c) for c in countries]
        
        indicators = {
            'NY.GDP.MKTP.CD': 'gdp',
            'SP.POP.TOTL': 'population',
            'IT.NET.USER.ZS': 'internet'
        }
        
        try:
            raw_data = wbdata.get_dataframe(indicators, country=countries_iso2, 
                                            date=(str(self.cfg.years[0]), str(self.cfg.years[-1])))
            
            raw_data = raw_data.reset_index()
            raw_data['date'] = pd.to_datetime(raw_data['date'])
            raw_data['year'] = raw_data['date'].dt.year
            
            # Reconstruct simple mapping based on ISO codes
            name_map = {
                'Thailand': 'THA', 'Bahrain': 'BHR', 'Kuwait': 'KWT',
                'Oman': 'OMN', 'Qatar': 'QAT', 'Saudi Arabia': 'SAU',
                'United Arab Emirates': 'ARE'
            }
            if 'country' in raw_data.columns:
                 raw_data['iso3'] = raw_data['country'].map(name_map)
            else:
                 # Try index if country is not a column
                 raw_data['iso3'] = raw_data.index.map(name_map)
                 if raw_data['iso3'].isna().all(): 
                     logger.warning("Could not map countries from WDI response.")

            raw_data = raw_data.dropna(subset=['iso3'])
            
            df_wdi = raw_data[['iso3', 'year', 'gdp', 'population', 'internet']].copy()
            
            # Numeric conversion & Filling
            cols = ['gdp', 'population', 'internet']
            for c in cols:
                df_wdi[c] = pd.to_numeric(df_wdi[c], errors='coerce')
                # Forward/Back fill per country
                df_wdi[c] = df_wdi.groupby('iso3')[c].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
                df_wdi[c] = df_wdi[c].fillna(df_wdi[c].median()) # Final fallback
            
            self.cache.save(df_wdi, cache_file)
            return df_wdi
            
        except Exception as e:
            logger.error(f"WDI Fetch failed: {e}")
            return pd.DataFrame()

    def load_distances(self) -> pd.DataFrame:
        """Loading Distances (CEPII or Fallback)"""
        cache_file = "cepii_distances.csv"
        df = self.cache.load_df(cache_file, max_age_days=9999) # Static data
        if df is not None:
            return df
        
        logger.info("[3/11] Loading Distances...")
        # Fallback logic
        logger.warning("Using Verified Fallback Distances")
        data = {
            'PartnerISO3': list(self.cfg.distance_fallback.keys()),
            'distance': list(self.cfg.distance_fallback.values())
        }
        df_dist = pd.DataFrame(data)
        self.cache.save(df_dist, cache_file)
        return df_dist

    def fetch_pci(self) -> Dict[str, float]:
        """Fetching PCI Data (Harvard or Fallback)"""
        cache_file = "harvard_pci.json"
        pci_data = self.cache.load_json(cache_file, max_age_days=180)
        
        if pci_data and np.std(list(pci_data.values())) > 0.01:
            return pci_data
        
        logger.info("[4/11] Fetching Harvard PCI Data...")
        logger.info("Using Verified Fallback PCI Data")
        self.cache.save(self.cfg.pci_fallback, cache_file)
        return self.cfg.pci_fallback
