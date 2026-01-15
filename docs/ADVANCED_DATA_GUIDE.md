# Advanced Data Integration Guide

This guide outlines how to upgrade the current project from "Simulated/Rule-Based" logic to "Fully Empirical" using official WTO datasets.

## 1. Real Tariff Data (WTO TAO)
Currently, `src/data_processor.py` uses a **Rule-Based** approach (`get_real_tariff`) which approximates real policies. To use actual CSV data:

**Prerequisite:** Download `tariff_download_2025.csv` from [WTO TAO](https://tao.wto.org).

**Code Implementation (in `src/data_processor.py`):**
```python
def load_real_tariffs(self, csv_path):
    tariff_df = pd.read_csv(csv_path)
    # Clean and standardize columns
    tariff_df['Product Code'] = tariff_df['Product Code'].astype(str).str.strip()
    return tariff_df

def get_tariff_from_csv(self, row, tariff_df):
    product = str(row['hs6'])
    country = str(row['importer'])
    
    match = tariff_df[
        (tariff_df['Product Code'] == product) & 
        (tariff_df['Reporting Economy'] == country)
    ]
    
    if not match.empty:
        return match['Applied Rate'].values[0]
    return 5.0 # Fallback
```

---

## 2. HS Code Harmonization (HS Tracker)
To solve data inconsistencies between years (e.g., HS2012 vs HS2017).

**Prerequisite:** Create a mapping dictionary based on [WTO HS Tracker](https://hstracker.wto.org).

**Code Implementation (in `src/config.py`):**
```python
# Add this mapping structure
HS_CONVERSION_MAP = {
    '123499': '123456', # New Code -> Old Code
    '854231': '854230'
}
```

**Code Implementation (in `src/data_processor.py`):**
```python
def harmonize_hs_code(self, hs_code):
    hs_code = str(hs_code)
    # Return mapped code or original
    return self.cfg.HS_CONVERSION_MAP.get(hs_code, hs_code)

# Usage inside integrate():
# df['hs6'] = df['hs6'].apply(self.harmonize_hs_code)
```

---

## 3. Non-Tariff Measures (I-TIP)
To incorporate barriers like SPS (Sanitary) or TBT (Technical Barriers).

**Prerequisite:** Download NTM data csv from [WTO I-TIP](https://i-tip.wto.org/goods).

**Code Implementation (in `src/data_processor.py`):**
```python
def check_ntm_barrier(self, row, ntm_df):
    product = str(row['hs6'])
    country = str(row['importer'])
    
    # Check for SPS measures (Sanitary and Phytosanitary)
    measures = ntm_df[
        (ntm_df['Product'] == product) & 
        (ntm_df['Member'] == country) &
        (ntm_df['Measure_Type'] == 'SPS')
    ]
    return 1 if not measures.empty else 0

# Usage:
# df['has_sps'] = df.apply(lambda x: self.check_ntm_barrier(x, ntm_df), axis=1)
```

---

### Integration Strategy
When these CSV files are available, modify the `DataProcessor.integrate` method to accept them as arguments and apply the logic above before the Model Training phase.
