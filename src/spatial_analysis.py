import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import radians, sin, cos, asin, sqrt

try:
    from esda.moran import Moran
    from libpysal.weights import W
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False

from utils import logger

class SpatialAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.coords = {
            "BHR": (26.2285, 50.5860),  # Manama
            "KWT": (29.3759, 47.9774),  # Kuwait City
            "OMN": (23.5880, 58.3829),  # Muscat
            "QAT": (25.2854, 51.5310),  # Doha
            "SAU": (24.7136, 46.6753),  # Riyadh
            "ARE": (24.4539, 54.3773)   # Abu Dhabi
        }

    def haversine_distance(self, coord1, coord2):
        lat1, lon1 = radians(coord1[0]), radians(coord1[1])
        lat2, lon2 = radians(coord2[0]), radians(coord2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return c * 6371  # Earth radius in km

    def run_moran_test(self, df: pd.DataFrame):
        if not SPATIAL_AVAILABLE:
            logger.warning("Spatial libraries (esda, libpysal) not found. Skipping Moran's I.")
            return

        logger.info("[8/11] Running Spatial Analysis (Moran's I)...")
        
        # Aggregate by importer
        spatial_df = df.groupby('importer').agg({
            'trade_value': 'mean'
        }).reset_index()
        
        # Add coordinates
        spatial_df['lat'] = spatial_df['importer'].map(lambda x: self.coords.get(x, (0,0))[0])
        spatial_df['lon'] = spatial_df['importer'].map(lambda x: self.coords.get(x, (0,0))[1])
        spatial_df = spatial_df[spatial_df['lat'] != 0]

        if len(spatial_df) < 2:
            logger.warning("Not enough countries for spatial analysis.")
            return

        # Prepare weights
        coords_arr = spatial_df[['lat', 'lon']].values
        n = len(coords_arr)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i, j] = self.haversine_distance(coords_arr[i], coords_arr[j])
        
        # Inverse distance weights
        weights_matrix = np.zeros_like(dist_matrix)
        with np.errstate(divide='ignore'):
            weights_matrix = 1.0 / dist_matrix
        np.fill_diagonal(weights_matrix, 0)
        
        # Row standardize
        row_sums = weights_matrix.sum(axis=1)
        weights_matrix = weights_matrix / row_sums[:, np.newaxis]
        
        # Create W object
        neighbors = {i: [j for j in range(n) if i != j] for i in range(n)}
        w_dict = {i: {j: weights_matrix[i, j] for j in neighbors[i]} for i in range(n)}
        w_spatial = W(w_dict)
        
        # Moran's I
        y = spatial_df['trade_value'].values
        moran = Moran(y, w_spatial, permutations=999)
        
        logger.info(f"Moran's I: {moran.I:.4f} (p-value: {moran.p_sim:.4f})")
        
        # Save Results
        res = pd.DataFrame([{
            'Statistic': "Moran's I",
            'Value': moran.I,
            'P-Value': moran.p_sim,
            'Z-Score': moran.z_sim,
            'Interpretation': 'Significant Clustering' if moran.p_sim < 0.05 else 'Random Spatial Pattern'
        }])
        res.to_csv(self.output_dir / 'tables' / 'spatial_results.csv', index=False)
        
        # Visualization
        self._plot_spatial(spatial_df, y, w_spatial, moran)

    def _plot_spatial(self, df, y, w, moran):
        lag = w.sparse.dot(y)
        plt.figure(figsize=(8, 6))
        plt.scatter(y, lag, s=100, alpha=0.7)
        plt.axhline(y=lag.mean(), color='r', linestyle='--')
        plt.axvline(x=y.mean(), color='b', linestyle='--')
        plt.xlabel('Trade Value')
        plt.ylabel('Spatial Lag (Neighbor Avg)')
        plt.title(f"Moran's I Scatterplot (I={moran.I:.2f})")
        for i, txt in enumerate(df['importer']):
            plt.annotate(txt, (y[i], lag[i]))
        plt.savefig(self.output_dir / 'figures' / 'spatial_analysis.png')
        plt.close()
