import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

class Visualizer:
    def __init__(self, output_dir: Path):
        self.fig_dir = output_dir / 'figures'
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
    
    def plot_trade_overview(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        data = df.groupby('importer')['trade_value'].sum().sort_values(ascending=False)
        sns.barplot(x=data.index, y=data.values, palette='viridis')
        plt.title('Total Trade by Country')
        plt.savefig(self.fig_dir / 'trade_by_country.png')
        plt.close()

    def plot_trade_over_time(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        year_trade = df.groupby('year')['trade_value'].sum()
        plt.plot(year_trade.index, year_trade.values, marker='o', linewidth=2, color='steelblue')
        plt.xlabel('Year')
        plt.ylabel('Total Trade Value (USD)')
        plt.title('Thailand-GCC Trade Over Time')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.fig_dir / 'trade_over_time.png')
        plt.close()

    def plot_distance_vs_trade(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        # Unique importers for coloring
        importers = df['importer'].unique()
        colors = plt.cm.Set2(range(len(importers)))
        
        for i, country in enumerate(importers):
            country_data = df[df['importer'] == country]
            plt.scatter(country_data['distance'], country_data['trade_value'],
                    label=country, alpha=0.6, s=100)
        
        plt.xlabel('Distance (km)')
        plt.ylabel('Trade Value (USD)')
        plt.title('Distance vs Trade Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.fig_dir / 'distance_vs_trade.png')
        plt.close()
