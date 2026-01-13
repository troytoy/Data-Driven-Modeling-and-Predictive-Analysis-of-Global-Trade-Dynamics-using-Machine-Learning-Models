#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis Version 6.7 ULTIMATE (Refactored) - Thailand-GCC Trade Analysis
Refactored for Clean Code, Modularity, and Maintainability.

Original Author: Research Team
Refactored by: AI Assistant
Date: January 13, 2026
"""

import sys
from pathlib import Path

# Local Module Imports
try:
    from config import ThesisConfig
    from utils import CacheManager, logger, check_and_install_dependencies
    from data_collector import DataCollector
    from data_processor import DataProcessor
    from models import ModelEngine
    from visualization import Visualizer
except ImportError as e:
    # If running from outside src without -m, this might fail if not handled.
    # But standard python execution 'python src/script.py' adds src to path.
    print(f"Import Error: {e}")
    print("Ensure you are running the script from the correct directory or have dependencies installed.")
    sys.exit(1)

def main():
    print("="*60)
    print(" THESIS v6.7 ULTIMATE (Refactored) ")
    print("="*60)

    # 0. Check Dependencies
    check_and_install_dependencies()

    # 1. Setup
    config = ThesisConfig()
    
    # Platform specific path adjustment
    if 'colab' in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        config.base_dir = Path("/content/drive/MyDrive")
    else:
        # User Scatch Directory or Current Directory
        config.base_dir = Path("./thesis_v6_7_ultimate")
    
    # Create Directories
    logger.info(f"Setting up output directory: {config.output_dir}")
    for d in ['tables', 'figures', 'latex']:
        (config.output_dir / d).mkdir(parents=True, exist_ok=True)
    
    cache_mgr = CacheManager(config.cache_dir)
    
    # 2. Collect Data
    collector = DataCollector(config, cache_mgr)
    trade_df = collector.fetch_comtrade()
    wdi_df = collector.fetch_wdi()
    dist_df = collector.load_distances()
    pci_data = collector.fetch_pci()
    
    if trade_df.empty or wdi_df.empty:
        logger.error("Critical data missing. Aborting.")
        return

    # 3. Process Data
    processor = DataProcessor(config)
    full_df = processor.integrate(trade_df, wdi_df, dist_df, pci_data)
    full_df.to_csv(config.output_dir / 'integrated_dataset.csv', index=False)
    
    train_df, test_df = processor.split_train_test(full_df)
    
    # 4. Analysis
    engine = ModelEngine(config.output_dir)
    engine.run_ppml(train_df, test_df)
    engine.run_ml_models(train_df, test_df)
    engine.save_comparison()
    
    # 5. Visuals
    viz = Visualizer(config.output_dir)
    viz.plot_trade_overview(full_df)
    
    print("\nWorkflow Complete! Check:", config.output_dir)

if __name__ == "__main__":
    main()
