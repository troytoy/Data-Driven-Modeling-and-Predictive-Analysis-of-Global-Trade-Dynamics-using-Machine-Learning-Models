import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

@dataclass
class AnalysisConfig:
    # API Keys - SECURED: Reading from Environment Variables
    # keys removed from source code
    comtrade_key: str = field(default_factory=lambda: os.getenv("COMTRADE_KEY", ""))
    wto_key: str = field(default_factory=lambda: os.getenv("WTO_KEY", ""))

    # Study Parameters
    exporter: str = "THA"
    importers: List[str] = field(default_factory=lambda: ["BHR", "KWT", "OMN", "QAT", "SAU", "ARE"])
    years: List[int] = field(default_factory=lambda: list(range(2018, 2023)))
    hs6_products: List[str] = field(default_factory=lambda: [
        "020714", "160100", "160232", "040210", 
        "190531", "200599", "210390", "210690"
    ])
    split_year: int = 2021

    # Mappings
    country_codes: Dict[str, str] = field(default_factory=lambda: {
        'THA': '764', 'BHR': '048', 'KWT': '414', 'OMN': '512',
        'QAT': '634', 'SAU': '682', 'ARE': '784'
    })
    
    iso3_to_iso2: Dict[str, str] = field(default_factory=lambda: {
        'THA': 'TH', 'BHR': 'BH', 'KWT': 'KW', 'OMN': 'OM',
        'QAT': 'QA', 'SAU': 'SA', 'ARE': 'AE'
    })

    # Paths
    base_dir: Path = field(default_factory=lambda: Path("./trade_analysis_results"))
    
    @property
    def output_dir(self) -> Path:
        return self.base_dir / "output"
    
    @property
    def cache_dir(self) -> Path:
        # Changed from Thesis_DATA_CACHE
        d = self.base_dir / "data_cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # Verified Fallbacks (Static Data)
    pci_fallback: Dict[str, float] = field(default_factory=lambda: {
        '020714': 0.823, '160100': -0.231, '160232': 0.287,
        '040210': -0.534, '190531': -0.298, '200599': 0.145,
        '210390': 0.421, '210690': 0.198
    })

    distance_fallback: Dict[str, int] = field(default_factory=lambda: {
        'SAU': 6106, 'ARE': 4904, 'QAT': 5288,
        'KWT': 5666, 'OMN': 4666, 'BHR': 5382
    })

    legal_systems: Dict[str, str] = field(default_factory=lambda: {
        'THA': 'uk', 'BHR': 'uk', 'KWT': 'fr', 'OMN': 'uk',
        'QAT': 'fr', 'SAU': 'is', 'ARE': 'uk'
    })
