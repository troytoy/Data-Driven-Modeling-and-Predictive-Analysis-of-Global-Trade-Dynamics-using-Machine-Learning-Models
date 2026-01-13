from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

@dataclass
class ThesisConfig:
    # API Keys
    comtrade_key: str = "5d86bad9fa714c979dc95c0ea2ec9ba8"
    wto_key: str = "68946a81b94841b19279c7800d52b213"

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

    # Output Paths
    base_dir: Path = field(default_factory=Path.cwd)
    
    @property
    def output_dir(self) -> Path:
        return self.base_dir / "Thesis_v6_7_ULTIMATE"

    @property
    def cache_dir(self) -> Path:
        return self.base_dir / "Thesis_DATA_CACHE"

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
