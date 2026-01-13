import sys
import logging
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Dict
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Check for optional libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available. API calls will be limited.")

try:
    import wbdata
    WBDATA_AVAILABLE = True
except ImportError:
    WBDATA_AVAILABLE = False
    logger.warning("wbdata library not available.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost not available.")

class CacheManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, filename: str, subdir: str = None) -> Path:
        if subdir:
            path = self.cache_dir / subdir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return self.cache_dir / filename

    def is_valid(self, filepath: Path, max_age_days: int) -> bool:
        if not filepath.exists():
            return False
        age_days = (datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)).days
        return age_days <= max_age_days

    def save(self, data: Any, filename: str, subdir: str = None) -> Path:
        path = self.get_path(filename, subdir)
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        elif isinstance(data, dict) or isinstance(data, list):
            with open(path, 'w') as f:
                json.dump(data, f)
        return path

    def load_df(self, filename: str, subdir: str = None, max_age_days: int = 30) -> Optional[pd.DataFrame]:
        path = self.get_path(filename, subdir)
        if self.is_valid(path, max_age_days):
            logger.info(f"Using cached file: {filename}")
            return pd.read_csv(path)
        return None

    def load_json(self, filename: str, max_age_days: int = 180) -> Optional[Dict]:
        path = self.get_path(filename)
        if self.is_valid(path, max_age_days):
            logger.info(f"Using cached JSON: {filename}")
            with open(path, 'r') as f:
                return json.load(f)
        return None

def check_and_install_dependencies():
    """Checks for required packages and installs them if missing."""
    required_packages = [
        'requests', 'requests-cache', 'pandas', 'numpy', 
        'matplotlib', 'seaborn', 'scipy', 'statsmodels',
        'scikit-learn', 'xgboost', 'shap', 'openpyxl', 
        'xlrd', 'wbdata', 'pandas-datareader'
    ]
    
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg.replace('-', '_')) 
        except ImportError:
            missing.append(pkg)
            
    if not missing:
        logger.info("All dependencies are satisfied.")
        return

    logger.info(f"Installing missing packages: {', '.join(missing)}")
    
    def install(package):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", package, "--break-system-packages"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError:
            return False

    for pkg in missing:
        if install(pkg):
            logger.info(f"Installed {pkg}")
        else:
            logger.error(f"Failed to install {pkg}")
