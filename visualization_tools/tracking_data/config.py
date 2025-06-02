"""
Configuration handler with YAML support.
"""

import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path


class Config:
    """Configuration handler that loads from YAML files."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration from YAML file."""
        self._config = {}
        
        if config_path:
            self._load_yaml(config_path)
    
    def _load_yaml(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f) or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        