import yaml
import os
import re
from typing import Any, Dict

class Config:
    _env_pattern = re.compile(r'\$\{([^}^{]+)\}')
    def __init__(self, config_path: str, default_config: Dict[str, Any] = None):
        self.config_path = config_path
        self.config = default_config or {}
        self.load_config()
    def load_config(self):
        with open(self.config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        self.config = self._merge_dicts(self.config, yaml_config or {})
        self.config = self._substitute_env_vars(self.config)
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    def as_dict(self) -> Dict[str, Any]:
        return self.config
    def _merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._merge_dicts(base[key], value)
            else:
                base[key] = value
        return base
    def _substitute_env_vars(self, config: Any) -> Any:
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._env_pattern.sub(lambda match: os.environ.get(match.group(1), match.group(0)), config)
        else:
            return config