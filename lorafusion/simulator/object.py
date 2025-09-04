"""Base class for objects that can be configured via YAML files."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T", bound="ConfigurableObject")


@dataclass
class ConfigurableObject:
    """Base class for objects that can be configured via YAML files."""

    @classmethod
    def from_dict(cls: type[T], config_dict: dict[str, Any]) -> T:
        """Create an instance from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            An instance of the class initialized with the provided configuration.
        """
        # Filter out keys that aren't valid parameters for the class
        valid_keys = {field_.name for field_ in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    @classmethod
    def from_yaml(cls: type[T], yaml_path: str | Path) -> T:
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            An instance initialized with the configuration from the YAML file.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML file is invalid.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            msg = f"Configuration file not found: {yaml_path}"
            raise FileNotFoundError(msg)

        with yaml_path.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the object to a dictionary.

        Returns:
            A dictionary representation of the object.
        """
        return asdict(self)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save the object configuration to a YAML file.

        Args:
            yaml_path: Path where to save the YAML configuration.

        Raises:
            IOError: If the file cannot be written.
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)
