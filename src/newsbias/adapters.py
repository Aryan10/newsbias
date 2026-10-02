"""Instantiate configured plugins while keeping shared execution code extensible."""

from importlib import import_module
from pathlib import Path
from typing import TypedDict


class AdapterConfig(TypedDict):
    """Adapter plugin selection used by both experiment and benchmark runners.

    `adapter` names a Python class; `parameters` contains only that class's
    constructor options, so the shared runner does not need a model/dataset registry.
    """

    adapter: str
    parameters: dict[str, object]


def load_adapter(config: AdapterConfig) -> object:
    """Import and construct the dataset or model implementation named in config.

    Adapter paths use `python.module:ClassName`; the `parameters` table is passed
    unchanged as keyword arguments. Import and construction errors are allowed to
    identify configuration or optional-dependency problems at startup. Treat the
    config as trusted executable project configuration, not untrusted user input.
    """
    module_name, separator, class_name = config["adapter"].partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("Adapter must use 'python.module:ClassName' syntax")
    try:
        module = import_module(module_name)
    except ImportError as error:
        raise ImportError(f"Could not import adapter module {module_name!r}") from error
    try:
        adapter_class = getattr(module, class_name)
    except AttributeError as error:
        raise ValueError(f"Adapter {class_name!r} not found in {module_name!r}") from error
    if not callable(adapter_class):
        raise TypeError(f"Configured adapter {config['adapter']!r} is not callable")
    parameters = dict(config["parameters"])
    return adapter_class(**parameters)


def load_environment_file(path: Path) -> None:
    """Load a local dotenv file into the process without overriding existing variables.

    This lets provider adapters read credentials from their normal environment
    variables while keeping secret values out of TOML, run summaries, and stdout.
    The optional `python-dotenv` package is required only when this helper is used.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Environment file not found: {path}")
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError("Install python-dotenv to load an environment file") from error
    load_dotenv(path, override=False)
