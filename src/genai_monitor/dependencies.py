import importlib.util
import re
import typing
import warnings
from enum import Enum
from typing import Dict, List, Tuple

from loguru import logger

from genai_monitor.static.extras import EXTRAS_REQUIRE


def _check_package_available(package_name: str) -> bool:
    """Check if a package is available without importing it.

    Args:
        package_name: Name of the package to check

    Returns:
        True if the package is available, False otherwise
    """
    return importlib.util.find_spec(package_name) is not None


class VersionComparison(Enum):
    """Enum for version comparison types."""

    GREATER_THAN = ">"
    GREATER_THAN_EQUAL = ">="
    EQUAL = "=="


def _check_package_version(package_name: str, version_spec: str) -> bool:
    """Check if an installed package meets the version requirement.

    Args:
        package_name: Name of the package to check
        version_spec: Version specification (e.g., ">=1.0.0")

    Returns:
        bool: True if the package meets the version requirement, False otherwise
    """
    if not _check_package_available(package_name):
        return False

    try:
        package = __import__(package_name)

        version = getattr(package, "__version__", getattr(package, "VERSION", None))

        if version is None:
            try:
                from importlib.metadata import version as get_version

                version = get_version(package_name)
            except (ImportError, ModuleNotFoundError):
                try:
                    import pkg_resources  # type: ignore

                    version = pkg_resources.get_distribution(package_name).version
                except (ImportError, pkg_resources.DistributionNotFound):
                    return False

        for version_comparsion in VersionComparison:
            if version_comparsion.value in version_spec:
                req_version = version_spec.split(version_comparsion.value)[1].strip()
                return _compare_versions(version, req_version, version_comparsion.value)

        return True

    except (ImportError, AttributeError) as e:
        logger.debug(f"Dependency check failed: {str(e)}")
        return False


def _parse_version(version_str):
    """Parse a version string into parts, handling special formats like '0+cu124'."""
    clean_version = version_str.split("+")[0]
    clean_version = re.sub(r"(\.0+)*$", "", clean_version)
    try:
        return [int(x) for x in clean_version.split(".")]
    except ValueError:
        return [int(part) for part in re.findall(r"\d+", clean_version)]


def _compare_versions(version1: str, version2: str, version_comparison: str) -> bool:
    """Compare two version strings.

    Args:
        version1: First version string
        version2: Second version string
        version_comparison: Comparison type

    Returns:
        True if the first version satisfies the comparison with the second version.
    """
    v1_parts = _parse_version(version1)
    v2_parts = _parse_version(version2)

    for i in range(max(len(v1_parts), len(v2_parts))):
        v1_part = v1_parts[i] if i < len(v1_parts) else 0
        v2_part = v2_parts[i] if i < len(v2_parts) else 0

        if v1_part < v2_part:
            version_diff = -1
            break
        if v1_part > v2_part:
            version_diff = 1
            break
    else:
        version_diff = 0

    if version_comparison == VersionComparison.GREATER_THAN:
        return version_diff > 0
    if version_comparison == VersionComparison.GREATER_THAN_EQUAL:
        return version_diff >= 0
    if version_comparison == VersionComparison.EQUAL:
        return version_diff == 0

    return False


def is_extra_available(extra_name: str, required_extras: Dict[str, List[Tuple[str, str]]]) -> bool:
    """Check if all packages for an extra are available with correct versions.

    Args:
        extra_name: Name of the extra to check
        required_extras: Dictionary of required extras

    Returns:
        True if all packages in the extra are available with correct versions

    Raises:
        ValueError: If the extra name is not recognized
    """
    if extra_name not in required_extras:
        raise ValueError(f"Unknown extra: {extra_name}")

    for package_name, version_spec in required_extras[extra_name]:
        if not _check_package_version(package_name, version_spec):
            return False

    return True


def require_extra(extra_name: str, required_extras: Dict[str, List[Tuple[str, str]]], raise_error: bool = True) -> bool:
    """Check if an extra is available and either raise an error or return a boolean.

    Args:
        extra_name: Name of the extra to check
        required_extras: Dictionary of required extras
        raise_error: If True, raise RuntimeError when extra is not available
                     If False, return a boolean indicating availability

    Returns:
        True if the extra is available (only when raise_error is False)

    Raises:
        ValueError: If the extra name is not recognized
        RuntimeError: If the extra is not available and raise_error is True
    """
    if extra_name not in required_extras:
        raise ValueError(f"Unknown extra: {extra_name}")

    is_available = is_extra_available(extra_name, required_extras)

    if not is_available and raise_error:
        package_list = ", ".join(f"{pkg[0]}{pkg[1]}" for pkg in required_extras[extra_name])
        raise RuntimeError(
            f"The '{extra_name}' functionality requires additional dependencies: {package_list}. "
            f"Install with: pip install genai_eval[{extra_name}]"
        )

    return is_available


def warn_if_extra_unavailable(extra_name: str, required_extras: Dict[str, List[Tuple[str, str]]]) -> None:
    """Emit a warning if an extra is not available.

    Args:
        extra_name: Name of the extra to check
        required_extras: Dictionary of required extras
    """
    if not is_extra_available(extra_name, required_extras):
        package_list = ", ".join(f"{pkg[0]}{pkg[1]}" for pkg in required_extras[extra_name])
        warnings.warn( # noqa: B028
            f"The '{extra_name}' functionality requires additional dependencies: {package_list}. "
            f"Some features may not work. Install with: pip install genai_eval[{extra_name}]"
        )

@typing.no_type_check
def get_missing_packages(extra_name: str, required_extras: Dict[str, Tuple[str, str]]) -> List[str]:
    """Get a list of missing packages for an extra.

    Args:
        extra_name: Name of the extra to check
        required_extras: Dictionary of required extras

    Returns:
        List of missing package names with their version requirements

    Raises:
        ValueError: If the extra name is not recognized
    """
    if extra_name not in required_extras:
        raise ValueError(f"Unknown extra: {extra_name}")

    missing = []
    for package_name, version_spec in required_extras[extra_name]:
        if not _check_package_version(package_name, version_spec):
            missing.append(f"{package_name}{version_spec}")

    return missing


DIFFUSERS_AVAILABLE = is_extra_available("diffusers", EXTRAS_REQUIRE)
TRANSFORMERS_AVAILABLE = is_extra_available("transformers", EXTRAS_REQUIRE)
OPENAI_AVAILABLE = is_extra_available("openai", EXTRAS_REQUIRE)
LITELLM_AVAILABLE = is_extra_available("litellm", EXTRAS_REQUIRE)
ALL_AVAILABLE = all((DIFFUSERS_AVAILABLE, TRANSFORMERS_AVAILABLE, OPENAI_AVAILABLE, LITELLM_AVAILABLE))

if __name__ == "__main__":
    print(f"Diffusers available: {DIFFUSERS_AVAILABLE}")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    print(f"OpenAI available: {OPENAI_AVAILABLE}")
    print(f"LiteLLM available: {LITELLM_AVAILABLE}")
    print(f"All available: {ALL_AVAILABLE}")
