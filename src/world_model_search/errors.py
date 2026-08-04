"""User-facing errors with stable command-line exit behavior."""


class WorldModelSearchError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(WorldModelSearchError):
    """Raised when configuration is invalid."""


class PersistenceError(WorldModelSearchError):
    """Raised when recorded run data is missing, inconsistent, or unsafe."""


class ReplayError(WorldModelSearchError):
    """Raised when deterministic replay diverges from recorded data."""


class PhaseUnavailableError(WorldModelSearchError):
    """Raised for a CLI surface reserved for a later implementation phase."""
