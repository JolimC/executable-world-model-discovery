"""User-facing errors with stable command-line exit behavior."""


class WorldModelSearchError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(WorldModelSearchError):
    """Raised when configuration is invalid."""


class PersistenceError(WorldModelSearchError):
    """Raised when recorded run data is missing, inconsistent, or unsafe."""


class BudgetExhaustedError(PersistenceError):
    """Raised when a prospective action cannot fit a frozen hard ceiling."""


class ReplayError(WorldModelSearchError):
    """Raised when deterministic replay diverges from recorded data."""


class PhaseUnavailableError(WorldModelSearchError):
    """Raised for a CLI surface reserved for a later implementation phase."""


class CandidateValidationError(WorldModelSearchError):
    """Raised when untrusted candidate data fails closed."""


class OracleVerificationError(WorldModelSearchError):
    """Raised when authorized task data cannot be verified safely."""
