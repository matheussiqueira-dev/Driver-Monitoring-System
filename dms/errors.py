"""Domain errors with stable exit codes, handled once at the CLI boundary."""

from __future__ import annotations


class DMSError(Exception):
    """Base class for expected, user-facing failures."""

    exit_code = 1


class ConfigError(DMSError):
    exit_code = 2


class SourceUnavailableError(DMSError):
    exit_code = 3


class ModelDownloadError(DMSError):
    exit_code = 4
