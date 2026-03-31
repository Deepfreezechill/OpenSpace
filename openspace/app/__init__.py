"""Application layer — composition root and service wiring.

The :class:`AppContainer` holds references to all domain services,
wired through Protocol interfaces from :mod:`openspace.domain.ports`.
No module in the codebase should construct services directly — they
should receive them from the container.
"""
