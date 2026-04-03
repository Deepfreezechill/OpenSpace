"""Domain layer — ports (Protocol interfaces) and core value types.

This package defines the contract boundary between the application core
and its infrastructure adapters.  Nothing in ``scion.domain`` should
import from adapter packages (``cloud``, ``grounding.backends``, ``llm``,
``recording``, ``local_server``, etc.).
"""
