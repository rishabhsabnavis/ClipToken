"""ContextOS -- drop-in middleware that compresses LLM agent context before each API call.

The package is split into two paths:
- ``contextos.modules``  -- the per-call request path (Modules 1-5)
- ``contextos.learning`` -- the across-call learning loop (FidelityStore, LossDetector, CompressionPolicy)

See CLAUDE.md for the full architecture and module contracts.
"""

__version__ = "0.1.0"
