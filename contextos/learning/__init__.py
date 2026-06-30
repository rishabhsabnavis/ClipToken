"""Across-call learning loop: FidelityStore, LossDetector, CompressionPolicy.

These components are not on the hot path of every call -- they run alongside the
request-path modules and persist state across sessions. Together they are what let
ContextOS get *less* lossy over time.
"""
