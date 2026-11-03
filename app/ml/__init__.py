"""Per-agent machine-learning models.

Each model is optional and lazily loaded. Agents fall back to their curated
rule paths when a model is disabled or unavailable, and always report which
path was used in the response envelope.
"""
