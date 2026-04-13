# src/__init__.py
"""
Package initialization.
Expose only what is needed for lazy import.
"""

def get_app():
    """Import paresseux de l'application Flask (non utilisé dans la version Streamlit)."""
    from .app import app  # noqa: F401
    return app
