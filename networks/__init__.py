# __init__.py
# All comments are in English as requested.

# Re-export only the public, high-level API.
from .BuildNet import buildnet, model_dict

# Optional: define the explicit public surface to prevent star-import leaks.
__all__ = ["buildnet", "model_dict"]
