"""Vercel Python entrypoint — serves the FastAPI app as an ASGI function.

vercel.json rewrites every path here, so FastAPI handles its own routing
(/, /health, /search, /agent).
"""

from src.api.routes import app  # noqa: F401  (Vercel serves this `app`)
