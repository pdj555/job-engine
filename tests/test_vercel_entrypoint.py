"""The Vercel entrypoint (api/index.py) is the deployed API surface.

vercel.json rewrites every path to /api/index, so the object it exposes is
what production serves. The rest of the suite imports src.api.routes
directly; these tests pin the deployment shim itself, which previously had
no test coverage at all.
"""

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.routes import app as routes_app

_ENTRYPOINT = Path(__file__).resolve().parents[1] / "api" / "index.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("vercel_api_index", _ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entrypoint_reexports_the_routes_app():
    module = _load_entrypoint()
    assert module.app is routes_app
    assert module.app.title == "Job Engine"
    assert module.app.version == "0.2.0"


def test_entrypoint_serves_health_through_the_shim():
    client = TestClient(_load_entrypoint().app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["search_ready"] is True
    assert set(body["apis"]) == {"openai", "brave", "perplexity"}


def test_entrypoint_root_describes_usage():
    client = TestClient(_load_entrypoint().app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "/search" in response.json()["usage"]
