import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestMainApp:
    """Test suite for main.py FastAPI application"""

    @pytest.fixture
    def client(self, mock_main_service_router):
        """Create a test client for the FastAPI app"""
        # Import after mocking
        from src.interface_layer.main import app
        return TestClient(app)

    def test_app_initialization(self, mock_main_service_router):
        """Test that the FastAPI app is initialized correctly"""
        from src.interface_layer.main import app

        assert app is not None
        assert isinstance(app, FastAPI)
        assert app.title == "Story Sense Analyser"
        assert app.description == "API for generating metrics report"
        assert app.version == "1.0"

    def test_app_has_cors_middleware(self, mock_main_service_router):
        """Test that CORS middleware is configured"""
        from src.interface_layer.main import app
        from starlette.middleware.cors import CORSMiddleware

        # Check if CORS middleware is in the middleware stack
        middleware_found = False
        for middleware in app.user_middleware:
            if middleware.cls == CORSMiddleware:
                middleware_found = True
                break

        assert middleware_found, "CORS middleware not found in app"

    def test_cors_configuration(self, client):
        """Test CORS configuration allows all origins"""
        response = client.options(
            "/api/test",
            headers={
                "Origin": "http://testserver",
                "Access-Control-Request-Method": "POST",
            }
        )

        assert response.headers.get("access-control-allow-origin") == "*"

    def test_router_inclusion(self, mock_main_service_router):
        """Test that the story_sense_router is included"""
        from src.interface_layer.main import app

        # Check that routes exist
        route_paths = [route.path for route in app.routes if hasattr(route, 'path')]
        assert len(route_paths) > 0

    def test_api_prefix(self, client):
        """Test that API routes are prefixed with /api"""
        response = client.post("/process-stories")
        assert response.status_code == 404

        # Routes with /api prefix should exist (even if they return errors)
        response = client.post("/api/process-stories")
        assert response.status_code != 404

    def test_openapi_schema_generation(self, client):
        """Test that OpenAPI schema is generated correctly"""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert schema["info"]["title"] == "Story Sense Analyser"
        assert schema["info"]["version"] == "1.0"

    def test_docs_endpoint(self, client):
        """Test that API documentation endpoint is accessible"""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint(self, client):
        """Test that ReDoc documentation endpoint is accessible"""
        response = client.get("/redoc")
        assert response.status_code == 200

    @pytest.mark.parametrize("origin", [
        "http://localhost:3000",
        "http://example.com",
        "https://example.com"
    ])
    def test_cors_multiple_origins(self, client, origin):
        """Test CORS with multiple different origins"""
        response = client.options(
            "/api/test",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            }
        )

        assert response.headers.get("access-control-allow-origin") == "*"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])