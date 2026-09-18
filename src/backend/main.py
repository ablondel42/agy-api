"""FastAPI application factory and ASGI entry point for agy-api backend."""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from logging_config import setup_logging
from routes.agents import router as agents_router
from routes.chat import router as chat_router
from routes.models import router as models_router
from session_manager import SessionManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for the FastAPI application."""
    setup_logging()
    session_manager = SessionManager()
    await session_manager.start()
    app.state.session_manager = session_manager
    yield
    await session_manager.stop()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app with all routers, middleware, and lifespan.
    """
    app = FastAPI(
        title="agy-api",
        description="OpenAI-compatible API server powered by AGY CLI",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers under /v1
    app.include_router(chat_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(agents_router, prefix="/v1")

    # Root endpoint for health check and version
    @app.get("/")
    async def root():
        return {"message": "agy-api", "version": "0.1.0"}

    return app


# Default ASGI app instance for uvicorn (e.g. `uvicorn main:app`)
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        reload_dirs=["routes", "."],
        reload_excludes=[".venv", ".git", ".brain", "log", "logs", "tests"],
    )
