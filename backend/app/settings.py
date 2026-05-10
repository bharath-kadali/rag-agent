from __future__ import annotations

from pydantic import BaseModel
import os


class Settings(BaseModel):
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    app_data_dir: str = "./data"
    cors_origins: list[str] = ["https://*.vercel.app"]

    @staticmethod
    def load() -> "Settings":
        cors = os.getenv("CORS_ORIGINS", "http://localhost:5173").strip()
        cors_origins = [o.strip() for o in cors.split(",") if o.strip()]
        return Settings(
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
            app_data_dir=os.getenv("APP_DATA_DIR", "./data"),
            cors_origins=cors_origins or ["http://localhost:5173"],
        )

