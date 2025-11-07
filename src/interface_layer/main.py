from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from main_service_router import story_sense_router

app = FastAPI(
    title="Story Sense Analyser",
    description="API for generating metrics report",
    version="1.0"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=['*'],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(story_sense_router, prefix="/api")
