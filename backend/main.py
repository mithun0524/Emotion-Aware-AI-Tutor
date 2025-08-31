from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from routes.tutor import router as tutor_router

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the tutor route
app.include_router(tutor_router, prefix="/api", tags=["Tutor"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Emotion-Aware Tutor Backend!"}
