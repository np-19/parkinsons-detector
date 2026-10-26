"""FastAPI Application Entrypoint for Parkinson's Acoustic Screening."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import CORS_ORIGINS
from backend.routers import health, predict
from backend.services.predictor_service import PredictorService

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context: preloads model weights on boot."""
    logger.info("Initializing Parkinson's Acoustic Screening Backend...")
    try:
        service = PredictorService.get_instance()
        service.initialize()
        logger.info("Model ready for inference on device: %s", service.get_device_name())
    except Exception as e:
        logger.exception("Failed to initialize model during startup: %s", e)
        raise e

    yield

    logger.info("Shutting down Parkinson's Screening Backend...")


app = FastAPI(
    title="NeuroVocal API | Parkinson's Acoustic Screening Engine",
    description=(
        "Production-grade clinical decision support API for speech-based Parkinson's Disease detection. "
        "Powered by fine-tuned WavLM Base+ (mPower benchmark) with count-invariant Gated Attention-MIL "
        "participant aggregation."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS) for React / Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route handlers
app.include_router(health.router)
app.include_router(predict.router)


@app.get("/", tags=["Root"])
def root():
    return JSONResponse(
        content={
            "app": "NeuroVocal Acoustic Screening API",
            "version": "2.0.0",
            "status": "operational",
            "documentation": "/docs",
            "health_check": "/api/health",
            "prediction_endpoint": "/api/v1/predict",
        }
    )


if __name__ == "__main__":
    import uvicorn
    from backend.config import HOST, PORT

    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False)
