"""
DreamLite FastAPI Server
Wraps the DreamLite diffusion pipeline and serves a /generate endpoint
for the frontend UI at public/dreamlite/index.html.

Usage:
    pip install fastapi uvicorn pillow
    python dreamlite_server.py [--model models/DreamLite-base] [--device cuda] [--port 8787]
"""

import argparse
import base64
import io
import sys
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# Add DreamLite source to path
SCRIPT_DIR = Path(__file__).parent
DREAMLITE_DIR = SCRIPT_DIR / "DreamLite"
sys.path.insert(0, str(DREAMLITE_DIR))

warnings.filterwarnings("ignore")

# ── CLI args ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DreamLite inference server")
parser.add_argument("--model", default="DreamLite/models/DreamLite-base", help="Path to DreamLite model")
parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
parser.add_argument("--port", type=int, default=8787)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--mobile", action="store_true", help="Use DreamLite-Mobile pipeline (4-step fast)")
args = parser.parse_args()

# ── Pipeline state ────────────────────────────────────────────────────
pipeline = None
model_name = ""

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, model_name
    dtype = DTYPE_MAP[args.dtype]
    device = args.device

    print(f"[DreamLite] Loading pipeline from: {args.model}")
    print(f"[DreamLite] Device: {device}  dtype: {args.dtype}")

    try:
        if args.mobile:
            from dreamlite import DreamLiteMobilePipeline
            pipeline = DreamLiteMobilePipeline.from_pretrained(
                args.model, torch_dtype=dtype
            ).to(device)
            model_name = "DreamLite-Mobile"
        else:
            from dreamlite import DreamLitePipeline
            pipeline = DreamLitePipeline.from_pretrained(
                args.model, torch_dtype=dtype
            ).to(device)
            model_name = "DreamLite-Base"
        print(f"[DreamLite] {model_name} loaded successfully.")
    except Exception as e:
        print(f"[DreamLite] WARNING: Could not load pipeline: {e}")
        print("[DreamLite] Server will start but /generate will return errors until model is available.")

    yield  # server runs here

    pipeline = None


# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(title="DreamLite Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response schemas ────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_steps: int = 28
    guidance_scale: float = 3.5
    image_guidance_scale: float = 1.0
    width: int = 1024
    height: int = 1024
    mode: str = "t2i"          # "t2i" or "edit"
    image: str | None = None   # base64-encoded PNG/JPEG (edit mode)
    seed: int = 42


class GenerateResponse(BaseModel):
    image: str       # base64-encoded PNG
    elapsed: float   # seconds


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": model_name if pipeline else None,
        "ready": pipeline is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if pipeline is None:
        raise HTTPException(503, detail="Model not loaded. Check server logs.")

    # Decode optional input image
    input_image = None
    if req.mode == "edit" and req.image:
        try:
            img_bytes = base64.b64decode(req.image)
            input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, detail=f"Invalid input image: {e}")

    if req.mode == "edit" and input_image is None:
        raise HTTPException(400, detail="Image editing mode requires an input image.")

    # Determine size from input image when editing
    w, h = req.width, req.height
    if input_image is not None:
        w, h = input_image.size

    generator = torch.Generator("cpu").manual_seed(req.seed)

    t0 = time.perf_counter()
    try:
        result = pipeline(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            image=input_image,
            height=h,
            width=w,
            guidance_scale=req.guidance_scale,
            image_guidance_scale=req.image_guidance_scale,
            num_inference_steps=req.num_steps,
            generator=generator,
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Inference failed: {e}")

    elapsed = time.perf_counter() - t0
    out_image = result.images[0]

    # Resize to requested output size if needed
    target = (req.width, req.height)
    if out_image.size != target:
        out_image = out_image.resize(target, Image.Resampling.LANCZOS)

    # Encode as base64 PNG
    buf = io.BytesIO()
    out_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return GenerateResponse(image=b64, elapsed=round(elapsed, 2))


# ── Serve frontend static files ───────────────────────────────────────
frontend_path = SCRIPT_DIR / "public" / "dreamlite"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[DreamLite] Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
