# serve/app.py
"""
FastAPI backend for DeepFashion retrieval.
Endpoints:
  GET  /              -> health
  POST /search        -> upload image and return top-K hits (requires model or fallback)
  GET  /thumb/{idx}   -> return image bytes by gallery index
"""

from pathlib import Path
import io
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import faiss
import torch
from torchvision import transforms
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
df = pd.read_csv(r"C:\Users\dilku\deepfashion-recsys\data\deepfashion_index.csv")
print(df.iloc[0]["image_path"])

# Project paths (adjust if needed)
ROOT = Path(__file__).resolve().parents[1]   
EXPORT_DIR = ROOT / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# try several common index names
INDEX_CANDIDATES = [
    EXPORT_DIR / "gallery_index.faiss",
    EXPORT_DIR / "gallery_index.index",
    EXPORT_DIR / "faiss_gallery_finetuned.index",
    EXPORT_DIR / "gallery_index.faiss.index",
]

# metadata file (CSV or parquet)
GALLERY_META_CSV = ROOT / "data" / "deepfashion_index.csv"
GALLERY_META_PARQUET = ROOT / "data" / "gallery_meta.parquet"
GALLERY_EMB = EXPORT_DIR / "gallery_embs_finetuned.npy"
CHECKPOINT_DIR = ROOT / "checkpoints"

# FastAPI app
app = FastAPI(title="DeepFashion Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# response models
class Hit(BaseModel):
    image_path: str
    item_id: str
    score: float
    index: int

class SearchResponse(BaseModel):
    results: List[Hit]

# image transforms
IMG_TFMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- Load gallery metadata ---
import pandas as pd
if GALLERY_META_CSV.exists():
    gallery_df = pd.read_csv(GALLERY_META_CSV)
elif GALLERY_META_PARQUET.exists():
    gallery_df = pd.read_parquet(GALLERY_META_PARQUET)
else:
    raise FileNotFoundError(f"Gallery metadata not found. Place CSV at {GALLERY_META_CSV} or parquet at {GALLERY_META_PARQUET}")

# If there's a 'split' column, filter for gallery split
if "split" in gallery_df.columns:
    gallery_df = gallery_df[gallery_df["split"].str.contains("gallery", case=False, na=False)].reset_index(drop=True)
else:
    gallery_df = gallery_df.reset_index(drop=True)

# Normalize image paths to absolute strings
def _abs_path(x):
    p = Path(x)
    if not p.is_absolute():
        p = (ROOT / str(x)).resolve()
    return str(p)
if "image_path" not in gallery_df.columns:
    raise ValueError("gallery metadata must contain 'image_path' column")
gallery_df["image_path"] = gallery_df["image_path"].apply(_abs_path)

# --- Load FAISS index ---
INDEX_PATH: Optional[Path] = None
for cand in INDEX_CANDIDATES:
    if cand.exists():
        INDEX_PATH = cand
        break
if INDEX_PATH is None:
    raise FileNotFoundError(f"No FAISS index found. Looked for: {INDEX_CANDIDATES}")
index = faiss.read_index(str(INDEX_PATH))
print("Loaded FAISS index:", INDEX_PATH)

# optionally load gallery embeddings if present
gallery_embs = None
if GALLERY_EMB.exists():
    try:
        gallery_embs = np.load(GALLERY_EMB)
        print("Loaded gallery embeddings:", GALLERY_EMB)
    except Exception as e:
        print("Warning loading gallery_embs:", e)
        gallery_embs = None

# --- Model loading (optional) ---
MODEL = None
FALLBACK_BACKBONE = None
# try to load a checkpointed model (user may add a TripletModel here)
try:
    import sys
    sys.path.append(str(ROOT / "src"))
    # user-provided model import — comment out if not available
    try:
        from training.train_triplet import TripletModel  # optional
        ckpts = sorted(list(CHECKPOINT_DIR.glob("*.ckpt")), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            ckpt = ckpts[0]
            MODEL = TripletModel.load_from_checkpoint(str(ckpt))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            MODEL = MODEL.to(device)
            MODEL.eval()
            print("Loaded TripletModel checkpoint:", ckpt.name)
        else:
            print("No checkpoint found in", CHECKPOINT_DIR)
    except Exception:
        # no user TripletModel available; continue to fallback below
        MODEL = None
except Exception as e:
    print("Model import warning:", e)
    MODEL = None

# fallback backbone (ResNet50) for quick testing if MODEL is None
if MODEL is None:
    try:
        import torchvision.models as tvmodels
        fb = tvmodels.resnet50(weights=tvmodels.ResNet50_Weights.DEFAULT)
        fb.fc = torch.nn.Identity()
        fb.eval()
        FALLBACK_BACKBONE = fb
        print("Using torchvision ResNet50 fallback backbone for embeddings.")
    except Exception as e:
        FALLBACK_BACKBONE = None
        print("No fallback backbone available:", e)

def image_to_embedding(pil_img: Image.Image):
    """
    Returns numpy array shape (1, D) float32 embedding.
    Uses MODEL if present else FALLBACK_BACKBONE if available.
    """
    x = IMG_TFMS(pil_img).unsqueeze(0)
    if MODEL is not None:
        device = next(MODEL.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            feat = MODEL(x).cpu().numpy().astype("float32")
        return feat
    elif FALLBACK_BACKBONE is not None:
        # run on CPU
        with torch.no_grad():
            feat = FALLBACK_BACKBONE(x).cpu().numpy().astype("float32")
        return feat
    else:
        raise RuntimeError("No model available for embedding extraction. Add a checkpoint or enable fallback backbone.")

# --- Endpoints ---
@app.get("/")
def root():
    return {"status": "ok", "note": "POST an image to /search (form-data field name: file)."}

@app.get("/thumb/{idx}")
def get_thumb(idx: int):
    if idx < 0 or idx >= len(gallery_df):
        return JSONResponse(status_code=404, content={"error": "index out of range"})
    img_path = Path(gallery_df.iloc[idx]["image_path"])
    if not img_path.exists():
        return JSONResponse(status_code=404, content={"error": "file not found", "path": str(img_path)})
    return StreamingResponse(img_path.open("rb"), media_type="image/jpeg")

@app.post("/search", response_model=SearchResponse)
async def search(file: UploadFile = File(...), topk: int = 5):
    """
    Upload an image and return top-K similar gallery items.
    """
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image uploaded.")

    # compute query embedding
    try:
        q_emb = image_to_embedding(img)   # shape (1, D)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding error: {e}")

    # normalize and search (assumes IndexFlatIP with normalized vectors for cosine)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype("float32"), int(topk))

    hits = []
    for score, idx in zip(D[0], I[0]):
        idx = int(idx)
        if idx < 0 or idx >= len(gallery_df):
            continue
        row = gallery_df.iloc[idx]
        img_p = str(Path(row["image_path"]).resolve())
        item_id = row.get("item_id") if "item_id" in row else str(idx)
        hits.append(Hit(image_path=img_p, item_id=item_id, score=float(score), index=idx))

    return SearchResponse(results=hits)
