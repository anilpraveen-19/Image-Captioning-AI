import os
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.services.caption_service import CaptionService


app = FastAPI(
    title="AI Image Captioning API",
    description="CNN + LSTM Image Captioning System",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


caption_service = CaptionService()


@app.get("/")
def root():
    return {
        "message": "AI Image Captioning API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/caption")
async def generate_caption(
    file: UploadFile = File(...)
):

    suffix = os.path.splitext(
        file.filename
    )[1] or ".jpg"

    contents = await file.read()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        temp.write(contents)
        temp_path = temp.name

    try:

        caption = caption_service.generate_caption(
            temp_path
        )

        return {
            "success": True,
            "caption": caption
        }

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)