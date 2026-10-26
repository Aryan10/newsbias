"""Article import endpoints; blocking fetch/agent work runs in the thread pool."""

from fastapi import APIRouter, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from newsbias.ingestion.schemas import (
    MAX_FILE_BYTES,
    ExtractedArticle,
    ExtractionError,
    RawImportRequest,
    URLImportRequest,
)
from newsbias.ingestion.service import import_file, import_raw, import_url

router = APIRouter(prefix="/bias/import", tags=["Article import"])


class ImportSizeLimitMiddleware:
    """Stop oversized import bodies before the multipart parser spools them to disk."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/bias/import/"):
            return await self.app(scope, receive, send)
        limit = MAX_FILE_BYTES + 64_000  # Allow multipart field/header overhead.
        headers = dict(scope.get("headers", []))
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            content_length = 0
        if content_length > limit:
            response = JSONResponse({"detail": "The import request exceeds the size limit."}, 413)
            return await response(scope, receive, send)
        total = 0

        async def limited_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > limit:
                    raise HTTPException(413, "The import request exceeds the size limit.")
            return message

        return await self.app(scope, limited_receive, send)


@router.post("/raw", response_model=ExtractedArticle)
def extract_raw(req: RawImportRequest):
    try:
        return import_raw(req.text)
    except ExtractionError as error:
        raise HTTPException(error.status_code, str(error)) from error


@router.post("/url", response_model=ExtractedArticle)
def extract_url(req: URLImportRequest):
    try:
        return import_url(req.url)
    except ExtractionError as error:
        raise HTTPException(error.status_code, str(error)) from error


@router.post("/file", response_model=ExtractedArticle)
async def extract_file(file: UploadFile):
    try:
        data = await file.read(MAX_FILE_BYTES + 1)
        return await run_in_threadpool(import_file, data, file.filename or "")
    except ExtractionError as error:
        raise HTTPException(error.status_code, str(error)) from error
    finally:
        await file.close()
