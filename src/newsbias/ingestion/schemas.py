"""Contracts shared by all article import paths."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_INPUT_CHARS = 120_000
MAX_FILE_BYTES = 1_000_000


class ExtractionError(Exception):
    """An import failure with a safe, user-facing message."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


class RawImportRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


class URLImportRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class ArticleSource(BaseModel):
    type: Literal["raw", "url", "upload"]
    url: str | None = None
    filename: str | None = None
    author: str | None = None
    publication_date: str | None = None


class ExtractionDetails(BaseModel):
    method: str
    notices: list[str] = Field(default_factory=list)


class ExtractedArticle(BaseModel):
    title: str
    text: str
    source: ArticleSource
    extraction: ExtractionDetails


class BlockSelection(BaseModel):
    """The agent selects original text, rather than generating replacement prose."""

    model_config = ConfigDict(extra="forbid", strict=True)

    is_article: bool
    title_block_id: int | None
    body_block_ids: list[int]
    notices: list[str] = Field(default_factory=list)
