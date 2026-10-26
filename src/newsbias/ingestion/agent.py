"""Structured, source-preserving news extraction using the configured OpenAI client."""

from pydantic import ValidationError

from newsbias.ingestion.blocks import TextBlock
from newsbias.ingestion.schemas import BlockSelection, ExtractionError
from newsbias.prompts import JinjaPrompt


def select_article_blocks(
    blocks: list[TextBlock], *, title_hint: str, client, prompt: JinjaPrompt,
) -> BlockSelection:
    try:
        rendered = prompt.render(
            title_hint=title_hint,
            blocks=[{"id": block.id, "text": block.text} for block in blocks],
        )
        result = rendered.complete_and_parse_json(client=client)
        return BlockSelection.model_validate(result)
    except ValidationError as error:
        raise ExtractionError(
            "The extraction agent returned an invalid response. Try again.", 502,
        ) from error
    except ExtractionError:
        raise
    except Exception as error:
        raise ExtractionError(
            "Article extraction could not reach the configured model. Check OPENAI_* settings.",
            502,
        ) from error
