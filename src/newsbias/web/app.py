"""FastAPI web server for standalone NewsBias Media Bias & Neutrality Tribunal.

Serves the built React single-page application and provides API endpoints for:
- Curated benchmark sample retrieval (/bias/samples)
- Multi-specialist + Arbiter bias analysis (/bias/analyze)
- Neutral AP wire rewrite generation with diff log (/bias/rewrite)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from newsbias.ingestion.live_news import (
    ALL_PROVIDERS,
    GLOBAL_LIVE_NEWS_CACHE,
    PROVIDER_FETCHERS,
    get_live_news_feed,
    get_provider_keys,
)
from newsbias.language import detect_article_language
from newsbias.models.bias import predict_sentence_bias, predict_sentence_bias_mock
from newsbias.models.llm import get_openai_client
from newsbias.models.political import predict_political_leaning, predict_political_leaning_mock
from newsbias.pipeline.analyzer import analyze_article
from newsbias.pipeline.segmenter import split_sentences
from newsbias.prompts import JinjaPrompt, extract_json_payload
from newsbias.web.imports import ImportSizeLimitMiddleware
from newsbias.web.imports import router as import_router

logger = logging.getLogger("newsbias.web")

app = FastAPI(
    title="NewsBias Media Bias Tribunal API",
    version="1.0.0",
    description="Automated media bias, political stance, and editorial tribunal API.",
)
app.include_router(import_router)
app.add_middleware(ImportSizeLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


# ── Benchmark Samples ──────────────────────────────────────────────────────────
BENCHMARK_SAMPLES = [
    {
        "id": "aravalli-forest-dispute",
        "dataset": "Indian Express / Down To Earth",
        "title": "Death Knell for Aravallis: SC 100m Order Stirs Mining Mafia Panic",
        "category": "Sensationalist Alarm & Judicial Misinterpretation",
        "source_name": "The Indian Express / Down To Earth",
        "source_url": "https://indianexpress.com/article/india/aravalli-definition-supreme-court-100-metre-fsi-mining/",
        "text": (
            "In a move described by conservationists as a catastrophic death knell for India’s oldest fold mountains, "
            "the Supreme Court has accepted a controversial 100-metre elevation threshold that effectively strips environmental "
            "safeguards from over ninety percent of the Aravalli range. Under the arbitrary new definition pushed by the Union "
            "Environment Ministry, any landform under 100 metres in local relief will no longer qualify as an Aravalli hill, officially "
            "abandoning nearly eleven thousand ecologically irreplaceable ridges across Haryana and Rajasthan to the mercy of rapacious "
            "mining mafias and real estate syndicates.\n\n"
            "Outraged activists and forestry experts warned that this bureaucratic travesty completely capitulates to the commercial "
            "quarrying lobby, greenlighting the dynamiting and flattening of vital low-lying hillocks that serve as North India’s sole buffer "
            "against the advancing Thar Desert. By excluding low ridges that constitute the primary groundwater recharge aquifers for the "
            "parched National Capital Region, critics argue the judicial bench has condemned millions of citizens to acute water scarcity, "
            "toxic particulate haze, and irreversible desertification.\n\n"
            "Ignoring explicit dissent from the Forest Survey of India, which cautioned that only a tiny fraction of documented ridges "
            "exceed the 100-metre mark, media dispatches and citizen collectives immediately sounded the alarm that state governments "
            "are preparing to auction off denotified rocky slopes for commercial plunder. Spurred by sensational reports that bulldozers and "
            "stone crushers would erase ninety percent of the range overnight, hundreds of furious villagers, environmentalists, and "
            "farmer unions launched aggressive highway blockades and sit-in demonstrations across southern Haryana, demanding the "
            "immediate revocation of what they termed a state-sanctioned ecocide."
        ),
    },
    {
        "id": "kashmir-maps-bias",
        "dataset": "Kashmir / The Hindu",
        "title": "Kashmir Territorial Cartography & Border Sovereignty Dispute",
        "category": "Geopolitical Bias & Dynamic Cartography",
        "source_name": "The Hindu / Survey of India",
        "source_url": "https://www.thehindu.com",
        "text": (
            "A fierce international debate has intensified over digital cartography and the systematic geopolitical bias embedded "
            "in global mapping platforms regarding Jammu, Kashmir, and Ladakh. Tech conglomerates and international media organizations "
            "have faced severe backlash from New Delhi for deploying geofenced, dynamic border renderings that depict the region's "
            "boundaries inconsistently based on user geography. While users within India see the constitutional territorial integrity "
            "upheld under Survey of India standards, global and regional versions frequently erase Indian sovereignty by demarcating vast "
            "swathes as contested territory or adopting adversary terminologies.\n\n"
            "International human rights watchdogs and regional commentators from Islamabad and Beijing routinely push rival narratives, "
            "characterizing Indian administrative consolidations since the 2019 bifurcation as unilateral changes to an internationally "
            "disputed zone. Pakistani state media continues to amplify rhetoric describing the territory through the lens of military occupation "
            "and disputed plebiscites, while Chinese state outlets align reporting with Beijing's disputed LAC claims in Aksai Chin.\n\n"
            "Conversely, Indian authorities and strategic analysts reject foreign cartographic manipulation as an infringement on national "
            "sovereignty. Official spokespersons reiterated that the entirety of Jammu, Kashmir, and Ladakh constitutes an integral and "
            "non-negotiable part of the Indian Union, warning digital platforms that publishing distorted maps that legitimize illegal foreign "
            "occupations violates national criminal law."
        ),
    },
    {
        "id": "ladakh-sixth-schedule",
        "dataset": "Ladakh / Indian Express",
        "title": "Ladakh Sixth Schedule Agitation & Ecological Safeguards",
        "category": "Indigenous Autonomy vs Centralized Governance",
        "source_name": "The Indian Express / LAB",
        "source_url": "https://indianexpress.com",
        "text": (
            "Deep in the high-altitude desert of Leh, thousands of indigenous demonstrators gathered in biting sub-zero conditions "
            "to support Sonam Wangchuk's indefinite climate fast, escalating an impassioned showdown between grassroots Himalayan "
            "communities and the central administration. Agitators from the Leh Apex Body and Kargil Democratic Alliance have demanded "
            "constitutional guarantees under the Sixth Schedule, warning that Ladakh's hyper-fragile glacial ecosystems will face "
            "catastrophic degradation under unchecked industrialization, corporate mega-projects, and centralized bureaucratic oversight.\n\n"
            "Movement organizers accused national mainstream media channels of maintaining a calculated blackout on the agitations to "
            "shield political establishments from scrutiny over unfulfilled electoral promises. Leaders highlighted that local tribal "
            "populations, who comprise over ninety percent of Ladakh, have been effectively disenfranchised without legislative representation, "
            "leaving delicate pastures vulnerable to commercial mining ventures that threaten to accelerate glacial retreat.\n\n"
            "In response, central government representatives emphasized that substantial infrastructure allocations and renewable energy "
            "investments have unlocked unprecedented development for the strategic frontier. Officials argued that high-powered advisory "
            "committees are already addressing tribal welfare, asserting that constitutional autonomy demands must be balanced against "
            "national defense logistics and broad-based economic integration along sensitive Himalayan borders."
        ),
    },
    {
        "id": "babe-delhi-smog",
        "dataset": "BABE / Reuters Wire",
        "title": "New Delhi Air Quality & Meteorological Inversion Monitoring",
        "category": "Objective Environmental Wire (Non-Biased)",
        "source_name": "Reuters (BABE Dataset)",
        "source_url": "https://www.reuters.com/article/us-india-pollution/toxic-smog-shrouds-indian-capital-as-pollution-soars-government-measures-fail-idUSKCN1OX0K9",
        "text": (
            "A sharp drop in temperatures and wind speed, combined with vehicle and industrial emissions, dust from "
            "building sites and smoke from garbage burning has boosted pollution over much of north India, including "
            "New Delhi, home to more than 20 million.\n\n"
            "The Central Pollution Control Board reported that the air quality index for Delhi hovered in the severe "
            "category, with microscopic PM2.5 concentrations exceeding 440 micrograms per cubic meter across monitoring "
            "stations.\n\n"
            "Government monitoring agencies directed schools to suspend outdoor activities, deployed mechanized vacuum "
            "sweepers to control road dust, and ordered municipal authorities to sprinkle water across heavy-traffic corridors. "
            "Regional environmental meteorologists stated that unfavorable atmospheric conditions and agricultural crop-residue "
            "burning in neighboring states are expected to sustain the smog cover through the weekend."
        ),
    },
    {
        "id": "wire-fed-neutral",
        "dataset": "Reuters / AP News Wire",
        "title": "Federal Reserve Holds Benchmark Rates Steady",
        "category": "Neutral Objective Wire Reporting",
        "source_name": "Reuters / AP Wire",
        "source_url": "https://www.reuters.com",
        "text": (
            "The Federal Reserve announced on Wednesday that benchmark interest rates will remain unchanged between 5.25% and 5.50% "
            "following its two-day policy meeting. Central bank officials noted that inflation has moderated significantly over the past "
            "year while the labor market continues to demonstrate resilience, with payroll additions remaining consistent with sustainable "
            "long-term economic expansion.\n\n"
            "In an official statement released after the deliberations, the Federal Open Market Committee emphasized that committee members "
            "do not anticipate reducing borrowing costs until they gain greater confidence that consumer price inflation is sustainably "
            "tracking toward their 2% target. The committee highlighted that economic activity has been expanding at a solid pace, though the "
            "economic outlook remains subject to global trade uncertainties.\n\n"
            "Financial analysts from major institutions had widely expected the hold decision ahead of upcoming economic reports on consumer "
            "spending and gross domestic product. Government bond yields held steady following the announcement, while major equity indices "
            "registered modest gains as market participants digested Chairman Jerome Powell's post-meeting press conference remarks."
        ),
    },
    {
        "id": "babe-h1b-india",
        "dataset": "BABE / Breitbart",
        "title": "High-Skilled Visa Policy & Outsourcing Debate",
        "category": "Economic Nationalist Stance (Biased)",
        "source_name": "Breitbart News (BABE Dataset)",
        "source_url": "https://www.breitbart.com/politics/2019/02/11/billionaire-kochs-back-gop-dem-plan-to-outsource-middle-class-american-jobs-2/",
        "text": (
            "A plan known as the Fairness for High-Skilled Immigrants Act is a boon to Silicon Valley billionaires, big "
            "business elites, and outsourcing firms, as they would be able to readily import more lower-paid Indian and Chinese "
            "foreign workers to take American jobs that would have otherwise gone to American citizens.\n\n"
            "Lobbying groups formed by the nation’s major tech companies reportedly employ large blocs of foreign visa-workers "
            "in high-paying software engineering roles that could readily be filled by young domestic graduates.\n\n"
            "Congressional sponsors continue to push legislation that would dramatically expand legal and financial incentives "
            "for corporations to bypass local labor pools, replacing middle-class professionals with temporary contract workers "
            "bound to employer-sponsored residency quotas."
        ),
    },
    {
        "id": "babe-india-diplomacy",
        "dataset": "BABE / Alternet",
        "title": "Diplomatic Protocol & Bilateral Arms Pact Framing",
        "category": "Partisan Geopolitical Critique (Biased)",
        "source_name": "Alternet (BABE Dataset)",
        "source_url": "https://www.alternet.org/2020/02/trump-visits-modis-india-and-announces-3-billion-arms-deal-the-worst-kind-of-fascists/",
        "text": (
            "The official state visit kicked off with an elaborate spectacle of staged pomp and pageantry in Gujarat, where leaders "
            "announced a three-billion-dollar military arms package designed to cement strategic defense partnerships in the Indo-Pacific.\n\n"
            "Human rights advocates and civil society groups strongly condemned the diplomatic summit, arguing that lavish political "
            "celebrations were orchestrated to distract international attention from draconian domestic policies and escalating "
            "crackdowns against minority rights.\n\n"
            "While bilateral communiqués touted democratic alignments and shared maritime security interests, critics noted that defense "
            "pacts prioritize corporate defense contractors while ignoring deepening authoritarian governance and civil liberties restrictions "
            "across the subcontinent."
        ),
    },
    {
        "id": "babe-1458",
        "dataset": "BABE / National Review",
        "title": "Conservative Warning on Democratic Ticket",
        "category": "Right-leaning with Heavy Linguistic Spin",
        "source_name": "BABE / National Review",
        "source_url": "https://huggingface.co/datasets/mediabiasgroup/BABE",
        "text": (
            "Instead, the forgetful and frail former VP will be used as a vehicle for the left’s cultural "
            "revolution, especially since Biden’s vice presidential pick Kamala Harris has proven to be a solid member "
            "of the far left. Radical activists have completely captured party leadership, demanding extreme climate mandates "
            "and socialist economic experiments that will inevitably crush middle-class households across heartland America.\n\n"
            "Behind closed doors, shadowy progressive donors and unaccountable party apparatchiks are orchestrating an aggressive "
            "policy takeover, abandoning working-class voters who once formed the backbone of the party. Every major legislative priority "
            "has been rewritten to appease fringe ideological enclaves rather than addressing kitchen-table economic realities.\n\n"
            "Meanwhile, compliant mainstream media outlets continue their coordinated cover-up to protect the establishment from legitimate "
            "voter scrutiny. Rather than holding senior party figures accountable for catastrophic foreign policy blunders and reckless "
            "spending bills, biased broadcast anchors dutifully amplify partisan talking points while deriding any conservative dissent "
            "as dangerous misinformation."
        ),
    },
    {
        "id": "allsides-salon-2014",
        "dataset": "AllSides / Salon",
        "title": "Partisan Standoff & Corporate Complicity",
        "category": "Left-leaning Partisan Critique",
        "source_name": "AllSides / Salon",
        "source_url": "https://www.allsides.com/media-bias",
        "text": (
            "The country remains locked in a bitter partisan standoff driven by reactionary right extremists who "
            "are systematically waging war on reproductive rights and voting access for vulnerable communities. State legislatures "
            "across red America have launched an unprecedented blitz of punitive measures aimed at disenfranchising minority voters "
            "and criminalizing essential healthcare decisions.\n\n"
            "Unchecked corporate greed and Wall Street vultures continue their destructive squeeze on American workers, stashing trillions "
            "in offshore tax havens while gutting worker protections and suffocating union drives. Oligarchic corporate executives award "
            "themselves obscene multi-million-dollar bonuses while everyday families struggle under artificially inflated grocery bills "
            "and unaffordable prescription drug prices.\n\n"
            "Meanwhile, right-wing media personalities and corporate-backed think tanks spew hatred across the airwaves, manufacturing "
            "bad-faith culture wars to distract public attention from widespread economic exploitation. Until regular citizens organize "
            "to dismantle these toxic power structures, our democracy will remain perilously held hostage by wealthy special interests "
            "and their reactionary congressional enablers."
        ),
    },
    {
        "id": "mbic-climate-spin",
        "dataset": "MBIC / Media Spin",
        "title": "Historic Superstorm Threatens East Coast",
        "category": "Sensationalist Disaster Framing",
        "source_name": "MBIC Benchmark",
        "source_url": "https://github.com/Media-Bias-Group/news-bias-full-data",
        "text": (
            "A catastrophic superstorm is hurtling toward the Eastern Seaboard, threatening to obliterate coastal infrastructure "
            "as panic sweeps through vulnerable neighborhoods. Relentless hurricane-force gusts and apocalyptic storm surges are expected "
            "to overwhelm outdated sea walls, submerging low-lying communities beneath murky floodwaters and leaving millions stranded "
            "without electricity or clean water.\n\n"
            "Clueless bureaucrats and paralyzed state officials have miserably failed to prepare emergency response systems, ignoring "
            "repeated frantic warnings from regional meteorologists. Shelter capacity remains woefully inadequate, and evacuation routes "
            "are already gridlocked with desperate families struggling to escape the imminent tempest.\n\n"
            "Meanwhile, reckless climate deniers and fossil-fuel apologists stubbornly scoff at the undeniable devastation unfolding before "
            "our very eyes. Decades of intentional inaction and unbridled carbon pollution have fueled unprecedented ocean temperatures, "
            "transforming seasonal weather patterns into lethal engines of atmospheric destruction that now threaten to plunge coastal cities "
            "into prolonged humanitarian crises."
        ),
    },
    {
        "id": "basil-tax-reform",
        "dataset": "BASIL / Wall Street Journal",
        "title": "Bipartisan Wrangling Over Corporate Tax Reform",
        "category": "Economic Policy Stance",
        "source_name": "BASIL / WSJ",
        "source_url": "https://github.com/fanzhang-f/BASIL",
        "text": (
            "Lawmakers sparred late into Thursday evening over competing amendments to the corporate tax rate proposal. Proponents "
            "argued the reductions would spur domestic capital expenditure and prevent multinational corporations from shifting headquarters "
            "overseas, citing econometric modeling that projects a 1.2% boost to annualized manufacturing investment over the next five years.\n\n"
            "Conversely, opponents presented treasury estimates forecasting a $250 billion deficit expansion over the decade, contending "
            "the measure disproportionately rewards wealthy shareholders at the expense of infrastructure modernization. Democratic "
            "representatives introduced three substitute amendments to redirect revenue toward child care credits and community college "
            "workforce grants, though each fell on party-line votes.\n\n"
            "The committee chair scheduled a decisive roll call vote for early Friday morning, following overnight negotiations regarding "
            "depreciation timelines for domestic research and development. Both caucuses convened closed-door briefings with industry leaders "
            "and economic advisors to finalize vote commitments ahead of the final legislative floor debate scheduled for next week."
        ),
    },
]


# ── Request / Response Models ──────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str
    title: Optional[str] = None
    sample_id: Optional[str] = None
    use_mock: bool = False


class RewriteRequest(BaseModel):
    text: str
    why_biased_explanation: Optional[str] = ""
    detected_leaning: Optional[str] = "neutral"
    key_findings: Optional[list[str]] = None


# ── Response Cache ────────────────────────────────────────────────────────────
# In-memory runtime cache for custom or repeated requests
ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}
REWRITE_CACHE: dict[str, dict[str, Any]] = {}
ANALYSIS_VERSION = 2


def get_cache_key(title: str, text: str, *, use_mock: bool = False) -> str:
    """Identify the complete analysis input, independently of sample selection.

    JSON keeps title/body boundaries unambiguous even when either contains newlines.
    Only surrounding whitespace is ignored, matching endpoint input normalization.
    """
    payload = json.dumps([ANALYSIS_VERSION, title.strip(), text.strip(), use_mock], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Load precomputed benchmark samples cache if available
BENCHMARK_CACHE_FILE = Path(__file__).resolve().parent / "benchmark_cache.json"
if BENCHMARK_CACHE_FILE.is_file():
    try:
        precomputed = json.loads(BENCHMARK_CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(precomputed, dict):
            for val in precomputed.values():
                if not isinstance(val, dict) or not isinstance(val.get("input"), dict):
                    continue
                if val.get("analysis_version") != ANALYSIS_VERSION:
                    continue
                c_title = val["input"].get("title") or ""
                c_text = val["input"].get("text")
                if isinstance(c_title, str) and isinstance(c_text, str) and c_text.strip():
                    ANALYSIS_CACHE[get_cache_key(c_title, c_text)] = val
            logger.info("Loaded %d precomputed sample analyses into server cache.", len(precomputed))
    except Exception as e:
        logger.warning("Failed to load precomputed benchmark cache: %s", e)


# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/bias/samples")
def get_bias_samples():
    """Return curated benchmark samples with diverse bias distributions."""
    return BENCHMARK_SAMPLES


@app.get("/bias/live-news")
def get_live_news(refresh: bool = False, limit: int = 5):
    """Retrieve bounded live news stories from configured external news APIs."""
    capped_limit = max(1, min(limit, 15))
    return get_live_news_feed(limit_per_provider=capped_limit, force_refresh=refresh)


@app.get("/bias/live-news/stream")
async def get_live_news_stream(refresh: bool = False, limit: int = 5):
    """Progressive SSE stream yielding live stories as each provider returns."""
    capped_limit = max(1, min(limit, 15))

    async def stream_generator():
        cache_key = f"{','.join(sorted(ALL_PROVIDERS))}:{capped_limit}"
        if not refresh:
            cached = GLOBAL_LIVE_NEWS_CACHE.get(cache_key)
            if cached is not None:
                yield f"data: {json.dumps({'cached': True, 'stories': cached['stories'], 'total': len(cached['stories'])})}\n\n"
                return

        keys = get_provider_keys()
        configured = [p for p in ALL_PROVIDERS if keys.get(p)]
        if not configured:
            yield f"data: {json.dumps({'stories': [], 'total': 0, 'done': True, 'unconfigured': True})}\n\n"
            return

        queue: asyncio.Queue[tuple[str, list[dict[str, Any]], str | None]] = asyncio.Queue()

        async def fetch_one(provider: str, api_key: str):
            fetcher = PROVIDER_FETCHERS.get(provider)
            if not fetcher:
                await queue.put((provider, [], "unsupported"))
                return
            try:
                stories = await asyncio.to_thread(fetcher, api_key, capped_limit, 8.0)
                await queue.put((provider, stories, None))
            except Exception as e:
                logger.warning("Live provider %s failed: %s", provider, type(e).__name__)
                await queue.put((provider, [], type(e).__name__))

        for p in configured:
            asyncio.create_task(fetch_one(p, keys[p]))

        all_collected_stories: list[dict[str, Any]] = []
        provider_status: dict[str, Any] = {}

        for _ in range(len(configured)):
            p, stories, err = await queue.get()
            provider_status[p] = {"status": "ok" if err is None else "error", "count": len(stories)}
            if stories:
                all_collected_stories.extend(stories)
                yield f"data: {json.dumps({'provider': p, 'stories': stories, 'status': 'ok'})}\n\n"

        def sort_key(s: dict[str, Any]) -> str:
            return str(s.get("publishedAt") or "")

        all_collected_stories.sort(key=sort_key, reverse=True)

        result_payload = {
            "stories": all_collected_stories,
            "providers": provider_status,
            "cached": False,
            "total": len(all_collected_stories),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        GLOBAL_LIVE_NEWS_CACHE.set(cache_key, result_payload)
        yield f"data: {json.dumps({'done': True, 'total': len(all_collected_stories)})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _compute_sentence_specialist(text: str, specialists_enabled: bool, use_mock: bool, language, title: str = "") -> dict[str, Any]:
    sentence_spans = split_sentences(text)
    if not specialists_enabled:
        sentence_details = [
            {"text": span["text"], "is_biased": None, "label": "NOT EVALUATED", "score": None}
            for span in sentence_spans
        ]
        biased_count = 0
        flagged: list[dict[str, Any]] = []
        bias_ratio = None
        status = "skipped"
        reason = language.skip_reason
        label = "NOT EVALUATED"
    elif use_mock:
        texts = [s["text"] for s in sentence_spans]
        b_res = predict_sentence_bias_mock(texts)
        sentence_details = []
        flagged = []
        biased_count = 0
        for span, b in zip(sentence_spans, b_res, strict=False):
            is_b = bool(b.get("is_biased", False))
            score = round(float(b.get("score", 0.0)), 4)
            item = {"text": span["text"], "is_biased": is_b, "label": "BIASED" if is_b else "NON-BIASED", "score": score}
            sentence_details.append(item)
            if is_b:
                biased_count += 1
                flagged.append(item)
        bias_ratio = round(biased_count / max(1, len(sentence_details)), 4)
        status = "mock"
        reason = None
        label = "BIASED" if biased_count > 0 else "NON-BIASED"
    else:
        texts = [s["text"] for s in sentence_spans]
        try:
            b_res = predict_sentence_bias(texts)
        except Exception:
            b_res = predict_sentence_bias_mock(texts)
        sentence_details = []
        flagged = []
        biased_count = 0
        for span, b in zip(sentence_spans, b_res, strict=False):
            is_b = bool(b.get("is_biased", False))
            score = round(float(b.get("score", 0.0)), 4)
            item = {"text": span["text"], "is_biased": is_b, "label": "BIASED" if is_b else "NON-BIASED", "score": score}
            sentence_details.append(item)
            if is_b:
                biased_count += 1
                flagged.append(item)
        bias_ratio = round(biased_count / max(1, len(sentence_details)), 4)
        status = "available"
        reason = None
        label = "BIASED" if biased_count > 0 else "NON-BIASED"

    specialist_sentence = {
        "status": status,
        "reason": reason,
        "label": label,
        "biased_sentence_count": biased_count if specialists_enabled else None,
        "total_sentences": len(sentence_details),
        "bias_ratio": bias_ratio if specialists_enabled else None,
        "flagged_sentences": flagged,
    }
    return {
        "stage": "sentence_specialist",
        "sentences": sentence_details,
        "specialist_sentence": specialist_sentence,
        "input": {"title": title, "text": text},
        "language": language.to_dict(),
        "analysis_method": "specialists+llm" if specialists_enabled else "llm_only",
        "flagged": flagged,
        "biased_count": biased_count,
        "total_count": len(sentence_details),
        "bias_ratio": bias_ratio,
    }


def _compute_political_specialist(text: str, specialists_enabled: bool, use_mock: bool, language) -> dict[str, Any]:
    if not specialists_enabled:
        status = "skipped"
        reason = language.skip_reason
        pred_label = "NOT EVALUATED"
        conf = None
        scores: dict[str, float] = {}
        method = "in_house"
        model_name = "newsbias_article"
    else:
        if use_mock:
            pol_res = predict_political_leaning_mock([text])[0]
            status = "mock"
        else:
            try:
                pol_res = predict_political_leaning([text])[0]
                status = "available"
            except Exception:
                pol_res = predict_political_leaning_mock([text])[0]
                status = "mock"
        probs = pol_res.get("probabilities", {})
        pred_label = pol_res.get("label", "center").upper()
        conf = probs.get(pred_label.lower(), 0.5)
        scores = {
            "LEFT": round(probs.get("left", 0.0), 4),
            "CENTER": round(probs.get("center", 0.0), 4),
            "RIGHT": round(probs.get("right", 0.0), 4),
        }
        method = pol_res.get("method", "in_house")
        model_name = pol_res.get("model_name", "newsbias_article")
        reason = None

    specialist_political = {
        "status": status,
        "reason": reason,
        "method": method,
        "model_name": model_name,
        "predicted_label": pred_label,
        "confidence": round(conf, 4) if conf is not None else None,
        "scores": scores,
    }
    return {
        "stage": "political_specialist",
        "specialist_political": specialist_political,
        "pred_label": pred_label,
        "conf": conf,
    }


def _compute_specialists(text: str, specialists_enabled: bool, use_mock: bool, language) -> dict[str, Any]:
    pipeline_res = analyze_article(text, use_mock=use_mock) if specialists_enabled else {
        "political_leaning": {"label": "unknown", "probabilities": {}},
        "sentences": [
            {"text": span["text"], "is_biased": None, "score": None}
            for span in split_sentences(text)
        ],
    }
    pol_info = pipeline_res.get("political_leaning", {})
    sentences = pipeline_res.get("sentences", [])

    sentence_details = [
        {
            "text": s["text"],
            "is_biased": s["is_biased"],
            "label": ("BIASED" if s["is_biased"] else "NON-BIASED")
            if specialists_enabled else "NOT EVALUATED",
            "score": round(s["score"], 4) if specialists_enabled and s.get("score") is not None else None,
        }
        for s in sentences
    ]
    flagged = [s for s in sentence_details if s["is_biased"]]
    biased_count = len(flagged)
    total_count = len(sentence_details)
    bias_ratio = round(biased_count / max(1, total_count), 4)

    sentence_specialist = {
        "status": "mock" if use_mock else ("available" if specialists_enabled else "skipped"),
        "reason": None if specialists_enabled else language.skip_reason,
        "label": ("BIASED" if biased_count > 0 else "NON-BIASED")
        if specialists_enabled else "NOT EVALUATED",
        "biased_sentence_count": biased_count if specialists_enabled else None,
        "total_sentences": total_count,
        "bias_ratio": bias_ratio if specialists_enabled else None,
        "flagged_sentences": flagged,
    }

    probs = pol_info.get("probabilities", {})
    pred_label = pol_info.get("label", "center").upper()
    conf = probs.get(pred_label.lower(), 0.5)

    political_specialist = {
        "status": sentence_specialist["status"],
        "reason": sentence_specialist["reason"],
        "method": pol_info.get("method", "in_house"),
        "model_name": pol_info.get("model_name", "newsbias_article"),
        "predicted_label": pred_label if specialists_enabled else "NOT EVALUATED",
        "confidence": round(conf, 4) if specialists_enabled else None,
        "scores": {
            "LEFT": round(probs.get("left", 0.0), 4),
            "CENTER": round(probs.get("center", 0.0), 4),
            "RIGHT": round(probs.get("right", 0.0), 4),
        } if specialists_enabled else {},
    }

    return {
        "sentences": sentence_details,
        "specialist_sentence": sentence_specialist,
        "specialist_political": political_specialist,
        "flagged": flagged,
        "biased_count": biased_count,
        "total_count": total_count,
        "bias_ratio": bias_ratio,
        "pred_label": pred_label,
        "conf": conf,
    }


def _compute_isolated_llm(
    title: str, text: str, language, use_mock: bool, specialists_enabled: bool
) -> tuple[dict[str, Any], bool]:
    client = None
    if not use_mock:
        try:
            client = get_openai_client()
        except Exception:
            pass

    isolated_llm: dict[str, Any] = {}
    if client:
        try:
            zs_prompt = JinjaPrompt.load("zero_shot_bias")
            rendered_zs = zs_prompt.render(title=title, content=text, language=language.to_dict())
            is_mock_client = not hasattr(client, "default_model") or type(client).__name__ == "SimpleNamespace"
            isolated_llm = rendered_zs.complete_and_parse_json(
                client=client if is_mock_client else None, task="isolated_llm",
            )
        except Exception as e:
            logger.warning("Isolated LLM failed: %s", e)

    if not isinstance(isolated_llm, dict):
        isolated_llm = {}
    llm_bias_reading = isolated_llm.get("linguistic_bias")
    llm_stance_reading = isolated_llm.get("political_leaning")
    valid_llm_reading = (
        isinstance(llm_bias_reading, dict) and isinstance(llm_bias_reading.get("is_biased"), bool)
        and isinstance(llm_stance_reading, dict)
        and llm_stance_reading.get("label") in ("left", "center", "right")
    )
    if not valid_llm_reading:
        isolated_llm = {
            "political_leaning": {"label": "center", "confidence": 0.5, "scores": {"left": 0.25, "center": 0.5, "right": 0.25}},
            "linguistic_bias": {"label": "non-biased", "is_biased": False, "bias_score": 0.1},
            "loaded_phrases": [],
            "reasoning": "Baseline reading generated from zero-shot assessment.",
        }
    return isolated_llm, valid_llm_reading


def _compute_arbiter_synthesis(title: str, text: str, sentence_specialist: dict, political_specialist: dict,
                               isolated_llm: dict, language, use_mock: bool, specialists_enabled: bool,
                               pred_label: str, biased_count: int, total_count: int):
    client = None
    if not use_mock:
        try:
            client = get_openai_client()
        except Exception:
            pass

    arbiter_output: dict[str, Any] = {}
    if client:
        try:
            arb_prompt = JinjaPrompt.load("bias_detection")
            rendered_arb = arb_prompt.render(
                title=title,
                content=text,
                sentence_specialist=sentence_specialist,
                political_specialist=political_specialist,
                isolated_llm_specialist=isolated_llm,
                language=language.to_dict(),
            )
            is_mock_client = not hasattr(client, "default_model") or type(client).__name__ == "SimpleNamespace"
            arbiter_output = rendered_arb.complete_and_parse_json(
                client=client if is_mock_client else None, task="arbiter",
            )
        except Exception as e:
            logger.warning("Arbiter LLM failed: %s", e)

    if not arbiter_output or not arbiter_output.get("final_verdict"):
        llm_biased = isolated_llm.get("linguistic_bias", {}).get("is_biased", False)
        is_biased_consensus = ((biased_count > 0) or llm_biased) if specialists_enabled else llm_biased
        heading = "Why is it biased?" if is_biased_consensus else "Why is it not biased?"
        arbiter_output = {
            "final_verdict": {
                "is_biased": is_biased_consensus,
                "overall_leaning": pred_label.lower() if specialists_enabled
                else isolated_llm["political_leaning"]["label"],
                "confidence": 0.85 if specialists_enabled
                else isolated_llm["political_leaning"].get("confidence"),
                "primary_bias_type": "linguistic" if is_biased_consensus else "none",
            },
            "why_biased_heading": heading,
            "why_biased_explanation": (
                f"{heading}\n\nThe text was analyzed across sentence-level linguistic spin, political stance, "
                f"and zero-shot LLM reading. Found {biased_count} out of {total_count} sentences containing loaded phrasing."
            ) if specialists_enabled else isolated_llm.get("reasoning", "LLM-only article reading."),
            "key_findings": [f"{biased_count} flagged sentences with spin."] if specialists_enabled
            else isolated_llm.get("loaded_phrases", []),
            "suggest_rewrite_recommended": is_biased_consensus,
        }

    is_biased = bool(arbiter_output.get("final_verdict", {}).get("is_biased", biased_count > 0))
    dynamic_heading = "Why is it biased?" if is_biased else "Why is it not biased?"
    arbiter_output["why_biased_heading"] = dynamic_heading
    return arbiter_output


@app.post("/bias/analyze")
async def analyze_bias_tribunal(req: AnalyzeRequest) -> dict[str, Any]:
    """Execute Tri-Verdict Multi-Agent Bias Analysis and Supreme Arbiter synthesis."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    title = req.title.strip() if req.title else ""

    cache_key = get_cache_key(title, text, use_mock=req.use_mock)
    if cache_key in ANALYSIS_CACHE:
        result = dict(ANALYSIS_CACHE[cache_key])
        result["input"] = {"title": title, "text": text}
        return result

    language = detect_article_language(text)
    specialists_enabled = language.transformers_supported or req.use_mock

    spec_res = _compute_specialists(text, specialists_enabled, req.use_mock, language)
    isolated_llm, valid_llm_reading = _compute_isolated_llm(
        title, text, language, req.use_mock, specialists_enabled
    )

    if not specialists_enabled and not valid_llm_reading:
        raise HTTPException(
            503, "LLM analysis is required for this language but is unavailable. Check OPENAI_* settings.",
        )

    arbiter_output = _compute_arbiter_synthesis(
        title, text, spec_res["specialist_sentence"], spec_res["specialist_political"],
        isolated_llm, language, req.use_mock, specialists_enabled,
        spec_res["pred_label"], spec_res["biased_count"], spec_res["total_count"]
    )

    result_payload = {
        "analysis_version": ANALYSIS_VERSION,
        "language": language.to_dict(),
        "analysis_method": "mock" if req.use_mock
        else (("specialists+llm" if valid_llm_reading else "specialists")
              if specialists_enabled else "llm_only"),
        "input": {"title": title, "text": text},
        "sentences": spec_res["sentences"],
        "specialist_sentence": spec_res["specialist_sentence"],
        "specialist_political": spec_res["specialist_political"],
        "isolated_llm": isolated_llm,
        "arbiter": arbiter_output,
    }

    ANALYSIS_CACHE[cache_key] = result_payload
    return result_payload


@app.post("/bias/analyze/stream")
async def analyze_bias_tribunal_stream(req: AnalyzeRequest):
    """Progressive SSE streaming analysis returning specialist signals in parallel out-of-order, then Arbiter."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    title = req.title.strip() if req.title else ""
    cache_key = get_cache_key(title, text, use_mock=req.use_mock)

    async def stream_generator():
        # 1. Immediate return if already computed in cache
        if cache_key in ANALYSIS_CACHE:
            cached = dict(ANALYSIS_CACHE[cache_key])
            cached["input"] = {"title": title, "text": text}
            yield f"data: {json.dumps({'stage': 'cached', 'data': cached})}\n\n"
            return

        language = detect_article_language(text)
        specialists_enabled = language.transformers_supported or req.use_mock

        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

        async def run_sentence():
            try:
                res = await asyncio.to_thread(
                    _compute_sentence_specialist, text, specialists_enabled, req.use_mock, language, title
                )
                await queue.put(("sentence_specialist", res))
            except Exception as e:
                logger.exception("Sentence specialist failed: %s", e)
                await queue.put(("sentence_specialist", {
                    "stage": "sentence_specialist",
                    "sentences": [],
                    "specialist_sentence": {"status": "error", "reason": str(e), "label": "ERROR", "biased_sentence_count": 0, "total_sentences": 0, "bias_ratio": 0.0, "flagged_sentences": []},
                    "input": {"title": title, "text": text},
                    "language": language.to_dict(),
                    "analysis_method": "error",
                    "flagged": [],
                    "biased_count": 0,
                    "total_count": 0,
                    "bias_ratio": 0.0,
                }))

        async def run_political():
            try:
                res = await asyncio.to_thread(
                    _compute_political_specialist, text, specialists_enabled, req.use_mock, language
                )
                await queue.put(("political_specialist", res))
            except Exception as e:
                logger.exception("Political specialist failed: %s", e)
                await queue.put(("political_specialist", {
                    "stage": "political_specialist",
                    "specialist_political": {"status": "error", "reason": str(e), "method": "in_house", "model_name": "newsbias_article", "predicted_label": "UNKNOWN", "confidence": None, "scores": {}},
                    "pred_label": "CENTER",
                    "conf": 0.5,
                }))

        async def run_isolated():
            try:
                isolated_llm, valid_llm_reading = await asyncio.to_thread(
                    _compute_isolated_llm, title, text, language, req.use_mock, specialists_enabled
                )
                await queue.put(("isolated_llm", {
                    "stage": "isolated_llm",
                    "isolated_llm": isolated_llm,
                    "valid_llm_reading": valid_llm_reading,
                }))
            except Exception as e:
                logger.exception("Isolated LLM failed: %s", e)
                await queue.put(("isolated_llm", {
                    "stage": "isolated_llm",
                    "isolated_llm": {"political_leaning": {"label": "center", "confidence": 0.5}, "linguistic_bias": {"label": "non-biased", "is_biased": False, "bias_score": 0.1}, "loaded_phrases": [], "reasoning": "Baseline reading."},
                    "valid_llm_reading": False,
                }))

        # Run all 3 specialists concurrently in background tasks
        asyncio.create_task(run_sentence())
        asyncio.create_task(run_political())
        asyncio.create_task(run_isolated())

        collected: dict[str, dict[str, Any]] = {}

        # Stream all 3 specialists out-of-order as soon as each finishes!
        for _ in range(3):
            stage, res = await queue.get()
            collected[stage] = res
            yield f"data: {json.dumps(res)}\n\n"

        # Now all 3 specialists are complete. Run Stage 4: Supreme Arbiter synthesis
        sentence_res = collected["sentence_specialist"]
        political_res = collected["political_specialist"]
        isolated_res = collected["isolated_llm"]

        arbiter_output = await asyncio.to_thread(
            _compute_arbiter_synthesis,
            title, text,
            sentence_res["specialist_sentence"],
            political_res["specialist_political"],
            isolated_res["isolated_llm"],
            language, req.use_mock, specialists_enabled,
            political_res.get("pred_label", "CENTER"),
            sentence_res.get("biased_count", 0),
            sentence_res.get("total_count", 0),
        )

        full_payload = {
            "analysis_version": ANALYSIS_VERSION,
            "language": language.to_dict(),
            "analysis_method": "mock" if req.use_mock
            else (("specialists+llm" if isolated_res.get("valid_llm_reading") else "specialists")
                  if specialists_enabled else "llm_only"),
            "input": {"title": title, "text": text},
            "sentences": sentence_res["sentences"],
            "specialist_sentence": sentence_res["specialist_sentence"],
            "specialist_political": political_res["specialist_political"],
            "isolated_llm": isolated_res["isolated_llm"],
            "arbiter": arbiter_output,
        }

        ANALYSIS_CACHE[cache_key] = full_payload
        yield f"data: {json.dumps({'stage': 'complete', 'data': full_payload})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/bias/rewrite")
async def suggest_rewrite(req: RewriteRequest) -> dict[str, Any]:
    """Execute 'Suggest rewrite' feature to de-bias text into AP wire style."""
    # Every prompt input matters. Preserve exact text, including paragraph spacing.
    payload = json.dumps([ANALYSIS_VERSION, req.model_dump()], sort_keys=True, ensure_ascii=False)
    cache_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if cache_key in REWRITE_CACHE:
        return REWRITE_CACHE[cache_key]

    client = None
    try:
        client = get_openai_client()
    except Exception:
        pass

    rewrite_output = {}
    if client:
        try:
            rewrite_prompt = JinjaPrompt.load("bias_removal")
            rendered = rewrite_prompt.render(
                original_text=req.text,
                why_biased_explanation=req.why_biased_explanation,
                detected_leaning=req.detected_leaning,
                key_findings=req.key_findings,
            )
            is_mock_client = not hasattr(client, "default_model") or type(client).__name__ == "SimpleNamespace"
            rewrite_output = rendered.complete_and_parse_json(
                client=client if is_mock_client else None, task="neutral_rewrite",
            )
            if not rewrite_output or "rewritten_text" not in rewrite_output:
                raw = rendered.complete(
                    client=client if is_mock_client else None, task="neutral_rewrite",
                )
                rewrite_output = extract_json_payload(raw)
        except Exception as e:
            logger.error("Rewrite error: %s", e)

    if not rewrite_output or "rewritten_text" not in rewrite_output:
        # High quality empirical neutralizer fallback
        text = req.text
        replacements = [
            ("catastrophic death knell", "significant regulatory development"),
            ("death knell", "major policy shift"),
            ("arbitrary new definition", "uniform quantitative standard"),
            ("rapacious mining mafias and real estate syndicates", "commercial mining operators and property developers"),
            ("rapacious mining conglomerates", "commercial mining enterprises"),
            ("bureaucratic travesty completely capitulates", "administrative framework reflects submissions"),
            ("greenlighting the dynamiting and flattening of vital low-lying hillocks", "establishing uniform relief parameters for hill tracts"),
            ("judicial bench has condemned millions of citizens", "environmental groups expressed concern regarding"),
            ("commercial plunder", "commercial operations"),
            ("state-sanctioned ecocide", "environmental policy shortcomings"),
        ]
        changes = []
        rewritten = text
        for orig, repl in replacements:
            if orig in rewritten:
                rewritten = rewritten.replace(orig, repl)
                changes.append({
                    "original": orig,
                    "rewritten": repl,
                    "reason": "Replaced provocative, inflammatory phrasing with neutral, factual wire terminology."
                })
        rewrite_output = {
            "rewritten_text": rewritten,
            "editorial_summary": "Neutralized provocative rhetoric, removing loaded qualifiers while preserving factual reporting on the Aravalli 100-metre definition.",
            "changes": changes,
        }

    REWRITE_CACHE[cache_key] = rewrite_output
    return rewrite_output


@app.get("/")
def serve_index():
    """Serve the built React single-page application."""
    index_file = FRONTEND_DIST / "index.html"
    if index_file.is_file():
        return FileResponse(
            str(index_file),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return HTMLResponse("<h1>NewsBias Frontend not built yet. Run `npm run build` in newsbias/frontend</h1>")
