"""
FactVibe: AI Hallucination Auditor
===================================
A production-quality Streamlit application that detects hallucinations
and verifies factual claims in AI-generated content using Groq (Qwen3-32B)
and DuckDuckGo search evidence.

Improvements implemented:
  1. Evidence-grounded verification prompt with evidence_strength field
  2. Structured evidence objects (list[dict]) from search_evidence()
  3. Source credibility scoring via calculate_source_credibility()
  4. Enhanced claim expander with strength + credibility display
  5. Dead code removed (pandas, VERDICT_EMOJIS, SCORE_THRESHOLDS, TA_LEFT)
  6. Extended dashboard metrics (avg confidence, avg credibility)
  7. Enhanced PDF with evidence strength, credibility, and URLs
  UI: Full enterprise SaaS redesign — all emojis replaced with badges/typography
"""
# pyrefly: ignore  # CSS inside st.markdown() strings causes false positive parse errors
# type: ignore     # same for mypy / pyright

import os
import re
import json
import time
import logging
import datetime
import pathlib
import traceback
from urllib.parse import urlparse
from typing import Optional

logger = logging.getLogger("factvibe")

import streamlit as st
from dotenv import load_dotenv
from ddgs import DDGS
import groq as groq_sdk
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER
import io

# ─────────────────────────────────────────────────────────────────────────────
# Environment & Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Primary model: Qwen3-32B on Groq.
# Fallback: llama-3.3-70b-versatile (used if the primary model is unavailable).
GROQ_MODEL_PRIMARY  = "qwen/qwen3-32b"
GROQ_MODEL_FALLBACK = "llama-3.3-70b-versatile"

# Active model — resolved at client-init time (see get_groq_client())
GROQ_MODEL: str = GROQ_MODEL_PRIMARY

# Verdict colour palette (hex) — used for badges and borders
VERDICT_COLORS: dict[str, str] = {
    "Verified":           "#22c55e",
    "Partially Supported": "#f59e0b",
    "Unverified":          "#eab308",
    "Contradicted":        "#ef4444",
}

# Evidence-strength colour palette
STRENGTH_COLORS: dict[str, str] = {
    "Strong":       "#22c55e",
    "Moderate":     "#60a5fa",
    "Weak":         "#f59e0b",
    "Insufficient": "#6b7280",
}

# High-credibility domain keywords (used in calculate_source_credibility)
_RESEARCH_KEYWORDS  = {"arxiv", "nature", "ieee", "springer", "pubmed", "ncbi",
                        "sciencedirect", "nih", "who", "cdc"}
_NEWS_KEYWORDS      = {"reuters", "apnews", "bbc", "nytimes", "theguardian",
                       "washingtonpost", "bloomberg", "economist", "ft",
                       "thehindu", "ndtv", "hindustantimes"}

# ─────────────────────────────────────────────────────────────────────────────
# Evidence Retrieval — Query Optimisation Helpers
# ─────────────────────────────────────────────────────────────────────────────

# English stop words to strip when building focused search queries
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "and", "but", "or", "nor", "so", "yet", "not", "only",
    "that", "this", "its", "it", "about", "which", "who", "what",
    "when", "where", "why", "how", "all", "each", "every", "some",
    "such", "any", "because", "if", "although", "while", "since",
    "just", "very", "than", "too", "both", "either", "neither", "own",
    "located", "known", "called", "named", "said",
})

# Minimum keyword-overlap ratio to keep a search result (0 = keep all, 1 = perfect)
_MIN_RELEVANCE: float = 0.15


def _build_search_query(claim: str) -> str:
    """
    Build a focused DuckDuckGo query by extracting meaningful keywords
    from a factual claim, stripping stop words and short tokens.

    Example:
        "The capital of France is Paris."  →  "capital France Paris"
        "The Eiffel Tower is located in Paris"  →  "Eiffel Tower Paris"

    Returns the optimised query string (falls back to the original claim
    if no keywords remain after filtering).
    """
    tokens = re.findall(r"[a-zA-Z0-9'\-]+", claim)
    keywords: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        lower = tok.lower()
        if lower not in _STOP_WORDS and len(lower) > 2:
            if lower not in seen:
                seen.add(lower)
                keywords.append(tok)
    query = " ".join(keywords[:8])   # cap at 8 terms to keep query tight
    return query if query.strip() else claim


def _relevance_score(claim: str, result: dict) -> float:
    """
    Compute a relevance score (0.0–1.0) between a claim and a search result.

    Score = (claim keywords that appear in title+snippet) / total claim keywords.

    Args:
        claim:  The factual claim string.
        result: Dict with "title" and "snippet" keys.

    Returns:
        Float in [0.0, 1.0].  Returns 0.5 if the claim has no scoreable keywords
        (so the result is passed through rather than silently dropped).
    """
    claim_tokens = {
        t.lower()
        for t in re.findall(r"[a-zA-Z0-9]+", claim)
        if t.lower() not in _STOP_WORDS and len(t) > 2
    }
    if not claim_tokens:
        return 0.5

    result_text  = (
        result.get("title", "") + " " + result.get("snippet", "")
    ).lower()
    result_tokens = set(re.findall(r"[a-zA-Z0-9]+", result_text))

    overlap = claim_tokens & result_tokens
    return len(overlap) / len(claim_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Groq Client Initialisation
# ─────────────────────────────────────────────────────────────────────────────

# Module-level singleton — initialised once via get_groq_client()
_groq_client: Optional[Groq] = None


def get_groq_client() -> Optional[Groq]:
    """
    Initialise and return a singleton Groq client.

    Probes the primary model (qwen/qwen3-32b) and falls back to
    llama-3.3-70b-versatile if the primary is unavailable.

    Returns:
        groq.Groq instance if GROQ_API_KEY is set, otherwise None.
    """
    global _groq_client, GROQ_MODEL

    if _groq_client is not None:
        return _groq_client

    if not GROQ_API_KEY:
        return None

    try:
        client = Groq(api_key=GROQ_API_KEY)

        # Probe primary model with a minimal token request to detect availability.
        try:
            client.chat.completions.create(
                model=GROQ_MODEL_PRIMARY,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            GROQ_MODEL = GROQ_MODEL_PRIMARY
        except groq_sdk.APIStatusError as probe_exc:
            # Model-not-found or similar: fall back silently
            if probe_exc.status_code in (400, 404):
                GROQ_MODEL = GROQ_MODEL_FALLBACK
            else:
                raise probe_exc

        _groq_client = client
        return _groq_client

    except Exception as exc:
        st.error(f"Failed to initialise Groq client: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core LLM Helper: call_llm
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    max_tokens: int = 1024,
    retries: int = 5,
    base_delay: float = 10.0,
) -> str:
    """
    Send a prompt to Groq and return plain-text output.

    Retry policy:
        - groq.RateLimitError     → exponential back-off (base 10 s, ×1.5 each attempt)
        - groq.APIStatusError 5xx → retry up to `retries` times
        - groq.APITimeoutError    → retry up to `retries` times

    Args:
        prompt:     The user prompt string.
        max_tokens: Maximum tokens in the completion.
        retries:    Maximum number of attempts before giving up.
        base_delay: Base sleep duration (seconds) for back-off.

    Returns:
        Plain-text content of the model's response.

    Raises:
        RuntimeError: If the client is not initialised or all retries are exhausted.
    """
    client = get_groq_client()
    if client is None:
        raise RuntimeError(
            "Groq client is not initialised. "
            "Ensure GROQ_API_KEY is set in your .env file."
        )

    last_exc: Optional[Exception] = None

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""

        except groq_sdk.RateLimitError as exc:
            last_exc = exc
            if attempt < retries - 1:
                delay = base_delay * (1.5 ** attempt)
                st.toast(f"Rate limit hit. Retrying in {delay:.1f}s…", icon="⏳")
                time.sleep(delay)

        except groq_sdk.APIStatusError as exc:
            last_exc = exc
            # Only retry on server-side errors (5xx); surface client errors immediately
            if exc.status_code >= 500 and attempt < retries - 1:
                delay = base_delay * (1.5 ** attempt)
                st.toast(f"Server error ({exc.status_code}). Retrying in {delay:.1f}s…", icon="⏳")
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"Groq API error (HTTP {exc.status_code}): {exc.message}"
                ) from exc

        except groq_sdk.APITimeoutError as exc:
            last_exc = exc
            if attempt < retries - 1:
                delay = base_delay * (1.5 ** attempt)
                st.toast(f"Request timed out. Retrying in {delay:.1f}s…", icon="⏳")
                time.sleep(delay)
            else:
                raise RuntimeError(
                    "Groq API request timed out after multiple retries."
                ) from exc

        except Exception as exc:
            # Surface unexpected errors immediately — don't mask them
            raise RuntimeError(f"Unexpected error calling Groq: {exc}") from exc

    raise RuntimeError(
        f"Groq API call failed after {retries} attempts. "
        f"Last error: {last_exc}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON Parsing Helper
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_response(raw_text: str) -> object:
    """
    Robustly parse a JSON response from the LLM.

    Handles:
      - Responses wrapped in markdown code fences (```json ... ```)
      - Leading/trailing whitespace
      - Models that add prose before or after the JSON block

    Args:
        raw_text: Raw string output from the LLM.

    Returns:
        Parsed Python object (list or dict).

    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted.
    """
    raw = raw_text.strip()

    # Strip markdown fences: ```json ... ``` or ``` ... ```
    if raw.startswith("```"):
        # Find the closing fence
        end_fence = raw.rfind("```")
        if end_fence > 3:
            # Strip opening fence line and closing fence
            inner = raw[3:end_fence]
            # Remove optional language tag on first line (e.g. "json\n")
            first_newline = inner.find("\n")
            if first_newline != -1:
                first_line = inner[:first_newline].strip().lower()
                if first_line in ("json", ""):
                    inner = inner[first_newline + 1:]
            raw = inner.strip()

    # If still not valid, try extracting first JSON array or object
    if not (raw.startswith("{") or raw.startswith("[")):
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start_idx = raw.find(start_char)
            end_idx   = raw.rfind(end_char)
            if start_idx != -1 and end_idx > start_idx:
                raw = raw[start_idx : end_idx + 1]
                break

    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Claim Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_claims(text: str) -> list[dict]:
    """
    Use Groq to extract 3–7 important factual claims from the provided text.

    Args:
        text: Raw text to analyse.

    Returns:
        List of dicts with key "claim", e.g. [{"claim": "..."}].
        Returns an empty list on failure.
    """
    prompt = f"""You are a professional fact-checker. Extract 3 to 7 important, \
verifiable factual claims from the text below.

RULES:
- Focus only on objective, checkable facts (dates, numbers, names, locations, events).
- Ignore opinions, subjective statements, or unverifiable claims.
- Return ONLY a valid JSON array. No markdown, no explanation, no extra text.
- Each element must have a single key: "claim".

OUTPUT FORMAT (strict JSON):
[
  {{"claim": "..."}},
  {{"claim": "..."}}
]

TEXT TO ANALYSE:
\"\"\"{text}\"\"\"
"""
    try:
        raw_text = call_llm(prompt, max_tokens=1024)
        if not raw_text:
            return []
        claims = _parse_json_response(raw_text)
        if isinstance(claims, list):
            return [c for c in claims if isinstance(c, dict) and "claim" in c]
        return []
    except json.JSONDecodeError:
        st.warning("Groq returned malformed JSON for claim extraction.")
        return []
    except RuntimeError as exc:
        err_msg = str(exc)
        if "rate limit" in err_msg.lower() or "429" in err_msg:
            st.error(
                "Groq API rate limit exceeded. Please wait a moment and try again."
            )
        else:
            st.error(f"Claim extraction failed: {err_msg}")
        return []
    except Exception as exc:
        st.error(f"Claim extraction failed: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Evidence Retrieval  (structured list[dict])
# ─────────────────────────────────────────────────────────────────────────────

def search_evidence(
    claim: str,
    max_results: int = 3,
) -> tuple[list[dict], dict]:
    """
    Search DuckDuckGo for evidence related to a factual claim.

    Strategy:
      1. Build an optimised keyword query (stop words removed).
      2. Fetch up to max_results * 3 raw results for scoring headroom.
      3. Score each result for relevance to the claim.
      4. Sort by score descending; reject results below _MIN_RELEVANCE.
      5. If all results are rejected, fall back to the top-scored unfiltered set.
      6. If the keyword query returns nothing, retry with the full claim string.

    Args:
        claim:       The factual claim to verify.
        max_results: Number of evidence items to return.

    Returns:
        Tuple of:
          - evidence list (list[dict], keys: title, snippet, url, _score)
          - debug dict   (keys: query, raw_count, scored, filtered_count)
    """
    query = _build_search_query(claim)
    logger.info("[FactVibe] Evidence query for claim %r → %r", claim[:60], query)

    _empty_debug: dict = {
        "query": query, "raw_count": 0, "scored": [], "filtered_count": 0
    }

    try:
        # ── Fetch candidates ──────────────────────────────────────────────
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results * 3))

        # Fallback: retry with the full claim if keyword query yields nothing
        if not raw:
            logger.info("[FactVibe] No results for keyword query; retrying with full claim.")
            with DDGS() as ddgs:
                raw = list(ddgs.text(claim, max_results=max_results * 2))

        if not raw:
            logger.info("[FactVibe] No search results at all for claim %r", claim[:60])
            return (
                [{"title": "No results", "snippet": "No reliable evidence found.", "url": "", "_score": 0.0}],
                {**_empty_debug, "raw_count": 0},
            )

        # ── Score and structure ───────────────────────────────────────────
        scored: list[tuple[float, dict]] = []
        for item in raw:
            structured = {
                "title":   item.get("title", "No title"),
                "snippet": item.get("body",  "No snippet available."),
                "url":     item.get("href",  ""),
            }
            score = _relevance_score(claim, structured)
            structured["_score"] = round(score, 3)
            scored.append((score, structured))
            logger.debug(
                "[FactVibe]   score=%.2f  %s  (%s)",
                score, structured["title"][:60], structured["url"][:60],
            )

        # Sort highest-relevance first
        scored.sort(key=lambda x: x[0], reverse=True)

        # ── Filter by relevance threshold ─────────────────────────────────
        above_threshold = [(s, r) for s, r in scored if s >= _MIN_RELEVANCE]

        if above_threshold:
            final = [r for _, r in above_threshold[:max_results]]
        else:
            # All results failed the threshold — keep the best ones anyway
            # (avoids silently discarding all evidence for niche topics)
            logger.info(
                "[FactVibe] All %d results below relevance threshold %.2f; "
                "using top %d unfiltered.",
                len(scored), _MIN_RELEVANCE, max_results,
            )
            final = [r for _, r in scored[:max_results]]

        debug_info: dict = {
            "query":          query,
            "raw_count":      len(raw),
            "scored":         [(round(s, 3), r["title"][:70], r["url"][:70]) for s, r in scored],
            "filtered_count": len(final),
        }
        logger.info(
            "[FactVibe] Returning %d/%d results (threshold=%.2f).",
            len(final), len(raw), _MIN_RELEVANCE,
        )
        return final, debug_info

    except Exception as exc:
        logger.exception("[FactVibe] search_evidence() error: %s", exc)
        err = [{"title": "Search Error",
                "snippet": f"Evidence retrieval error: {str(exc)[:200]}",
                "url": "", "_score": 0.0}]
        return err, {**_empty_debug, "raw_count": -1}


def _evidence_to_text(evidence: list[dict]) -> str:
    """
    Convert structured evidence list to a plain-text block for LLM prompts.

    Args:
        evidence: List of evidence dicts (title, snippet, url).

    Returns:
        Formatted multi-line string.
    """
    parts = []
    for i, item in enumerate(evidence, 1):
        parts.append(
            f"[Source {i}] {item['title']}\n"
            f"Snippet: {item['snippet']}\n"
            f"URL: {item['url']}"
        )
    return "\n\n".join(parts) if parts else "No evidence available."


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Claim Verification  (strict evidence-only prompt)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_confidence(raw_value: object) -> int:
    """
    Convert a model-returned confidence value to an integer on the 0–100 scale.

    Models sometimes return confidence on a 0–1 probability scale (e.g. 0.85)
    instead of the requested 0–100 percentage scale (e.g. 85).  This function
    detects that case and rescales automatically.

    Rules:
        - Parse the value as a float first (handles int, float, and string inputs).
        - If the float is strictly between 0.0 and 1.0 (exclusive), multiply
          by 100 to convert from probability to percentage.
        - Clamp the final result to [0, 100] and return as int.

    Examples:
        0.85  →  85      (0-1 scale detected, auto-converted)
        0.0   →   0      (boundary: kept as-is)
        1.0   →   1      (boundary: treated as 1%, not 100%)
        75    →  75      (already on 0-100 scale)
        100   → 100      (already on 0-100 scale)
        "0.9" →  90      (string input handled)
    """
    try:
        value = float(raw_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0

    # Detect 0-1 probability scale: strictly between 0 and 1 (exclusive)
    if 0.0 < value < 1.0:
        value = value * 100.0

    return max(0, min(100, int(round(value))))


def verify_claim(claim: str, evidence: list[dict]) -> dict:
    """
    Use Groq to verify a single claim against retrieved evidence.

    The prompt enforces strict evidence-based reasoning — the model must NOT
    rely on its own training knowledge.

    Args:
        claim:    The factual claim to evaluate.
        evidence: Structured evidence list from search_evidence().

    Returns:
        Dict with keys: verdict, confidence (0-100), evidence_strength, reason.
        Falls back to a safe default dict on error.
    """
    evidence_text = _evidence_to_text(evidence)

    prompt = f"""You are an evidence-based fact-checking system.

IMPORTANT RULES:
1. You MUST evaluate the claim ONLY using the evidence provided below.
2. Do NOT use your own knowledge.
3. Do NOT make assumptions beyond the evidence.
4. If the evidence clearly supports the claim, return "Verified".
5. If the evidence partially supports the claim or is ambiguous, return "Partially Supported".
6. If the evidence is insufficient to confirm or deny the claim, return "Unverified".
7. If the evidence clearly contradicts the claim, return "Contradicted".

Your decision must be based solely on the supplied evidence.

CLAIM:
"{claim}"

EVIDENCE:
{evidence_text}

Return ONLY valid JSON. No markdown, no explanation, no extra text.

OUTPUT FORMAT (strict JSON):
{{
  "verdict": "Verified | Partially Supported | Unverified | Contradicted",
  "confidence": 85,
  "evidence_strength": "Strong | Moderate | Weak | Insufficient",
  "reason": "One sentence explanation based only on the evidence above."
}}

IMPORTANT: "confidence" MUST be an integer between 0 and 100 (percentage scale, NOT 0–1).
"""
    default: dict = {
        "verdict":           "Unverified",
        "confidence":        0,
        "evidence_strength": "Insufficient",
        "reason":            "Verification could not be completed due to an error.",
    }

    try:
        raw_text = call_llm(prompt, max_tokens=512)
        if not raw_text:
            return default
        result = _parse_json_response(raw_text)

        if not isinstance(result, dict):
            return default

        # Normalise and validate fields
        valid_verdicts  = {"Verified", "Partially Supported", "Unverified", "Contradicted"}
        valid_strengths = {"Strong", "Moderate", "Weak", "Insufficient"}

        if result.get("verdict") not in valid_verdicts:
            result["verdict"] = "Unverified"
        if result.get("evidence_strength") not in valid_strengths:
            result["evidence_strength"] = "Insufficient"

        result["confidence"] = _normalise_confidence(result.get("confidence", 0))
        result["reason"]     = str(result.get("reason", "No reason provided."))
        return result

    except json.JSONDecodeError:
        default["reason"] = "Groq returned malformed JSON during verification."
        return default
    except RuntimeError as exc:
        err_msg = str(exc)
        if "rate limit" in err_msg.lower() or "429" in err_msg:
            default["reason"] = "Groq API rate limit exceeded."
        else:
            default["reason"] = f"Verification error: {err_msg[:150]}"
        return default
    except Exception as exc:
        default["reason"] = f"Verification error: {str(exc)[:150]}"
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Source Credibility Scoring
# ─────────────────────────────────────────────────────────────────────────────

def calculate_source_credibility(url: str) -> float:
    """
    Assign a credibility score (1–5) to a source URL based on domain heuristics.

    Scoring rubric:
        5 — Government (.gov), Educational (.edu / .ac.in / .edu.in),
            or peer-reviewed research platforms (arxiv, nature, ieee, etc.)
        4 — Major news organisations (Reuters, BBC, NYT, etc.)
        3 — Wikipedia
        2 — Blogs / content platforms
        1 — Unknown or unrecognised domains

    Args:
        url: Source URL string.

    Returns:
        Float credibility score between 1.0 and 5.0.
    """
    if not url:
        return 1.0

    try:
        parsed   = urlparse(url)
        hostname = parsed.netloc.lower().removeprefix("www.")
    except Exception:
        return 1.0

    # Government domains
    if hostname.endswith(".gov") or ".gov." in hostname:
        return 5.0

    # Educational domains
    if (hostname.endswith(".edu")
            or hostname.endswith(".ac.in")
            or hostname.endswith(".edu.in")
            or ".edu." in hostname):
        return 5.0

    # Peer-reviewed / research platforms
    for kw in _RESEARCH_KEYWORDS:
        if kw in hostname:
            return 5.0

    # Wikipedia
    if "wikipedia.org" in hostname:
        return 3.0

    # Major news sites
    for kw in _NEWS_KEYWORDS:
        if kw in hostname:
            return 4.0

    # Blog platforms — checked BEFORE generic heuristics to avoid false positives
    # (e.g. wordpress.com contains "press" but is a blog host, not a news outlet)
    blog_indicators = {"blogspot", "wordpress", "medium", "substack", "tumblr", "quora"}
    for kw in blog_indicators:
        if kw in hostname:
            return 2.0

    # Heuristic: domains with "news" or "press" in them (lower priority than explicit blog list)
    if "news" in hostname or "press" in hostname:
        return 3.0

    return 1.0


def average_credibility(evidence: list[dict]) -> float:
    """
    Compute the mean credibility score for a list of evidence sources.

    Args:
        evidence: Structured evidence list (each item has a "url" key).

    Returns:
        Average credibility score rounded to 1 decimal place, or 0.0 if empty.
    """
    if not evidence:
        return 0.0
    scores = [calculate_source_credibility(item.get("url", "")) for item in evidence]
    return round(sum(scores) / len(scores), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination Score
# ─────────────────────────────────────────────────────────────────────────────

def calculate_hallucination_score(results: list[dict]) -> float:
    """
    Calculate a hallucination risk score as a percentage.

    Formula: (unverified + contradicted) / total_claims * 100

    Args:
        results: List of verification result dicts (must have "verdict" key).

    Returns:
        Float score between 0.0 and 100.0.
    """
    if not results:
        return 0.0
    risky = sum(1 for r in results if r.get("verdict") in {"Unverified", "Contradicted"})
    return round((risky / len(results)) * 100, 1)


def score_color(score: float) -> str:
    """Return a hex colour based on hallucination score thresholds."""
    if score <= 20:
        return "#22c55e"
    elif score <= 50:
        return "#f59e0b"
    return "#ef4444"


def score_label(score: float) -> str:
    """Return a risk label string based on hallucination score."""
    if score <= 20:
        return "LOW"
    elif score <= 50:
        return "MODERATE"
    return "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# PDF Report Generation  (enhanced with strength + credibility)
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    input_text: str,
    claims: list[dict],
    results: list[dict],
    hallucination_score: float,
) -> bytes:
    """
    Generate a downloadable PDF audit report using ReportLab.

    Args:
        input_text:          Original user-provided text.
        claims:              Extracted claims list.
        results:             Verification results list (includes evidence list[dict]).
        hallucination_score: Calculated hallucination risk score (0-100).

    Returns:
        PDF file as raw bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Custom styles ──────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "FVTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#1e1b4b"),
        spaceAfter=4,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "FVSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=14,
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "FVSection",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1e1b4b"),
        spaceBefore=14,
        spaceAfter=5,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "FVBody",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#374151"),
        spaceAfter=6,
        leading=14,
    )
    url_style = ParagraphStyle(
        "FVUrl",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#4f46e5"),
        spaceAfter=3,
        leading=12,
    )

    # ── Header ─────────────────────────────────────────────────────────────
    story.append(Paragraph("FactVibe: AI Hallucination Auditor", title_style))
    story.append(Paragraph("Evidence-Based Fact Verification Audit Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e1b4b")))
    story.append(Spacer(1, 10))

    # ── Metadata ───────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Compute aggregate metrics for metadata block
    confidences = [r.get("confidence", 0) for r in results]
    avg_conf    = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

    credibilities = [
        average_credibility(r.get("evidence", []))
        for r in results
        if r.get("evidence")
    ]
    avg_cred = round(sum(credibilities) / len(credibilities), 1) if credibilities else 0.0

    meta_data = [
        ["Generated At:",          timestamp],
        ["Total Claims:",          str(len(claims))],
        ["Hallucination Risk:",    f"{hallucination_score}%  ({score_label(hallucination_score)} RISK)"],
        ["Avg. Confidence:",       f"{avg_conf}%"],
        ["Avg. Source Credibility:", f"{avg_cred} / 5"],
    ]
    meta_table = Table(meta_data, colWidths=[2.2 * inch, 4.3 * inch])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",   (0, 0), (0, -1),  colors.HexColor("#1e1b4b")),
        ("TEXTCOLOR",   (1, 0), (1, -1),  colors.HexColor("#374151")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ── Original Input ─────────────────────────────────────────────────────
    story.append(Paragraph("Original Input Text", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 4))
    truncated = input_text[:1000] + ("..." if len(input_text) > 1000 else "")
    story.append(Paragraph(truncated.replace("\n", "<br/>"), body_style))
    story.append(Spacer(1, 8))

    # ── Summary Table ──────────────────────────────────────────────────────
    story.append(Paragraph("Verification Summary", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 4))

    verdict_counts: dict[str, int] = {
        v: 0 for v in ["Verified", "Partially Supported", "Unverified", "Contradicted"]
    }
    for r in results:
        v = r.get("verdict", "Unverified")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    summary_data = [["Metric", "Value"]]
    for verdict, count in verdict_counts.items():
        summary_data.append([verdict, str(count)])
    summary_data.append(["Hallucination Score", f"{hallucination_score}%"])
    summary_data.append(["Average Confidence",  f"{avg_conf}%"])
    summary_data.append(["Avg. Source Credibility", f"{avg_cred} / 5"])

    summary_table = Table(summary_data, colWidths=[3.5 * inch, 1.5 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1e1b4b")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f9fafb"), colors.white]),
        ("FONTNAME",      (0, -3), (-1, -1), "Helvetica-Bold"),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # ── Claim Details ──────────────────────────────────────────────────────
    story.append(Paragraph("Detailed Claim Analysis", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 6))

    verdict_hex_map: dict[str, str] = {
        "Verified":           "#16a34a",
        "Partially Supported": "#d97706",
        "Unverified":          "#ca8a04",
        "Contradicted":        "#dc2626",
    }

    for idx, (claim_dict, result) in enumerate(zip(claims, results), 1):
        claim_text        = claim_dict.get("claim",            "N/A")
        verdict           = result.get("verdict",              "Unverified")
        confidence        = result.get("confidence",           0)
        evidence_strength = result.get("evidence_strength",    "Insufficient")
        reason            = result.get("reason",               "N/A")
        evidence_list     = result.get("evidence",             [])
        cred_score        = average_credibility(evidence_list)
        verdict_color_hex = verdict_hex_map.get(verdict, "#6b7280")

        claim_data = [
            [f"Claim {idx}", claim_text],
            ["Verdict",      verdict],
            ["Confidence",   f"{confidence}%"],
            ["Ev. Strength", evidence_strength],
            ["Credibility",  f"{cred_score} / 5"],
            ["Analysis",     reason],
        ]

        # Top evidence URLs
        top_urls = [
            item.get("url", "") for item in evidence_list if item.get("url")
        ][:3]
        if top_urls:
            claim_data.append(["Top Sources", "\n".join(top_urls)])

        claim_table = Table(claim_data, colWidths=[1.5 * inch, 5 * inch])
        claim_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, -1), colors.HexColor("#f3f4f6")),
            ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("TEXTCOLOR",     (1, 1), (1, 1),  colors.HexColor(verdict_color_hex)),
            ("FONTNAME",      (1, 1), (1, 1),  "Helvetica-Bold"),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        story.append(claim_table)
        story.append(Spacer(1, 8))

    # ── Footer ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Report generated by FactVibe  |  Powered by Groq ({GROQ_MODEL}) + DuckDuckGo Search",
        ParagraphStyle(
            "FVFooter",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#9ca3af"),
            alignment=TA_CENTER,
        ),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ─────────────────────────────────────────────────────────────────────────────
# UI Helpers  (enterprise SaaS design — no emojis)
# ─────────────────────────────────────────────────────────────────────────────

def render_verdict_badge(verdict: str) -> str:
    """
    Return an HTML bracket-style verdict badge string.

    Example output: <span ...>[ VERIFIED ]</span>
    """
    color = VERDICT_COLORS.get(verdict, "#6b7280")
    label = verdict.upper()
    return (
        f'<span style="'
        f'background:{color}18; color:{color}; '
        f'padding:4px 12px; border-radius:4px; font-weight:700; '
        f'border:1px solid {color}55; font-size:0.78rem; '
        f'letter-spacing:0.08em; font-family:monospace;">'
        f'[ {label} ]'
        f'</span>'
    )


def render_strength_badge(strength: str) -> str:
    """Return an HTML bracket-style evidence-strength badge."""
    color = STRENGTH_COLORS.get(strength, "#6b7280")
    return (
        f'<span style="'
        f'background:{color}18; color:{color}; '
        f'padding:3px 10px; border-radius:4px; font-weight:600; '
        f'border:1px solid {color}44; font-size:0.75rem; '
        f'letter-spacing:0.06em; font-family:monospace;">'
        f'{strength.upper()}'
        f'</span>'
    )


def render_confidence_bar(confidence: int) -> str:
    """Return an HTML progress bar for a confidence score."""
    color = "#22c55e" if confidence >= 70 else ("#f59e0b" if confidence >= 40 else "#ef4444")
    return (
        f'<div style="background:#e2e8f011; border-radius:3px; height:8px; width:100%; margin-bottom:4px;">'
        f'<div style="width:{confidence}%; background:{color}; height:8px; border-radius:3px;"></div></div>'
        f'<span style="color:#6b7280; font-size:0.78rem;">{confidence}%</span>'
    )


def render_credibility_bar(score: float) -> str:
    """Return an HTML credibility score visual (filled pips out of 5)."""
    filled_color  = "#60a5fa"
    empty_color   = "#1e3a5f"
    pips = ""
    for i in range(1, 6):
        bg = filled_color if i <= round(score) else empty_color
        pips += (
            f'<span style="display:inline-block; width:14px; height:6px; '
            f'background:{bg}; border-radius:2px; margin-right:3px;"></span>'
        )
    return (
        f'<div style="display:flex; align-items:center; gap:6px;">'
        f'<div>{pips}</div>'
        f'<span style="color:#6b7280; font-size:0.78rem;">{score} / 5</span>'
        f'</div>'
    )


def inject_custom_css() -> None:
    """Inject custom CSS for enterprise SaaS aesthetic.

    CSS is loaded from styles.css (same directory as this file) rather than
    embedded as a Python string literal, which prevents Pyrefly's language
    server from mis-parsing CSS tokens as Python expressions.
    """
    css_path = pathlib.Path(__file__).parent / "styles.css"
    try:
        css = css_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Graceful fallback — app still works without custom styles
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header() -> None:
    """Render the application header — clean enterprise typographic style."""
    st.markdown("""
    <div style="padding: 2rem 0 1.5rem 0; border-bottom: 1px solid #21262d; margin-bottom: 1.8rem;">
        <div style="display:flex; align-items:baseline; gap:12px; margin-bottom:0.4rem;">
            <span style="
                font-size: 1.7rem;
                font-weight: 800;
                color: #f0f6fc;
                letter-spacing: -0.03em;
                line-height: 1;
            ">FactVibe</span>
            <span style="
                font-size: 0.72rem;
                font-weight: 600;
                color: #388bfd;
                background: #1f6feb22;
                border: 1px solid #1f6feb44;
                border-radius: 4px;
                padding: 2px 8px;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            ">AI Hallucination Auditor</span>
        </div>
        <p style="
            color: #8b949e;
            font-size: 0.88rem;
            margin: 0;
            line-height: 1.5;
            max-width: 560px;
        ">
            Detect hallucinations and verify factual claims using AI-powered reasoning
            and real-world evidence retrieval.
            Powered by <strong style="color:#c9d1d9;">Groq (Qwen3-32B)</strong>
            and <strong style="color:#c9d1d9;">DuckDuckGo Search</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_score_card(score: float) -> None:
    """Render a professional hallucination risk card — no emojis."""
    color = score_color(score)
    label = score_label(score)

    st.markdown(f"""
    <div style="
        background: #161b22;
        border: 1px solid {color}44;
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 1.5rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 1.5rem 0;
    ">
        <div>
            <div class="fv-section-label">Hallucination Risk Score</div>
            <div style="font-size:2.8rem; font-weight:800; color:{color};
                        line-height:1; letter-spacing:-0.03em;">{score}%</div>
        </div>
        <div style="text-align:right;">
            <div style="
                display:inline-block;
                background:{color}18;
                border:1px solid {color}55;
                border-radius:4px;
                padding:6px 16px;
                font-size:0.78rem;
                font-weight:700;
                color:{color};
                letter-spacing:0.1em;
                font-family:monospace;
            ">RISK LEVEL: {label}</div>
            <div style="color:#6e7681; font-size:0.78rem; margin-top:6px;">
                Based on unverified + contradicted claims
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _section_heading(text: str) -> None:
    """Render a consistent section heading in enterprise style."""
    st.markdown(
        f'<h3 style="color:#c9d1d9; font-weight:600; font-size:1rem; '
        f'letter-spacing:-0.01em; margin:1.5rem 0 0.75rem 0;">{text}</h3>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the FactVibe Streamlit application."""

    st.set_page_config(
        page_title="FactVibe: AI Hallucination Auditor",
        page_icon="https://www.google.com/s2/favicons?domain=groq.com",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    inject_custom_css()
    render_header()

    # ── API Key Guard ──────────────────────────────────────────────────────
    if not GROQ_API_KEY:
        st.error(
            "**GROQ_API_KEY not found.**\n\n"
            "Create a `.env` file in the project root and add:\n"
            "```\nGROQ_API_KEY=your_api_key_here\n```\n\n"
            "Get your free API key at [Groq Console](https://console.groq.com/)."
        )
        st.stop()

    client = get_groq_client()
    if not client:
        st.error("Failed to initialise Groq client. Check your API key and try again.")
        st.stop()

    # ── Input Section ──────────────────────────────────────────────────────
    st.markdown(
        '<p class="fv-section-label">Input</p>'
        '<p style="color:#8b949e; font-size:0.85rem; margin:0 0 0.6rem 0;">'
        'Paste any AI-generated content, article, blog post, or text passage below.</p>',
        unsafe_allow_html=True,
    )

    example_text = (
        "The Eiffel Tower is located in Berlin and was built in 2021 by Gustave "
        "Flaubert. It stands 450 meters tall, making it the tallest structure in Europe. "
        "Napoleon Bonaparte commissioned the tower to celebrate France's victory in "
        "World War II."
    )

    user_input = st.text_area(
        label="Input Text",
        placeholder=example_text,
        height=200,
        label_visibility="collapsed",
        key="input_text",
    )

    # Character counter
    char_count    = len(user_input)
    counter_color = "#22c55e" if char_count >= 50 else "#8b949e"
    st.markdown(
        f'<p style="text-align:right; color:{counter_color}; font-size:0.75rem; '
        f'margin-top:-0.4rem;">{char_count:,} characters</p>',
        unsafe_allow_html=True,
    )

    col_btn, col_hint = st.columns([1, 3])
    with col_btn:
        audit_clicked = st.button("Audit Claims", use_container_width=True)
    with col_hint:
        st.markdown(
            '<p style="color:#6e7681; font-size:0.8rem; padding-top:0.55rem;">'
            'Provide at least 50 characters for best results.</p>',
            unsafe_allow_html=True,
        )

    # ── Audit Pipeline ─────────────────────────────────────────────────────
    if audit_clicked:
        if not user_input.strip():
            st.warning("Please paste some text before auditing.")
            st.stop()
        if len(user_input.strip()) < 20:
            st.warning("Text is too short. Please provide a longer passage to audit.")
            st.stop()

        st.markdown("---")

        # ── Step 1: Claim Extraction ───────────────────────────────────────
        with st.spinner("Extracting factual claims with Groq (Qwen3-32B)…"):
            claims = extract_claims(user_input)

        if not claims:
            st.error(
                "No factual claims could be extracted from the provided text.\n\n"
                "This may happen if:\n"
                "- The text contains only opinions or subjective statements.\n"
                "- Groq rate limit was hit — please wait and try again.\n"
                "- The API key is invalid or quota is exhausted."
            )
            st.stop()

        st.success(f"Extracted {len(claims)} factual claim(s). Searching for evidence…")

        # ── Step 2 & 3: Evidence + Verification ───────────────────────────
        results: list[dict] = []

        progress_bar = st.progress(0, text="Verifying claims…")
        total        = len(claims)

        for i, claim_dict in enumerate(claims):
            claim_text = claim_dict.get("claim", "")
            progress_bar.progress(
                i / total,
                text=f"Verifying claim {i + 1} of {total}…",
            )

            # Search evidence — returns (list[dict], debug_dict)
            evidence, ev_debug = search_evidence(claim_text)

            # Verify claim against structured evidence
            result                    = verify_claim(claim_text, evidence)
            result["claim"]           = claim_text
            result["evidence"]        = evidence
            result["avg_credibility"] = average_credibility(evidence)
            result["_ev_debug"]       = ev_debug     # debug metadata
            results.append(result)

            time.sleep(0.3)

        progress_bar.progress(1.0, text="Verification complete.")
        time.sleep(0.5)
        progress_bar.empty()

        # ── Hallucination Score ────────────────────────────────────────────
        h_score = calculate_hallucination_score(results)
        render_score_card(h_score)

        # ── Summary Metrics  (extended dashboard) ─────────────────────────
        _section_heading("Summary Metrics")

        verified_count     = sum(1 for r in results if r["verdict"] == "Verified")
        partial_count      = sum(1 for r in results if r["verdict"] == "Partially Supported")
        unverified_count   = sum(1 for r in results if r["verdict"] == "Unverified")
        contradicted_count = sum(1 for r in results if r["verdict"] == "Contradicted")

        confidences = [r.get("confidence", 0) for r in results]
        avg_conf    = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

        cred_scores = [r.get("avg_credibility", 0.0) for r in results]
        avg_cred    = round(sum(cred_scores) / len(cred_scores), 1) if cred_scores else 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Claims",    len(claims))
        col2.metric("Verified",        verified_count)
        col3.metric("Partially Supported", partial_count)
        col4.metric("Contradicted",    contradicted_count)

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Unverified",      unverified_count)
        col6.metric("Avg. Confidence", f"{avg_conf}%")
        col7.metric("Avg. Credibility",f"{avg_cred} / 5")
        col8.metric("Hallucination Risk", f"{h_score}%")

        st.markdown("---")

        # ── Claim Analysis Section  (enhanced display) ────────────────────
        _section_heading("Claim Analysis")

        for idx, result in enumerate(results, 1):
            verdict           = result.get("verdict",          "Unverified")
            confidence        = result.get("confidence",       0)
            evidence_strength = result.get("evidence_strength","Insufficient")
            reason            = result.get("reason",           "N/A")
            claim_text        = result.get("claim",            "N/A")
            evidence_list     = result.get("evidence",         [])
            cred_score        = result.get("avg_credibility",  0.0)
            border_color      = VERDICT_COLORS.get(verdict, "#30363d")

            with st.expander(
                f"Claim {idx}  —  {claim_text[:85]}{'...' if len(claim_text) > 85 else ''}",
                expanded=(idx == 1),
            ):
                # ── Claim text ─────────────────────────────────────────────
                st.markdown(
                    f'<p style="color:#c9d1d9; font-size:0.95rem; font-weight:500; '
                    f'border-left:3px solid {border_color}; padding-left:0.8rem; '
                    f'margin-bottom:1rem; line-height:1.5;">{claim_text}</p>',
                    unsafe_allow_html=True,
                )

                # ── Row 1: Verdict | Confidence | Evidence Strength | Credibility ──
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.markdown(
                        '<p class="fv-section-label">Verdict</p>'
                        f'{render_verdict_badge(verdict)}',
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        '<p class="fv-section-label">Confidence</p>'
                        f'{render_confidence_bar(confidence)}',
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        '<p class="fv-section-label">Evidence Strength</p>'
                        f'{render_strength_badge(evidence_strength)}',
                        unsafe_allow_html=True,
                    )
                with c4:
                    st.markdown(
                        '<p class="fv-section-label">Source Credibility</p>'
                        f'{render_credibility_bar(cred_score)}',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Analysis / Reason ──────────────────────────────────────
                st.markdown(
                    f'<div style="background:#161b22; border:1px solid #21262d; '
                    f'border-radius:6px; padding:0.75rem 1rem; margin-bottom:1rem;">'
                    f'<p class="fv-section-label" style="margin:0 0 4px 0;">Analysis</p>'
                    f'<span style="color:#c9d1d9; font-size:0.88rem; '
                    f'line-height:1.5;">{reason}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── Evidence Sources ───────────────────────────────────────
                st.markdown(
                    '<p class="fv-section-label">Evidence Sources</p>',
                    unsafe_allow_html=True,
                )

                if not evidence_list or (
                    len(evidence_list) == 1
                    and ("No reliable" in evidence_list[0].get("snippet", "")
                         or "retrieval error" in evidence_list[0].get("snippet", "").lower())
                ):
                    st.markdown(
                        f'<p style="color:#8b949e; font-size:0.85rem;">'
                        f'{evidence_list[0].get("snippet", "No evidence retrieved.")}</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    for s_idx, src in enumerate(evidence_list, 1):
                        title       = src.get("title",   "Untitled")
                        snippet     = src.get("snippet", "No snippet available.")
                        url         = src.get("url",     "")
                        rel_score   = src.get("_score",  None)
                        cred        = calculate_source_credibility(url)
                        cred_color  = (
                            "#22c55e" if cred >= 4
                            else ("#f59e0b" if cred >= 3
                            else "#6b7280")
                        )
                        url_display = url[:75] + "..." if len(url) > 75 else url
                        rel_badge   = (
                            f'<span style="background:#38383e; color:#8b949e; '
                            f'border:1px solid #444; border-radius:3px; '
                            f'padding:1px 6px; font-size:0.68rem; font-weight:600; '
                            f'font-family:monospace; margin-left:6px;">'
                            f'REL {rel_score:.0%}</span>'
                            if rel_score is not None else ""
                        )

                        st.markdown(
                            f'<div style="background:#0d1117; border:1px solid #21262d; '
                            f'border-radius:6px; padding:0.7rem 1rem; '
                            f'margin-bottom:0.5rem;">'
                            f'<div style="display:flex; justify-content:space-between; '
                            f'align-items:flex-start; margin-bottom:4px;">'
                            f'<span style="color:#c9d1d9; font-size:0.85rem; '
                            f'font-weight:600;">[{s_idx}] {title}{rel_badge}</span>'
                            f'<span style="background:{cred_color}18; color:{cred_color}; '
                            f'border:1px solid {cred_color}44; border-radius:3px; '
                            f'padding:1px 7px; font-size:0.72rem; font-weight:700; '
                            f'white-space:nowrap; font-family:monospace;">CRED {cred:.0f}/5</span>'
                            f'</div>'
                            f'<p style="color:#8b949e; font-size:0.83rem; '
                            f'margin:0 0 6px 0; line-height:1.45;">{snippet}</p>'
                            f'<a href="{url}" target="_blank" style="color:#58a6ff; '
                            f'font-size:0.78rem; text-decoration:none; '
                            f'font-family:monospace;">{url_display}</a>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # ── Debug Panel (visible when Debug Mode is ON) ────────────
                ev_debug = result.get("_ev_debug", {})
                if ev_debug and st.session_state.get("debug_mode", False):
                    with st.expander("Debug: Evidence Retrieval", expanded=False):
                        st.markdown(
                            f'<p style="color:#6e7681; font-size:0.78rem; margin:0 0 6px 0;">'
                            f'Search query sent to DuckDuckGo: '
                            f'<code style="color:#58a6ff;">{ev_debug.get("query", "n/a")}</code> '
                            f'&nbsp;|&nbsp; Raw results: <strong>{ev_debug.get("raw_count", 0)}</strong> '
                            f'&nbsp;|&nbsp; After filtering: <strong>{ev_debug.get("filtered_count", 0)}</strong>'
                            f'</p>',
                            unsafe_allow_html=True,
                        )
                        scored_rows = ev_debug.get("scored", [])
                        if scored_rows:
                            rows_html = "".join(
                                f'<tr>'
                                f'<td style="padding:3px 8px; color:{"#22c55e" if s >= _MIN_RELEVANCE else "#6b7280"}; '
                                f'font-family:monospace; font-size:0.75rem;">{s:.0%}</td>'
                                f'<td style="padding:3px 8px; color:#8b949e; font-size:0.75rem;">{t}</td>'
                                f'<td style="padding:3px 8px; color:#58a6ff; font-size:0.72rem; '
                                f'font-family:monospace;">{u}</td>'
                                f'</tr>'
                                for s, t, u in scored_rows
                            )
                            st.markdown(
                                f'<table style="width:100%; border-collapse:collapse; margin-top:6px;">'
                                f'<thead><tr>'
                                f'<th style="padding:3px 8px; color:#6e7681; font-size:0.7rem; '
                                f'text-align:left; font-weight:700;">RELEVANCE</th>'
                                f'<th style="padding:3px 8px; color:#6e7681; font-size:0.7rem; '
                                f'text-align:left; font-weight:700;">TITLE</th>'
                                f'<th style="padding:3px 8px; color:#6e7681; font-size:0.7rem; '
                                f'text-align:left; font-weight:700;">URL</th>'
                                f'</tr></thead><tbody>{rows_html}</tbody></table>',
                                unsafe_allow_html=True,
                            )

        st.markdown("---")

        # ── PDF Export ─────────────────────────────────────────────────────
        _section_heading("Export Report")

        try:
            pdf_bytes     = generate_pdf_report(user_input, claims, results, h_score)
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Report (PDF)",
                data=pdf_bytes,
                file_name=f"factvibe_audit_{timestamp_str}.pdf",
                mime="application/pdf",
                use_container_width=False,
            )
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}\n{traceback.format_exc()}")

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        # Debug Mode toggle ────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.65rem; font-weight:700; color:#6e7681; '
            'text-transform:uppercase; letter-spacing:0.12em; '
            'margin:0.5rem 0 0.4rem 0;">Developer Options</p>',
            unsafe_allow_html=True,
        )
        debug_on = st.toggle(
            "Debug Mode",
            value=st.session_state.get("debug_mode", False),
            key="debug_mode",
            help="Show search queries, relevance scores, and raw result counts inside each claim expander.",
        )
        if debug_on:
            st.markdown(
                f'<p style="color:#6e7681; font-size:0.75rem; margin:0 0 0.8rem 0;">'
                f'Minimum relevance threshold: <code>{_MIN_RELEVANCE:.0%}</code></p>',
                unsafe_allow_html=True,
            )
        st.markdown("<hr style='border-color:#21262d; margin:0.8rem 0;'>", unsafe_allow_html=True)

        st.markdown("""
        <div style="padding:0.5rem 0;">
            <p style="font-size:0.65rem; font-weight:700; color:#6e7681;
                      text-transform:uppercase; letter-spacing:0.12em;
                      margin-bottom:0.8rem;">About FactVibe</p>
            <p style="color:#8b949e; font-size:0.83rem; line-height:1.7; margin:0;">
                FactVibe uses a 3-step AI pipeline:<br><br>
                <strong style="color:#c9d1d9;">1. Claim Extraction</strong><br>
                Groq (Qwen3-32B) identifies verifiable facts.<br><br>
                <strong style="color:#c9d1d9;">2. Evidence Retrieval</strong><br>
                DuckDuckGo fetches real-world sources.<br><br>
                <strong style="color:#c9d1d9;">3. Verification</strong><br>
                Groq cross-references claims vs evidence only — no internal knowledge used.
            </p>
            <hr style="border-color:#21262d; margin:1rem 0;">
            <p style="font-size:0.65rem; font-weight:700; color:#6e7681;
                      text-transform:uppercase; letter-spacing:0.12em;
                      margin-bottom:0.6rem;">Verdict Scale</p>
            <table style="font-size:0.8rem; color:#8b949e; border-collapse:collapse; width:100%;">
                <tr><td style="padding:3px 0;">
                    <span style="background:#22c55e18; color:#22c55e;
                    border:1px solid #22c55e44; border-radius:3px;
                    padding:1px 6px; font-family:monospace;
                    font-size:0.7rem; font-weight:700;">VERIFIED</span>
                </td><td style="padding:3px 0 3px 8px; color:#6e7681;">Evidence confirms</td></tr>
                <tr><td style="padding:3px 0;">
                    <span style="background:#f59e0b18; color:#f59e0b;
                    border:1px solid #f59e0b44; border-radius:3px;
                    padding:1px 6px; font-family:monospace;
                    font-size:0.7rem; font-weight:700;">PARTIAL</span>
                </td><td style="padding:3px 0 3px 8px; color:#6e7681;">Inconclusive</td></tr>
                <tr><td style="padding:3px 0;">
                    <span style="background:#eab30818; color:#eab308;
                    border:1px solid #eab30844; border-radius:3px;
                    padding:1px 6px; font-family:monospace;
                    font-size:0.7rem; font-weight:700;">UNVERIFIED</span>
                </td><td style="padding:3px 0 3px 8px; color:#6e7681;">No evidence</td></tr>
                <tr><td style="padding:3px 0;">
                    <span style="background:#ef444418; color:#ef4444;
                    border:1px solid #ef444444; border-radius:3px;
                    padding:1px 6px; font-family:monospace;
                    font-size:0.7rem; font-weight:700;">CONTRADICTED</span>
                </td><td style="padding:3px 0 3px 8px; color:#6e7681;">Evidence refutes</td></tr>
            </table>
            <hr style="border-color:#21262d; margin:1rem 0;">
            <p style="font-size:0.65rem; font-weight:700; color:#6e7681;
                      text-transform:uppercase; letter-spacing:0.12em;
                      margin-bottom:0.4rem;">Credibility Scale</p>
            <p style="font-size:0.78rem; color:#6e7681; line-height:1.6; margin:0;">
                5 — Government / Academic / Research<br>
                4 — Major news organisations<br>
                3 — Wikipedia<br>
                2 — Blogs / content platforms<br>
                1 — Unknown domains
            </p>
            <hr style="border-color:#21262d; margin:1rem 0;">
            <p style="color:#6e7681; font-size:0.75rem; line-height:1.5; margin:0;">
                FactVibe is an AI-assisted tool. Always verify critical claims
                with authoritative primary sources.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
