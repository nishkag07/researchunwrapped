#!/usr/bin/env python3
"""
Research Unwrapped automation

What this does:
1. Pulls recent life-science articles from Europe PMC.
2. Chooses a balanced mix of topics and difficulty levels.
3. Uses OpenAI to create student-friendly summaries, deeper explanations, and scientific term definitions.
4. Writes data/articles.json so index.html can update automatically on GitHub Pages.

Required GitHub secret for best results:
- OPENAI_API_KEY

Optional variables:
- OPENAI_MODEL, default: gpt-4.1-mini
- ARTICLE_COUNT, default: 50
- RECENT_DAYS, default: 60
- OUTPUT_FILE, default: data/articles.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - allows fallback mode if openai is not installed locally
    OpenAI = None

EPMC_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

ARTICLE_COUNT = int(os.getenv("ARTICLE_COUNT", "50"))
RECENT_DAYS = int(os.getenv("RECENT_DAYS", "60"))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "data/articles.json"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Keep categories aligned with index.html filters.
CATEGORY_QUERIES = [
    {
        "category": "Neuroscience",
        "image": "RUimages/neuro.jpg",
        "query": '(neuroscience OR brain OR neural OR cognition OR Alzheimer* OR Parkinson* OR "brain-computer interface" OR neuroimaging)',
    },
    {
        "category": "Biomedical Engineering",
        "image": "RUimages/prosthetic.jpg",
        "query": '("biomedical engineering" OR biomaterial* OR biosensor* OR prosthetic* OR "tissue engineering" OR "drug delivery" OR biofabrication OR hydrogel*)',
    },
    {
        "category": "AI & Computer Science",
        "image": "RUimages/AI.jpg",
        "query": '("artificial intelligence" OR "machine learning" OR "deep learning" OR bioinformatics OR "large language model" OR "foundation model")',
    },
    {
        "category": "Medicine",
        "image": "RUimages/prosthetic.jpg",
        "query": '(medicine OR therapeutic* OR diagnostic* OR oncology OR immunotherapy OR "clinical trial" OR vaccine* OR biomarker*)',
    },
    {
        "category": "Genetics",
        "image": "RUimages/genetics.jpg",
        "query": '(genetic* OR genomic* OR CRISPR OR epigenetic* OR transcriptomic* OR "gene editing" OR "single-cell")',
    },
    {
        "category": "Women’s Health",
        "image": "RUimages/prosthetic.jpg",
        "query": '("women health" OR pregnancy OR maternal OR menopause OR endometriosis OR "breast cancer" OR ovarian OR reproductive)',
    },
    {
        "category": "Public Health",
        "image": "RUimages/genetics.jpg",
        "query": '("public health" OR epidemiology OR population OR infectious OR prevention OR health equity OR "global health")',
    },
    {
        "category": "Nutrition",
        "image": "RUimages/genetics.jpg",
        "query": '(nutrition OR diet OR microbiome OR metabolism OR obesity OR "dietary" OR nutrient*)',
    },
    {
        "category": "Environment",
        "image": "RUimages/genetics.jpg",
        "query": '(environment* OR climate OR pollution OR microplastic* OR sustainability OR toxicology OR "environmental health")',
    },
]

ADVANCED_MARKERS = [
    "single-cell", "transcriptomic", "proteomic", "metabolomic", "epigen", "CRISPR",
    "genome-wide", "pathway", "mechanism", "molecular", "nanoparticle", "hydrogel",
    "immunotherapy", "checkpoint", "randomized", "phase 2", "phase II", "phase 3",
    "algorithm", "transformer", "foundation model", "deep learning", "causal", "bayesian",
    "organoid", "spatial", "multi-omics", "receptor", "assay", "biomarker",
]

ARTICLE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "blurb", "summary", "deepDive", "whyItMatters", "keyTakeaways",
        "limitations", "scientificTerms", "difficulty", "readTime"
    ],
    "properties": {
        "blurb": {
            "type": "string",
            "description": "One catchy but accurate sentence for the card. No hype."
        },
        "summary": {
            "type": "string",
            "description": "A student-friendly 2-4 sentence summary."
        },
        "deepDive": {
            "type": "string",
            "description": "A more difficult explanation that teaches the actual science in 2-3 short paragraphs."
        },
        "whyItMatters": {
            "type": "string",
            "description": "Why a curious student or reader should care."
        },
        "keyTakeaways": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {"type": "string"}
        },
        "limitations": {
            "type": "string",
            "description": "Cautious note explaining what the article does not prove yet."
        },
        "scientificTerms": {
            "type": "array",
            "minItems": 3,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["term", "explanation"],
                "properties": {
                    "term": {"type": "string"},
                    "explanation": {"type": "string"}
                }
            }
        },
        "difficulty": {
            "type": "string",
            "enum": ["Beginner", "Intermediate", "Advanced"]
        },
        "readTime": {
            "type": "integer",
            "minimum": 2,
            "maximum": 8
        }
    }
}


def log(message: str) -> None:
    print(f"[ResearchUnwrapped] {message}", flush=True)


def clean_text(value: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")[:70] or "article"


def parse_date(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return datetime.now(timezone.utc).date().isoformat()

    # Europe PMC often returns YYYY-MM-DD, but keep this forgiving.
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.date().isoformat()
        except ValueError:
            continue

    match = re.search(r"(20\d{2}|19\d{2})", raw)
    if match:
        return f"{match.group(1)}-01-01"

    return datetime.now(timezone.utc).date().isoformat()


def display_date(date_iso: str) -> str:
    try:
        parsed = datetime.strptime(date_iso, "%Y-%m-%d")
        return parsed.strftime("%b %-d, %Y")
    except Exception:
        try:
            parsed = datetime.strptime(date_iso, "%Y-%m-%d")
            return parsed.strftime("%b %#d, %Y")
        except Exception:
            return date_iso


def split_authors(author_string: str) -> List[str]:
    if not author_string:
        return []
    pieces = [clean_text(piece) for piece in re.split(r",\s*", author_string) if clean_text(piece)]
    return pieces[:8]


def europe_pmc_url(source: str, record_id: str) -> str:
    if source and record_id:
        return f"https://europepmc.org/article/{source}/{record_id}"
    return "https://europepmc.org/"


def source_url(record: Dict[str, Any]) -> str:
    pmid = clean_text(record.get("pmid"))
    doi = clean_text(record.get("doi"))
    source = clean_text(record.get("source"))
    record_id = clean_text(record.get("id"))

    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if doi:
        return f"https://doi.org/{doi}"
    return europe_pmc_url(source, record_id)


def source_name(record: Dict[str, Any]) -> str:
    if clean_text(record.get("pmid")):
        return "PubMed"
    if clean_text(record.get("doi")):
        return "DOI"
    return "Europe PMC"


def difficulty_score(title: str, abstract: str) -> int:
    text = f"{title} {abstract}".lower()
    score = sum(1 for marker in ADVANCED_MARKERS if marker.lower() in text)
    # Dense abstracts often indicate a more technical article.
    if len(abstract) > 1800:
        score += 1
    if len(re.findall(r"\b[A-Z]{2,}\b", title + " " + abstract)) >= 3:
        score += 1
    return score


def fallback_terms(title: str, abstract: str) -> List[Dict[str, str]]:
    candidates = []
    lower = f"{title} {abstract}".lower()
    term_bank = {
        "biomarker": "A measurable signal in the body that can help indicate a disease, risk, or treatment response.",
        "genomic": "Related to the full set of DNA instructions inside cells.",
        "transcriptomic": "Related to which genes are being actively read and turned into RNA inside cells.",
        "clinical trial": "A carefully designed study that tests a medical treatment or intervention in people.",
        "machine learning": "A type of AI where computers learn patterns from data instead of being explicitly programmed for every rule.",
        "neural": "Related to nerves, neurons, or the brain’s communication system.",
        "inflammation": "The immune system’s response to injury, infection, or stress, which can help healing but also cause harm if uncontrolled.",
        "microbiome": "The community of bacteria and other microbes living in or on the body or environment.",
        "CRISPR": "A gene-editing tool that can make targeted changes to DNA.",
        "epigenetic": "Changes in gene activity that do not alter the DNA sequence itself.",
        "randomized": "A study design where participants are assigned by chance to different groups to reduce bias.",
        "pathway": "A chain of molecular events inside cells that produces a biological effect.",
    }
    for term, explanation in term_bank.items():
        if term in lower:
            candidates.append({"term": term, "explanation": explanation})
    while len(candidates) < 3:
        fallback = [
            {"term": "abstract", "explanation": "A short summary of a scientific article’s purpose, methods, results, and conclusions."},
            {"term": "peer review", "explanation": "A quality-control process where experts evaluate a study before publication."},
            {"term": "correlation", "explanation": "A relationship between two things, which does not always mean one caused the other."},
        ]
        candidates.append(fallback[len(candidates) % len(fallback)])
    return candidates[:6]


def fallback_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    title = raw["title"]
    abstract = raw["abstract"]
    score = raw.get("difficultyScore", 0)
    difficulty = "Advanced" if score >= 3 else "Intermediate" if score >= 1 else "Beginner"
    sentence = re.split(r"(?<=[.!?])\s+", abstract)[0] if abstract else "This article reports a recent research finding."
    return {
        "blurb": sentence[:220].rstrip() + ("…" if len(sentence) > 220 else ""),
        "summary": f"This recent study explores {title.lower()}. The automated summary should be reviewed against the original article before publication.",
        "deepDive": abstract[:1200].rstrip() + ("…" if len(abstract) > 1200 else ""),
        "whyItMatters": "This article was selected because it connects to an active research area and can help readers learn how scientists investigate real problems.",
        "keyTakeaways": [
            "The study addresses a recent question in science or medicine.",
            "The original abstract gives the most precise technical details.",
            "Readers should use the source link to go deeper and verify the findings."
        ],
        "limitations": "This simplified summary is generated from metadata and the abstract. Read the original source before making scientific or medical claims.",
        "scientificTerms": fallback_terms(title, abstract),
        "difficulty": difficulty,
        "readTime": 4 if difficulty == "Advanced" else 3,
    }


def summarize_with_openai(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENAI_API_KEY or OpenAI is None:
        return fallback_summary(raw)

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
You are writing for Research Unwrapped, a student-friendly research discovery website.

Audience:
- Curious high school and college students.
- They want simple summaries, but they also want real science and difficult terms explained.

Rules:
- Use ONLY the article metadata and abstract below.
- Do not exaggerate. Do not imply medical advice.
- Mention uncertainty and limitations clearly.
- Make the blurb engaging but accurate.
- The deepDive should teach the science behind the paper, not just repeat the abstract.
- Explain 3-6 scientific terms that appear in or are directly relevant to the article.

Article metadata:
Title: {raw['title']}
Category: {raw['category']}
Journal: {raw.get('journal', '')}
Authors: {', '.join(raw.get('authors', [])[:6])}
Date: {raw.get('dateISO', '')}
DOI: {raw.get('doi', '')}
PMID: {raw.get('pmid', '')}
Source URL: {raw.get('sourceUrl', '')}

Abstract:
{raw['abstract'][:4500]}
""".strip()

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": "Return valid JSON that follows the schema exactly."},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "research_unwrapped_article",
                    "schema": ARTICLE_SCHEMA,
                    "strict": True,
                }
            },
        )
        return json.loads(response.output_text)
    except Exception as exc:
        log(f"OpenAI summary failed for {raw.get('id')}: {exc}. Using fallback summary.")
        return fallback_summary(raw)


def build_query(base_query: str, start_date: str, end_date: str) -> str:
    return f"({base_query}) AND HAS_ABSTRACT:Y AND FIRST_PDATE:[{start_date} TO {end_date}]"


def fetch_category(category_config: Dict[str, str], page_size: int, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    query = build_query(category_config["query"], start_date, end_date)
    params = {
        "query": query,
        "format": "json",
        "pageSize": str(page_size),
        "resultType": "core",
    }

    try:
        response = requests.get(EPMC_ENDPOINT, params=params, timeout=30)
        if response.status_code >= 400:
            # Retry without the date filter, because API query syntax can be strict.
            log(f"Date-filtered query failed for {category_config['category']} ({response.status_code}); retrying without date filter.")
            params["query"] = f"({category_config['query']}) AND HAS_ABSTRACT:Y"
            response = requests.get(EPMC_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        log(f"Europe PMC fetch failed for {category_config['category']}: {exc}")
        return []

    results = data.get("resultList", {}).get("result", []) or []
    normalized = []

    for record in results:
        title = clean_text(record.get("title"))
        abstract = clean_text(record.get("abstractText"))
        if len(title) < 20 or len(abstract) < 450:
            continue

        date_iso = parse_date(record.get("firstPublicationDate") or record.get("firstIndexDate") or record.get("pubYear"))
        journal = clean_text(record.get("journalTitle") or record.get("bookOrReportDetails"))
        authors = split_authors(clean_text(record.get("authorString")))
        doi = clean_text(record.get("doi"))
        pmid = clean_text(record.get("pmid"))
        record_id = clean_text(record.get("id")) or hashlib.sha1(title.encode("utf-8")).hexdigest()[:12]

        raw = {
            "rawId": record_id,
            "id": f"{category_config['category'].lower().replace(' ', '-')}-{record_id}",
            "title": title,
            "abstract": abstract,
            "category": category_config["category"],
            "image": category_config["image"],
            "dateISO": date_iso,
            "displayDate": display_date(date_iso),
            "journal": journal,
            "authors": authors,
            "doi": doi,
            "pmid": pmid,
            "sourceName": source_name(record),
            "sourceUrl": source_url(record),
            "difficultyScore": difficulty_score(title, abstract),
        }
        normalized.append(raw)

    log(f"Fetched {len(normalized)} usable articles for {category_config['category']}.")
    return normalized


def dedupe_articles(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for item in candidates:
        key = (item.get("doi") or item.get("pmid") or item.get("title") or "").lower()
        key = re.sub(r"\s+", " ", key).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def date_sort_key(item: Dict[str, Any]) -> datetime:
    try:
        return datetime.strptime(item.get("dateISO", "1900-01-01"), "%Y-%m-%d")
    except Exception:
        return datetime(1900, 1, 1)


def choose_balanced_articles(candidates: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    candidates = sorted(candidates, key=lambda x: (date_sort_key(x), x.get("difficultyScore", 0)), reverse=True)

    # Ensure some advanced articles appear so users can learn challenging science.
    advanced_target = max(8, math.ceil(count * 0.30)) if count >= 20 else max(2, math.ceil(count * 0.25))
    advanced = [item for item in candidates if item.get("difficultyScore", 0) >= 2]
    selected = advanced[:advanced_target]
    selected_keys = {(item.get("doi") or item.get("pmid") or item.get("title") or "").lower() for item in selected}

    # Round-robin by category for diversity.
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        key = (item.get("doi") or item.get("pmid") or item.get("title") or "").lower()
        if key not in selected_keys:
            by_category[item["category"]].append(item)

    categories = [config["category"] for config in CATEGORY_QUERIES]
    while len(selected) < count and any(by_category.values()):
        for category in categories:
            if by_category[category] and len(selected) < count:
                selected.append(by_category[category].pop(0))

    return sorted(selected[:count], key=date_sort_key, reverse=True)


def assemble_article(raw: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    safe_id_base = raw.get("pmid") or raw.get("doi") or raw.get("rawId") or raw["title"]
    article_id = f"{slugify(raw['category'])}-{hashlib.sha1(str(safe_id_base).encode('utf-8')).hexdigest()[:10]}"

    return {
        "id": article_id,
        "title": raw["title"],
        "category": raw["category"],
        "date": raw["dateISO"],
        "dateISO": raw["dateISO"],
        "displayDate": raw["displayDate"],
        "readTime": summary.get("readTime", 4),
        "difficulty": summary.get("difficulty", "Intermediate"),
        "image": raw["image"],
        "blurb": clean_text(summary.get("blurb")),
        "excerpt": clean_text(summary.get("blurb")),
        "summary": clean_text(summary.get("summary")),
        "deepDive": clean_text(summary.get("deepDive")),
        "body": clean_text(summary.get("deepDive")),
        "whyItMatters": clean_text(summary.get("whyItMatters")),
        "keyTakeaways": [clean_text(x) for x in summary.get("keyTakeaways", [])][:5],
        "limitations": clean_text(summary.get("limitations")),
        "scientificTerms": [
            {
                "term": clean_text(term.get("term")),
                "explanation": clean_text(term.get("explanation")),
            }
            for term in summary.get("scientificTerms", [])[:6]
            if clean_text(term.get("term")) and clean_text(term.get("explanation"))
        ],
        "sourceName": raw.get("sourceName") or "Europe PMC",
        "journal": raw.get("journal") or "",
        "authors": raw.get("authors") or [],
        "sourceUrl": raw.get("sourceUrl") or "",
        "doi": raw.get("doi") or "",
        "pmid": raw.get("pmid") or "",
    }


def main() -> int:
    random.seed(42)
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=RECENT_DAYS)

    per_category = max(25, math.ceil(ARTICLE_COUNT / len(CATEGORY_QUERIES) * 5))
    all_candidates: List[Dict[str, Any]] = []

    log(f"Searching Europe PMC from {start_date} to {end_date} for {ARTICLE_COUNT} articles.")
    for config in CATEGORY_QUERIES:
        all_candidates.extend(fetch_category(config, per_category, start_date.isoformat(), end_date.isoformat()))
        time.sleep(0.35)

    all_candidates = dedupe_articles(all_candidates)
    log(f"{len(all_candidates)} unique usable candidates after deduping.")

    if not all_candidates:
        log("No candidates found. Nothing written.")
        return 1

    selected = choose_balanced_articles(all_candidates, ARTICLE_COUNT)
    log(f"Selected {len(selected)} articles for summarization.")

    articles = []
    for index, raw in enumerate(selected, start=1):
        log(f"Summarizing {index}/{len(selected)}: {raw['title'][:90]}")
        summary = summarize_with_openai(raw)
        article = assemble_article(raw, summary)
        articles.append(article)
        # Gentle pacing for APIs.
        time.sleep(0.4 if OPENAI_API_KEY else 0.05)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(articles, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"Wrote {len(articles)} articles to {OUTPUT_FILE}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
