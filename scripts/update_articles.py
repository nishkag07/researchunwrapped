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
        "simpleTitle",
        "blurb",
        "curiosityHook",
        "summary",
        "researchQuestion",
        "whatTheyDid",
        "whatTheyFound",
        "mainResult",
        "whatThisCouldLeadTo",
        "deepDive",
        "whyItMatters",
        "keyTakeaways",
        "limitations",
        "scientificTerms",
        "difficulty",
        "readTime",
        "imageSearchQuery",
        "imageAltText",
        "imageCaption"
    ],
    "properties": {
        "simpleTitle": {
            "type": "string",
            "description": "A short, clear title under 10 words. Easy for a general reader to understand. Do not copy the academic paper title unless it is already simple."
        },
        "blurb": {
            "type": "string",
            "description": "One very simple card sentence. Sixth-grade reading level. No technical phrases unless unavoidable."
        },
        "curiosityHook": {
            "type": "string",
            "description": "One engaging opening sentence that explains why a normal reader should care."
        },
        "summary": {
            "type": "string",
            "description": "A very simple 3-4 sentence summary for non-scientists. Sixth-grade reading level."
        },
        "researchQuestion": {
            "type": "string",
            "description": "One or two simple sentences stating the exact question this paper was trying to answer. Be specific to the article."
        },
        "whatTheyDid": {
            "type": "string",
            "description": "A simple but specific explanation of what the researchers used, tested, built, measured, compared, reviewed, or analyzed. Avoid jargon and do not copy the abstract."
        },
        "whatTheyFound": {
            "type": "string",
            "description": "The actual result or conclusion of the paper in very plain English. Must be specific to the article."
        },
        "mainResult": {
            "type": "string",
            "description": "One short sentence naming the clearest result, claim, or takeaway from the paper."
        },
        "whatThisCouldLeadTo": {
            "type": "string",
            "description": "A simple, specific explanation of what the result could lead to if future research works, without overclaiming."
        },
        "deepDive": {
            "type": "string",
            "description": "A simple explanation in 2-3 short paragraphs. Avoid jargon. Explain only the big idea, not every technical detail."
        },
        "whyItMatters": {
            "type": "string",
            "description": "One clear sentence explaining the general implication: why this matters for people, health, medicine, technology, environment, or science."
        },
        "keyTakeaways": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {"type": "string"}
        },
        "limitations": {
            "type": "string",
            "description": "What the study does not prove yet. Keep this careful and honest."
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
                    "explanation": {
                        "type": "string",
                        "description": "Simple definition for someone new to science."
                    }
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
        },
        "imageSearchQuery": {
            "type": "string",
            "description": "A specific image search phrase that visually matches the article topic."
        },
        "imageAltText": {
            "type": "string",
            "description": "Accessible alt text for the article image."
        },
        "imageCaption": {
            "type": "string",
            "description": "One short caption explaining how the image connects to the research."
        }
    }
}

# OpenAI strict JSON schemas require every property to be listed in required.
# This also prevents errors if we add or remove fields later.
ARTICLE_SCHEMA["required"] = list(ARTICLE_SCHEMA["properties"].keys())


SYSTEM_PROMPT = """
You are the editorial engine for Research Unwrapped.

Your job is to turn real research papers into simple, specific, interesting explanations for normal readers.

This is NOT a press release.
This is NOT a motivational science blog.
This is NOT PubMed copied into easier words.

Every answer must stay focused on the exact paper provided.
The reader should understand:
- what the paper was asking
- what the researchers actually used or did
- what they found, built, reviewed, tested, or argued
- what the result could lead to in the real world
- what the paper still does not prove

Style:
- ScienceDaily-like: simple, clear, specific, and curiosity-building.
- Use short sentences.
- Use plain words.
- Explain hard words immediately.
- Write for a smart student or curious parent with no research background.
- Keep it easy, but do not make it empty.
- Never write generic filler.
"""

HEADLINE_RULES = """
HEADLINE RULES:

Keep the headline style the user likes:
"Light-Based 3D Printing Could Bring Lab-Grown Organs Closer"

The headline should combine:
1. the actual thing studied or built,
2. a cautious verb like could, may, might, or suggests,
3. the specific result or real-world possibility.

It must be catchy, but still clearly connected to THIS paper.

Good:
- Light-Based 3D Printing Could Bring Lab-Grown Organs Closer
- AI Read Rat Brain Signals to Track Walking Speed
- Tiny Cancer Models Could Show How Tumors Change Nearby Cells
- Gut-Friendly Diets May Be Linked to Lower Blood Pressure
- Hot Homes May Raise Heat Risk for Older Adults

Bad:
- Brain signals could reveal more than we thought
- New biomedical engineering research could point to bigger possibilities
- Digital light processing bioprinting
- Gut microbiome and hypertension
- AI-assisted diagnosis

Rules:
- 8 to 14 words is ideal.
- 16 words maximum.
- Avoid vague words like "things," "ideas," "possibilities," or "problems" when a specific noun is available.
- Do not copy the academic title.
- Do not overclaim. Use "could," "may," or "might" when the research is early.
"""

SPECIFICITY_RULES = """
SPECIFICITY RULES — MOST IMPORTANT:

The user wants simple writing that still explains the exact study.
Do not drift into only future implications.
Do not summarize the whole field.
Do not write generic sentences that could fit any paper.

Before writing, silently extract these facts from the abstract:
1. Paper type: experiment, clinical trial, animal study, cell study, AI/data study, review, protocol, framework, commentary, etc.
2. Research question: what exact question or problem did the paper focus on?
3. Materials/data: what did they use? Examples: rats, EEG recordings, survey responses, patient records, cells, printed gels, previous studies, AI models.
4. Method: what did they do with those materials/data? Examples: trained an AI model, compared groups, reviewed papers, printed scaffolds, measured indoor heat.
5. Result: what did they find, show, build, argue, or propose?
6. Specific implication: if this work keeps improving, what could it specifically help with?
7. Limit: what does it NOT prove yet?

Writing must be simple but useful.
The reader should never think: "This tells me nothing."

BANNED FILLER PHRASES:
- This article looks at a recent idea...
- This study explores...
- This research points to a bigger question...
- This could help readers understand...
- Scientists are testing new ways to understand or solve a real problem.
- The study addresses a recent question in science or medicine.
- Readers should use the source link to go deeper.
- Offers potential to restore function.
- Could reveal more than we thought.
- Objective.
- Background.

Do not copy the abstract's technical wording.
Translate it.

Examples of translation:
- "Digital light processing bioprinting" → "a 3D printing method that uses light to shape soft, cell-friendly gels"
- "bioink" → "a gel-like material that can carry living cells"
- "bioactuation" → "the ability of engineered muscle to move or contract"
- "EEG" → "electrical brain signals recorded from the head or skull"
- "recurrent neural network" → "an AI model that learns patterns over time"
- "cross-sectional" → "a snapshot study taken at one point in time"
- "randomized clinical trial" → "a human study where people were placed into treatment groups by chance"

Length targets:
- blurb: 2 sentences, 45–75 words total.
- curiosityHook: 1 sentence, 15–30 words.
- summary: 7–10 short sentences, 160–240 words total.
- researchQuestion: 1–2 simple sentences.
- whatTheyDid: 90–150 words.
- whatTheyFound: 100–180 words.
- mainResult: 1 simple sentence.
- whatThisCouldLeadTo: 90–150 words.
- deepDive: 180–280 words.
- whyItMatters: 2–3 specific sentences.
- limitations: 2–4 specific sentences.

FIELD STYLE:

blurb:
Keep it hooky and specific. Include the study type or method and the main result/claim.
Do NOT use vague implications alone.
A good blurb sounds like ScienceDaily, but simpler.

summary:
This should act like a plain-English abstract.
Go straight into the paper. No intro filler.
Use this structure:
- What problem the paper focuses on.
- What kind of paper/study it is.
- What researchers used or looked at.
- What they did.
- What they found or argued.
- Why the result matters.
- What is still early or uncertain.

whatTheyDid:
Explain the materials and method. Be concrete.
Examples: how many people, what species, what dataset, what technology, what comparison, what trial phase, what type of model, what review scope.
If the abstract does not give a number, do not invent one.

whatTheyFound:
Explain the result, not just the goal.
If the paper is a review, explain the main pattern or argument it found across prior work.
If the paper is a protocol, explain what the planned study will examine, and say results are not available yet.
If the paper is a framework/commentary, explain the specific framework or argument.

whatThisCouldLeadTo:
Be specific about the future possibility.
Bad: "This could help medicine and technology."
Better: "If light-based printing and bioinks keep improving, researchers could make small tissue models that are better for testing drugs or studying how damaged tissue heals. That is different from printing a transplant-ready organ, which is still much harder."

limitations:
Be specific but simple.
Say exactly why the reader should be cautious.
Examples: animal-only, cell-only, review-only, observational, small sample, early-stage engineering, protocol without results, correlation not causation.
"""

USER_PROMPT_TEMPLATE = """
Convert this research article metadata into a Research Unwrapped article card and read-more explanation.

{headline_rules}

{specificity_rules}

ARTICLE METADATA:
Original title: {title}
Abstract: {abstract}
Journal: {journal}
Publication date: {date}
Authors: {authors}
DOI: {doi}
PMID: {pmid}
Source URL: {source_url}
Suggested category: {category}

Return valid JSON that follows the schema exactly.

Output must feel like this:
Simple enough for a normal person, but specific enough that it clearly belongs to this exact research paper.

Required writing approach:
- Lead with what happened in the paper.
- Explain what they used and how they did it.
- Explain what they found.
- Then explain what it could lead to, carefully.
- Keep the original paper link for technical details; your job is to make the paper understandable before they click it.

Do not include generic filler.
Do not write only broad implications.
Do not copy the abstract.
"""

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




def make_simple_fallback_title(title: str, category: str = "Science") -> str:
    """Create an implication-driven fallback title if OpenAI is unavailable."""
    lower = clean_text(title).lower()
    category_clean = clean_text(category) or "Science"

    if "bioprint" in lower or "biofabrication" in lower or "tissue engineering" in lower:
        return "Light-Based 3D Printing Could Bring Lab-Grown Organs Closer"
    if "bioink" in lower or "hydrogel" in lower:
        return "New Bioinks Could Help Scientists Build Living Tissues"
    if "microbiome" in lower and ("hypertension" in lower or "blood pressure" in lower):
        return "Gut-Friendly Diets May Be Linked to Lower Blood Pressure"
    if "microbiome" in lower:
        return "Tiny Gut Microbes Could Shape Bigger Health Questions"
    if "brain-computer" in lower or "prosthetic" in lower:
        return "Brain Signals Could Help Future Prosthetics Move More Naturally"
    if "brain" in lower or "neural" in lower or "eeg" in lower or "meg" in lower:
        return "Brain Signals Could Reveal More Than We Thought"
    if "cancer" in lower or "tumor" in lower or "oncology" in lower:
        return "Tiny Cancer Models Could Reveal How Tumors Spread"
    if "crispr" in lower or "gene editing" in lower:
        return "Gene Editing Could Open New Paths for Treatment"
    if "gene" in lower or "genomic" in lower or "genetic" in lower:
        return "DNA Clues Could Help Explain Hard-to-Treat Conditions"
    if "climate" in lower or "ozone" in lower or "pollution" in lower or "air quality" in lower:
        return "Cleaner Air Plans May Need to Target Pollution Earlier"
    if "microplastic" in lower:
        return "Tiny Plastics Could Create Bigger Environmental Risks"
    if "sustainability" in lower or "environment" in lower:
        return "Science Classes Could Help Students Tackle Climate Problems"
    if "artificial intelligence" in lower or "machine learning" in lower or "deep learning" in lower or "ai" in lower:
        return "AI Could Help Scientists Solve Harder Problems"
    if "swallow" in lower or "dysphagia" in lower:
        return "AI Could Help Doctors Catch Swallowing Problems Earlier"
    if "heat" in lower:
        return "Hotter Homes May Put Older Adults at Greater Risk"
    if "diet" in lower or "nutrition" in lower or "obesity" in lower:
        return "Everyday Diet Choices May Shape Long-Term Health Risks"

    return f"New {category_clean} Research Could Point to Bigger Possibilities"



def make_plain_topic(title: str) -> str:
    """Make a rough, simple topic phrase for fallback mode."""
    topic = clean_text(title)
    topic = re.split(r":|—|-", topic)[0].strip()
    topic = re.sub(r"\b(a|an|the)\b", "", topic, flags=re.IGNORECASE)
    topic = re.sub(r"\s+", " ", topic).strip()
    return topic[:90] or "this research"


def fallback_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    title = raw["title"]
    abstract = raw["abstract"]
    category = raw.get("category", "Science")
    score = raw.get("difficultyScore", 0)
    difficulty = "Advanced" if score >= 3 else "Intermediate" if score >= 1 else "Beginner"
    simple_title = make_simple_fallback_title(title, category)
    topic = make_plain_topic(title)
    category_lower = str(category).lower()

    return {
        "simpleTitle": simple_title,
        "blurb": clean_text(
            f"This research looks at {topic.lower()} and why it could matter beyond the lab."
        ),
        "curiosityHook": "Some scientific ideas start small, but they can point toward much bigger possibilities.",
        "summary": clean_text(
            f"This article is about {topic.lower()}. "
            f"The big idea is connected to {category_lower} and why it may matter in the real world. "
            "The original paper has the full scientific detail, but the main point is that researchers are exploring a possible new path forward."
        ),
        "whatTheyFound": clean_text(
            f"The paper focuses on {topic.lower()}. It suggests this area may help scientists ask better questions or build better tools. "
            "This does not mean the idea is ready for everyday use yet."
        ),
        "deepDive": clean_text(
            "Think of this as an early step in a larger scientific story. Researchers are trying to understand whether this idea can help solve a real problem. "
            "The technical paper gives the detailed methods and evidence, while this summary keeps the main idea simple."
        ),
        "whyItMatters": "This matters because early research can help scientists move closer to future tools, treatments, or better decisions.",
        "keyTakeaways": [
            "The study connects to a real-world science problem.",
            "The idea is promising, but it is not a finished solution yet.",
            "Readers can use the original source for the full technical details."
        ],
        "limitations": "This is a simplified summary based on the abstract. The original paper should be checked for exact methods, evidence, and limits.",
        "scientificTerms": fallback_terms(title, abstract),
        "difficulty": difficulty,
        "readTime": 3 if difficulty != "Advanced" else 4,
        "imageSearchQuery": "",
        "imageAltText": "",
        "imageCaption": "",
    }


BAD_GENERATION_PHRASES = [
    "this article looks at a recent idea",
    "this study explores",
    "this research points to a bigger question",
    "could help readers understand",
    "testing new ways to understand or solve a real problem",
    "study addresses a recent question",
    "readers should use the source link",
    "here is why this research is worth a closer look",
    "objective.",
    "background.",
    "offers potential to restore function",
    "could reveal more than we thought",
    "bigger possibilities",
]


def validate_summary_quality(summary: Dict[str, Any], raw: Dict[str, Any]) -> List[str]:
    """Return a list of quality problems. Empty list means acceptable."""
    problems: List[str] = []
    combined = " ".join(str(summary.get(k, "")) for k in [
        "simpleTitle", "blurb", "curiosityHook", "summary", "researchQuestion", "whatTheyDid",
        "whatTheyFound", "mainResult", "whatThisCouldLeadTo", "deepDive", "whyItMatters", "limitations"
    ]).lower()

    for phrase in BAD_GENERATION_PHRASES:
        if phrase in combined:
            problems.append(f"Generic/filler phrase found: {phrase}")

    if len(clean_text(summary.get("blurb"))) < 60:
        problems.append("Blurb is too short to explain the actual study.")
    if len(clean_text(summary.get("summary"))) < 450:
        problems.append("Summary is too short or generic; it needs real study details.")
    if len(clean_text(summary.get("whatTheyDid"))) < 350:
        problems.append("whatTheyDid is too short; it needs concrete method/material details from the abstract.")
    if len(clean_text(summary.get("whatTheyFound"))) < 400:
        problems.append("whatTheyFound is too short; it needs concrete results from the abstract.")
    if len(clean_text(summary.get("whatThisCouldLeadTo"))) < 300:
        problems.append("whatThisCouldLeadTo is too short or generic; it needs a specific future use/implication.")
    if len(clean_text(summary.get("deepDive"))) < 650:
        problems.append("deepDive is too short; it needs a simple but useful explanation.")

    title_words = set(re.findall(r"[a-z]{4,}", raw.get("title", "").lower()))
    abstract_words = set(re.findall(r"[a-z]{4,}", raw.get("abstract", "").lower()))
    source_words = title_words | abstract_words
    output_words = set(re.findall(r"[a-z]{4,}", combined))
    anchor_words = source_words & output_words

    # This is a rough guard against purely generic summaries.
    if len(anchor_words) < 10:
        problems.append("Output does not appear anchored enough to the article abstract.")

    if clean_text(summary.get("simpleTitle", "")).lower() in [
        "brain signals could reveal more than we thought",
        "new science research could point to bigger possibilities",
    ]:
        problems.append("Title is too generic.")

    return problems


def summarize_with_openai(raw: Dict[str, Any]) -> Dict[str, Any]:
    # Quality rule: do not silently publish generic fallback text.
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it in GitHub Settings → Secrets and variables → Actions. "
            "This script requires OpenAI so Research Unwrapped does not publish generic fallback summaries."
        )
    if OpenAI is None:
        raise RuntimeError(
            "The openai Python package is not installed. Make sure requirements.txt includes openai."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)
    base_prompt = USER_PROMPT_TEMPLATE.format(
        headline_rules=HEADLINE_RULES,
        specificity_rules=SPECIFICITY_RULES,
        title=raw["title"],
        abstract=raw["abstract"][:4500],
        journal=raw.get("journal", ""),
        date=raw.get("dateISO", ""),
        authors=", ".join(raw.get("authors", [])[:6]),
        doi=raw.get("doi", ""),
        pmid=raw.get("pmid", ""),
        source_url=raw.get("sourceUrl", ""),
        category=raw["category"],
    )

    last_problems: List[str] = []
    for attempt in range(1, 4):
        quality_feedback = ""
        if last_problems:
            quality_feedback = (
                "\n\nQUALITY FIX REQUIRED FROM PREVIOUS ATTEMPT:\n- "
                + "\n- ".join(last_problems)
                + "\nRewrite the JSON so it is specific to the paper, simple, and useful."
            )

        try:
            response = client.responses.create(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT + "\n\nReturn valid JSON that follows the schema exactly."},
                    {"role": "user", "content": base_prompt + quality_feedback},
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
            summary = json.loads(response.output_text)
            problems = validate_summary_quality(summary, raw)
            if not problems:
                return summary
            last_problems = problems
            log(f"Quality retry {attempt}/3 for {raw.get('id')}: {'; '.join(problems)}")
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI summary failed for {raw.get('id')}: {exc}. "
                "Stopping instead of publishing generic fallback summaries."
            ) from exc

    raise RuntimeError(
        f"Generated summary for {raw.get('id')} stayed too generic after 3 attempts: "
        + "; ".join(last_problems)
    )

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
        "title": clean_text(summary.get("simpleTitle")) or raw["title"],
        "originalTitle": raw["title"],
        "category": raw["category"],
        "date": raw["dateISO"],
        "dateISO": raw["dateISO"],
        "displayDate": raw["displayDate"],
        "readTime": summary.get("readTime", 4),
        "difficulty": summary.get("difficulty", "Intermediate"),
        "image": raw["image"],
        "imageSearchQuery": clean_text(summary.get("imageSearchQuery")),
        "imageAltText": clean_text(summary.get("imageAltText")),
        "imageCaption": clean_text(summary.get("imageCaption")),
        "blurb": clean_text(summary.get("blurb")),
        "excerpt": clean_text(summary.get("blurb")),
        "curiosityHook": clean_text(summary.get("curiosityHook")),
        "summary": clean_text(summary.get("summary")),
        "researchQuestion": clean_text(summary.get("researchQuestion")),
        "whatTheyDid": clean_text(summary.get("whatTheyDid")),
        "whatTheyFound": clean_text(summary.get("whatTheyFound")),
        "mainResult": clean_text(summary.get("mainResult")),
        "whatThisCouldLeadTo": clean_text(summary.get("whatThisCouldLeadTo")),
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

    # Final safety dedupe before writing, in case the same paper appears in multiple categories.
    final_articles = []
    final_seen = set()
    for article in articles:
        key = (article.get("doi") or article.get("pmid") or article.get("sourceUrl") or article.get("originalTitle") or article.get("title") or "").lower().strip()
        key = re.sub(r"\s+", " ", key)
        if key and key not in final_seen:
            final_seen.add(key)
            final_articles.append(article)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(final_articles, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"Wrote {len(final_articles)} articles to {OUTPUT_FILE}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
