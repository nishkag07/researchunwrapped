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
        "whatTheyFound",
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
            "description": "Two short sentences maximum for the card. Plain English. Explain what the study found."
        },
        "curiosityHook": {
            "type": "string",
            "description": "One engaging opening sentence that explains why a normal reader should care."
        },
        "summary": {
            "type": "string",
            "description": "A friendly 3-4 sentence summary for students and general readers."
        },
        "whatTheyFound": {
            "type": "string",
            "description": "The main finding in plain English. Mention if it was a human, animal, cell, data, review, or framework study when clear."
        },
        "deepDive": {
            "type": "string",
            "description": "A more detailed explanation that teaches the real science in 2-3 short paragraphs. Explain hard concepts clearly."
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


SYSTEM_PROMPT = """
You are the editorial engine for Research Unwrapped, a science-literacy platform for high school and college students, curious non-experts, and everyday readers.

Your job is NOT to rewrite papers in academic language.
Your job is to translate credible research into clear, engaging, accurate explanations that make people want to learn more.

Core editorial philosophy:
- Lead with the discovery, surprise, or real-world implication.
- Do not lead with the paper's academic framing.
- Make the card title understandable to a smart teenager.
- Preserve scientific accuracy.
- Never exaggerate beyond the evidence.
- Never imply human clinical impact if the study was only in cells, animals, simulations, data, or a small early trial.
- Keep the original scientific title separately for citation and transparency.

Audience:
- Smart high school students
- College applicants interested in STEM
- Curious non-scientists
- Patients and families who want plain-English research explanations
- Students looking for inspiration for independent research

Tone:
- Clear
- Curious
- Conversational
- Accessible
- Slightly exciting, but never clickbait
- Never childish
- Never overly technical on the card
"""

HEADLINE_RULES = """
HEADLINE RULES — VERY IMPORTANT:

The public article title should sound like this example:
"Light-Based 3D Printing Could Bring Lab-Grown Organs Closer"

That means the title should usually combine:
1. A simple version of the technology, discovery, or finding
2. A cautious verb such as could, may, might, or suggests
3. The larger real-world implication or future possibility

The title should make a curious student think:
"Wait, how does that work?"

Do NOT write a topic label.
Do NOT write a shortened academic title.
Do NOT copy the paper title unless it is already clear and exciting.
Do NOT use vague titles like:
- Digital light processing bioprinting
- Gut microbiome and hypertension
- AI-assisted diagnosis
- Climate and ozone pollution
- Tumor microenvironment interactions

Better title examples:
- Light-Based 3D Printing Could Bring Lab-Grown Organs Closer
- Gut-Friendly Diets May Be Linked to Lower Blood Pressure
- AI Could Help Doctors Catch Swallowing Problems Earlier
- Brain Signals Could Help Future Prosthetics Move More Naturally
- Tiny Cancer Models Could Reveal How Tumors Spread
- Cleaner Air Plans May Need to Target Pollution Earlier
- Gene Editing Could Help Scientists Understand Ovarian Disorders
- Hotter Homes May Put Older Adults at Greater Risk
- AI Models Could Work Better When They Team Up
- Microbiology Classes Could Help Train Future Climate Scientists

Rules:
- 8 to 14 words is ideal.
- 16 words maximum.
- Use simple, vivid words.
- Include the implication, not just the topic.
- Use “could,” “may,” or “might” for early-stage research.
- Do not say “will,” “proves,” “cures,” or “breakthrough” unless the abstract clearly supports it.
- Avoid acronyms in the title unless they are widely known, like AI or DNA.
- Avoid technical phrases like “spatial transcriptomics,” “randomized clinical trial,” “bioink innovations,” or “differential expression” in the title.
- The original academic title will be saved separately as originalTitle, so the public title should be reader-friendly.

Before writing the title, silently ask:
- What is the actual thing being studied?
- What could this eventually help with?
- Why would a non-scientist care?
- How can I say that in one clear headline?

The title should focus on the implication, not the method name.
"""


USER_PROMPT_TEMPLATE = """
Convert the following research article metadata into a Research Unwrapped article card and full-read explanation.

{headline_rules}

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

Return a valid JSON object that follows the required schema.

WRITING RULES:

1. simpleTitle:
Use the HEADLINE RULES exactly.
Create a captivating but accurate reader-facing title.
It must explain the implication or why the research matters.
It should make a student want to click.
It should NOT sound like PubMed, a journal article, or a topic label.

Do NOT copy the original paper title unless it is already simple, clear, and exciting.
Do NOT use academic title phrases such as:
- "A conceptual framework"
- "A systematic review"
- "An investigation of"
- "Associations between"
- "Characterization of"
- "Mechanisms underlying"
- "Integrating X into Y"
- "Evaluation of"
- "Assessment of"
- "Protocol"
- "Scoping review protocol"

Instead, translate the title into the real-world finding, question, or implication.

The title should usually follow this formula:
Simple technology/discovery + cautious verb + real-world future implication

Strong examples:
- Light-Based 3D Printing Could Bring Lab-Grown Organs Closer
- Gut-Friendly Diets May Be Linked to Lower Blood Pressure
- AI Could Help Doctors Catch Swallowing Problems Earlier
- Brain Signals Could Help Future Prosthetics Move More Naturally
- Tiny Cancer Models Could Reveal How Tumors Spread
- Hotter Homes May Put Older Adults at Greater Risk

Bad examples:
- Digital light processing bioprinting
- Gut microbiome and hypertension
- AI-assisted diagnosis
- Tumor microenvironment interactions
- Heat exhaustion in older adults

2. blurb:
Write 1-2 sentences for the article card.
It should explain the finding in simple language.
No jargon unless immediately understandable.
Keep it under 45 words.
Make it interesting enough that someone wants to click.
Do not begin every blurb with "Researchers found."

3. curiosityHook:
Write one opening sentence for the full article.
Start with why the reader should care.
Use a question, contrast, everyday scenario, or surprising implication.
Do not start with "This study explores..."

4. summary:
Write 3-4 sentences that tell the reader the story of the study in simple language:
- What problem is being studied?
- What did the scientists do or argue?
- What did they find or propose?
- Why is it interesting?

5. whatTheyFound:
Explain the main finding in plain English.
Use 2-4 short paragraphs.
Mention whether the study was in humans, animals, cells, data, AI models, a clinical trial, a review, or a framework paper.
Be clear about what was actually shown.

6. whyItMatters:
Explain the real-world implication.
Connect to health, learning, technology, environment, society, or future research.
Do not overclaim.

7. deepDive:
Explain the science behind the finding in accessible language.
This is where harder science can appear, but explain it as you go.
Use analogies when helpful.
Assume the reader is smart but new to the field.
Keep paragraphs short.
Do not make it sound like a textbook.

8. scientificTerms:
Pick 3-6 important scientific terms from the article.
For each term, explain it in simple language.
Definitions should be one sentence.
Use terms that help the reader understand the article, not random vocabulary.
Avoid generic terms like "abstract," "peer review," or "correlation" unless directly needed for this article.

9. keyTakeaways:
Write 3 short takeaways.
Each should be useful.
At least one should mention a limitation or uncertainty when appropriate.

10. limitations:
Explain what this study does NOT prove.
Mention sample size, model type, early-stage status, correlation vs causation, review/protocol status, preprint status, or need for further research when relevant.
This must be honest and specific.

11. difficulty:
Choose one:
- "Beginner" if most high school students can understand it easily
- "Intermediate" if it requires some biology/medicine/AI background
- "Advanced" if the topic is technical but still worth explaining

12. Accuracy rules:
- If it is an animal study, do not imply it is proven in humans.
- If it is observational, do not imply causation.
- If it is a review, commentary, protocol, or framework paper, do not describe it as a new experiment.
- If it is a preprint, mention that it has not been peer reviewed.
- Do not invent statistics, claims, authors, institutions, or applications.
- If the abstract does not provide enough detail, say so in limitations.

13. Image guidance:
- Choose an image concept that directly matches the research.
- Do not use generic lab photos unless the article is broadly about lab research.
- If the article is about the brain, use brain/neuron imagery.
- If it is about cancer, use cells, tumors, immune cells, or treatment imagery.
- If it is about genetics, use DNA, gene editing, sequencing, or chromosomes.
- If it is about AI, use data, models, code, or AI-assisted science visuals.
- If it is about environment, use the specific ecosystem, pollutant, organism, or climate issue.
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


def fallback_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    title = raw["title"]
    abstract = raw["abstract"]
    category = raw.get("category", "Science")
    score = raw.get("difficultyScore", 0)
    difficulty = "Advanced" if score >= 3 else "Intermediate" if score >= 1 else "Beginner"
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", abstract) if s]
    first_sentence = sentences[0] if sentences else "This research explores a recent scientific question."
    second_sentence = sentences[1] if len(sentences) > 1 else "The full paper gives the technical details behind the finding."
    simple_title = make_simple_fallback_title(title, category)

    return {
        "simpleTitle": simple_title,
        "blurb": clean_text(f"{first_sentence[:150].rstrip()}{'…' if len(first_sentence) > 150 else ''}"),
        "curiosityHook": "This research points to a bigger question: how could today’s science become tomorrow’s real-world solution?",
        "summary": clean_text(
            f"This article looks at a recent idea in {str(category).lower()} and why it may matter beyond the lab. "
            f"The original study focuses on: {title}. "
            "The details are technical, but the bigger idea is that scientists are testing new ways to understand or solve a real problem."
        ),
        "whatTheyFound": clean_text(f"{first_sentence} {second_sentence}"),
        "deepDive": clean_text(abstract[:1200].rstrip() + ("…" if len(abstract) > 1200 else "")),
        "whyItMatters": "This matters because it could help readers understand how early scientific ideas may eventually shape medicine, technology, health, or the environment.",
        "keyTakeaways": [
            "The study explores a recent scientific question with real-world implications.",
            "The finding is promising, but the original article has the most precise technical details.",
            "More research is needed before broad conclusions can be made."
        ],
        "limitations": "This is an automated simplified summary based on the abstract. Readers should check the original source before making scientific or medical claims.",
        "scientificTerms": fallback_terms(title, abstract),
        "difficulty": difficulty,
        "readTime": 4 if difficulty == "Advanced" else 3,
        "imageSearchQuery": "",
        "imageAltText": "",
        "imageCaption": "",
    }


def summarize_with_openai(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENAI_API_KEY or OpenAI is None:
        return fallback_summary(raw)

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = USER_PROMPT_TEMPLATE.format(
        headline_rules=HEADLINE_RULES,
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

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT + "\n\nReturn valid JSON that follows the schema exactly."},
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
        "whatTheyFound": clean_text(summary.get("whatTheyFound")),
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
