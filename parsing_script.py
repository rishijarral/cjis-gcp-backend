#!/usr/bin/env python3
from __future__ import annotations

import re
import json
import os
import argparse
import logging
from tqdm import tqdm
import unicodedata 
import nltk 
from typing import Any, List, Dict, Tuple, Optional, TYPE_CHECKING, Set, Callable
import uuid 
import random 
import shutil 
import hashlib
from collections import Counter 

# --- PyMuPDF Import ---
try:
    import fitz # PyMuPDF
    PYMUPDF_IMPORTED = True
except ImportError:
    logging.error("PyMuPDF (fitz) not found. Run: pip install PyMuPDF. PDF text extraction will fail.")
    fitz = None
    PYMUPDF_IMPORTED = False

# --- Camelot Import ---
try:
    import camelot
    import pandas as pd # pandas is a dependency of camelot and used for table formatting
    CAMELOT_IMPORTED = True
except ImportError:
    logging.error("Camelot or pandas not found. Run: pip install camelot-py[cv] pandas. PDF table extraction will fail.")
    camelot = None
    pd = None
    CAMELOT_IMPORTED = False

# --- LangChain Import ---
if TYPE_CHECKING:
    from langchain.prompts import PromptTemplate as _LangchainPromptTemplate
else:
    _LangchainPromptTemplate = Any

try:
    from langchain.prompts import PromptTemplate
    LANGCHAIN_IMPORTED = True
except ImportError:
    logging.error("LangChain not found. Run: pip install langchain langchain-core. Prompt generation will be impaired.")
    PromptTemplate = None
    LANGCHAIN_IMPORTED = False


# --- Script Version ---
SCRIPT_VERSION = "2.6.1_user_updates_1" # Updated version

# --- Vertex AI Embedding Model Imports ---
try:
    from google import generativeai as genai
    from google.api_core import exceptions as google_api_exceptions
    GOOGLE_API_EXCEPTIONS_IMPORTED = True
except ImportError:
    logging.error("Google Generative AI library or google-api-core not found. Run: pip install google-generativeai google-api-core")
    genai = None
    google_api_exceptions = None
    GOOGLE_API_EXCEPTIONS_IMPORTED = False


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- NLTK Setup ---
NLTK_DATA_PATHS = [
    os.path.join(os.path.expanduser("~"), "nltk_data"),
    "/usr/share/nltk_data", "/usr/local/share/nltk_data", "/usr/lib/nltk_data", "/usr/local/lib/nltk_data"
]
NLTK_DATA_PATH_SELECTED = None
for path_candidate in NLTK_DATA_PATHS:
    if os.path.exists(os.path.join(path_candidate, 'tokenizers/punkt')) and \
      os.path.exists(os.path.join(path_candidate, 'corpora/stopwords')) and \
      os.path.exists(os.path.join(path_candidate, 'taggers/averaged_perceptron_tagger')):
        NLTK_DATA_PATH_SELECTED = path_candidate
        logger.debug(f"NLTK resources found in selected path: {NLTK_DATA_PATH_SELECTED}") # Changed to debug, info later on success
        break
if NLTK_DATA_PATH_SELECTED is None:
    NLTK_DATA_PATH_SELECTED = NLTK_DATA_PATHS[0] # Default to user's home if not found elsewhere

if not os.path.exists(NLTK_DATA_PATH_SELECTED):
    os.makedirs(NLTK_DATA_PATH_SELECTED, exist_ok=True)
if NLTK_DATA_PATH_SELECTED not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH_SELECTED)

nltk_resources_to_check = {
    "tokenizers/punkt": "punkt",
    "corpora/stopwords": "stopwords",
    "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger"
}
for resource_path, resource_name in nltk_resources_to_check.items():
    try:
        nltk.data.find(resource_path, paths=[NLTK_DATA_PATH_SELECTED] + nltk.data.path)
    except LookupError:
        logger.info(f"NLTK '{resource_name}' resource not found in known paths (best candidate for download: "
                    f"'{NLTK_DATA_PATH_SELECTED}'). Attempting download...")
        try:
            nltk.download(resource_name, download_dir=NLTK_DATA_PATH_SELECTED, quiet=False)
            logger.info(f"Successfully downloaded '{resource_name}' to {NLTK_DATA_PATH_SELECTED}")
        except Exception as e:
            logger.error(f"Failed to download NLTK '{resource_name}' automatically to '{NLTK_DATA_PATH_SELECTED}': {e}. "
                         f"Please try manually: python -m nltk.downloader -d \"{NLTK_DATA_PATH_SELECTED}\" {resource_name}. "
                         f"Relevant functionality (e.g., sentence tokenization, keyword extraction) might be impaired.")

# --- Ghostscript Check for Camelot ---
def check_ghostscript():
    if shutil.which("gs") or shutil.which("gswin64c") or shutil.which("gswin32c"):
        logger.info("Ghostscript executable found. Camelot should function correctly.")
        return True
    else:
        logger.error("Ghostscript NOT FOUND. Camelot table extraction will likely fail. "
                     "Please install Ghostscript and ensure it's in your system PATH. "
                     "See: https://camelot-py.readthedocs.io/en/master/user/install-deps.html#install-deps")
        return False
GHOSTSCRIPT_AVAILABLE = False

# --- Default Configuration ---
DEFAULT_INPUT_FILE = os.getenv("INPUT_PDF_PATH", "./data/pdf/CJIS.pdf")
DEFAULT_OUTPUT_JSONL_FINETUNE_FILE = os.getenv("FT_JSONL_PATH", "./data/output_jsonl/cjis_finetune_dataset_conversational.jsonl")
DEFAULT_OUTPUT_JSONL_RAG_FILE = os.getenv("RAG_JSONL_PATH", "./data/output_jsonl/cjis_rag_data_with_embeddings.jsonl")
DEFAULT_PAGE_START = int(os.getenv("DEFAULT_PAGE_START", 1))
DEFAULT_PAGE_END = int(os.getenv("DEFAULT_PAGE_END", 9999))

DEFAULT_FT_MAX_WORDS_PER_CHUNK = int(os.getenv("DEFAULT_FT_MAX_WORDS_PER_CHUNK", 400))
DEFAULT_FT_WORD_OVERLAP_FOR_CHUNKS = int(os.getenv("DEFAULT_FT_WORD_OVERLAP_FOR_CHUNKS", 35))

DEFAULT_RAG_MAX_WORDS_PER_CHUNK = int(os.getenv("DEFAULT_RAG_MAX_WORDS_PER_CHUNK", 200))
DEFAULT_RAG_WORD_OVERLAP_FOR_CHUNKS = int(os.getenv("DEFAULT_RAG_WORD_OVERLAP_FOR_CHUNKS", 20))

# --- Vertex AI Embedding Model Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
VERTEX_EMBEDDING_MODEL = os.getenv("VERTEX_EMBEDDING_MODEL", "models/text-embedding-004") # Updated model reference

MIN_SECTION_WORD_COUNT = 15
MIN_RAG_CHUNK_WORD_COUNT = 30 # Ensure this is reasonable for embedding and meaningful content
MIN_FT_CHUNK_WORD_COUNT = 20

NUM_NOT_FOUND_EXAMPLES_PER_CHUNK = 1

TOC_PAGE_REF_PATTERN = re.compile(r'\s+\.{3,}\s*(\d+|[ivxlcdm]+)$', re.IGNORECASE)

HEADING_PATTERNS = [
    re.compile(r'^(?P<title>[A-Z][A-Za-z0-9\s\-\,\/\(\)\&\.]*[A-Za-z0-9\s\)])\s+\((?P<code>[A-Z0-9]{1,6}(?:[\-\.\s][A-Z0-9]{1,6})*)\)$'),
    re.compile(r'^(?P<code>[A-Z_0-9]+(?:[\.\-][A-Z_0-9]+)*(?:\.\d+)?)\s+(?P<title>[A-Z(].*?)$'),
    re.compile(r'^(?P<code>[A-Z_0-9]+(?:[\.\-][A-Z_0-9]+)*(?:\.\d+)?)\s+(?P<title>.+?)$'),
    re.compile(r'^(?P<code>(?:Appendix|Chapter|Section|Attachment)\s+[A-Z0-9]+(?:[\.\-][A-Z0-9]+)*)\s*[:\-–—]?\s*(?P<title>.*?)$', re.IGNORECASE),
    re.compile(r'^(?P<code>(?:Appendix|Chapter|Section|Attachment)\s+[A-Z0-9]+(?:[\.\-][A-Z0-9]+)*)$', re.IGNORECASE),
    re.compile(r'^(?P<title>(?:Introduction|Summary|Conclusion|Overview|References|Foreword|Preface|Acknowledgements|Policy Information|Term Definitions|Criminal Justice Information|Security and Management Control Outsourcing Standard|Table of Contents))(?P<code>\s*)$', re.IGNORECASE),
    re.compile(r'^(?P<title_base>[A-Z_0-9][A-Z_0-9\s\-\,\.\&()]*[A-Z_0-9])(?:[\s\:]+\((?P<code>[A-Z0-9-]+)\))?$', re.IGNORECASE),
    re.compile(r'^(?P<code>[A-Z_0-9]+(?:[\.\-][A-Z_0-9]+)*(?:\.\d+)*)$'),
]

SKIP_SECTION_CODES = ["TOC", "Table of Contents", "TOC_HEADING"]
SKIP_SECTION_TITLE_PATTERNS = [
    re.compile(r"^(list of figures|list of tables|table of contents|index|glossary(\s+of\s+terms)?|acronyms)$", re.IGNORECASE),
]

# Strict code pattern for user request (e.g., AC-2, PM-10)
STRICT_CODE_PATTERN_FOR_MATCH = re.compile(r'^[A-Z]{1,4}(?:-\d{1,3}(?:\([a-z0-9]+\))?)?$') # e.g. AC-2, PM-10, SI-4(2) PM-_10

# --- Fine-tuning Prompt Templates (Strings to be used by LangChain PromptTemplate) ---
CLASSIC_FT_SYSTEM_INSTRUCTION = (
    "You are PolicyBot, an AI assistant expertly trained on the CJIS Security Policy. "
    "You will be provided with a specific section of the CJIS Security Policy. "
    "Your task is to act as if you are explaining this section. When you later receive questions, "
    "you must answer them *strictly* based on the policy text that was originally provided for that section. "
    "Do not invent information or answer based on external knowledge. "
    "If the answer is not in the provided text for a given question, state that clearly."
)
RAG_FT_SYSTEM_INSTRUCTION = (
    "You are PolicyBot, an AI assistant expertly trained on the CJIS Security Policy. "
    "You will be given a Policy Text and a Question. Your task is to answer the question "
    "using *only* the information present in the provided Policy Text for that turn. "
    "Do not invent information or answer based on external knowledge. "
    "If the answer is not in the provided Policy Text for a given question, state that clearly."
)

CLASSIC_FT_USER_PROMPT_EXPLAIN_SINGLE_STR = "Explain the CJIS policy section '{section_code}: {section_title}' in detail."
CLASSIC_FT_USER_PROMPT_EXPLAIN_MULTI_STR = "Explain Part {chunk_num} of {total_chunks} for the CJIS policy section '{section_code}: {section_title}' in detail."
CLASSIC_FT_USER_PROMPT_QA_SINGLE_STR = "What specific information does the CJIS policy section '{section_code}: {section_title}' provide?"
CLASSIC_FT_USER_PROMPT_QA_MULTI_STR = "Regarding Part {chunk_num} of {total_chunks} for CJIS policy section '{section_code}: {section_title}', what specific information does this part provide?"
CLASSIC_FT_USER_PROMPT_NOT_FOUND_STR = (
    "I am looking for information about '{other_section_topic}' (which I believe is addressed in section '{other_section_code}'). "
    "Is this topic covered in the current CJIS policy section '{current_section_code}: {current_section_title}' (Part {chunk_num} of {total_chunks}) that I'm reviewing?"
)

RAG_FT_USER_PROMPT_EXPLAIN_TEMPLATE_STR = (
    "Policy Text: {policy_chunk}\n\n"
    "Question: Based on the provided Policy Text, explain the CJIS policy section '{section_code}: {section_title}' (Part {chunk_num} of {total_chunks})."
)
RAG_FT_USER_PROMPT_QA_TEMPLATE_STR = (
    "Policy Text: {policy_chunk}\n\n"
    "Question: Based on the provided Policy Text, what specific information does this part (Part {chunk_num} of {total_chunks}) of CJIS policy section '{section_code}: {section_title}' provide?"
)
RAG_FT_USER_PROMPT_REQUIREMENTS_TEMPLATE_STR = (
    "Policy Text: {policy_chunk}\n\n"
    "Question: Based on the provided Policy Text, what are the requirements, if any, outlined in this part of the CJIS policy section '{section_code}: {section_title}' (Part {chunk_num} of {total_chunks})?"
)
RAG_FT_USER_PROMPT_HOW_OFTEN_TEMPLATE_STR = (
    "Policy Text: {policy_chunk}\n\n"
    "Question: Based on the provided Policy Text, how often should any specified activities be done, according to this part of CJIS policy section '{section_code}: {section_title}' (Part {chunk_num} of {total_chunks})?"
)
RAG_FT_USER_PROMPT_RESPONSIBILITIES_TEMPLATE_STR = (
    "Policy Text: {policy_chunk}\n\n"
    "Question: Based on the provided Policy Text, list the key responsibilities, if any, outlined in this part of CJIS policy section '{section_code}: {section_title}' (Part {chunk_num} of {total_chunks})."
)
RAG_FT_USER_PROMPT_SUMMARY_TEMPLATE_STR = (
    "Policy Text: {policy_chunk}\n\n"
    "Question: Based on the provided Policy Text, provide a summary of this part of CJIS policy section '{section_code}: {section_title}' (Part {chunk_num} of {total_chunks})."
)
RAG_FT_USER_PROMPT_NOT_FOUND_TEMPLATE_STR = (
    "Policy Text: {policy_chunk}\n\n"
    "Question: I am looking for information about '{other_section_topic}' (which I believe is addressed in section '{other_section_code}'). "
    "Is this topic covered in the provided Policy Text for CJIS policy section '{current_section_code}: {current_section_title}' (Part {chunk_num} of {total_chunks})?"
)

CLASSIC_FT_MODEL_COMPLETION_EXPLAIN_SINGLE_STR = "The CJIS policy section \"{full_section_heading}\" states:\n\n{body_text}"
CLASSIC_FT_MODEL_COMPLETION_EXPLAIN_MULTI_STR = "Part {chunk_num} of {total_chunks} for the CJIS policy section \"{full_section_heading}\" states:\n\n{body_text}"
CLASSIC_FT_MODEL_COMPLETION_QA_SINGLE_STR = "Regarding section '{full_section_heading}', the policy provides the following information:\n\n{body_text}"
CLASSIC_FT_MODEL_COMPLETION_QA_MULTI_STR = "Part {chunk_num} of {total_chunks} of section '{full_section_heading}' provides the following information:\n\n{body_text}"
CLASSIC_FT_MODEL_COMPLETION_NOT_FOUND_STR = (
    "Based on the provided text for CJIS policy section '{current_section_code}: {current_section_title}' (Part {chunk_num} of {total_chunks}), "
    "information regarding '{other_section_topic}' (typically from section '{other_section_code}') is not present in this specific part."
)

RAG_FT_MODEL_COMPLETION_EXPLAIN_TEMPLATE_STR = (
    "Based on the provided Policy Text, to explain section '{full_section_heading}' (Part {chunk_num} of {total_chunks}), "
    "the relevant information is: {policy_chunk}"
)
RAG_FT_MODEL_COMPLETION_QA_TEMPLATE_STR = (
    "Based on the provided Policy Text, the specific information Part {chunk_num} of {total_chunks} of section '{full_section_heading}' provides is: {policy_chunk}"
)
RAG_FT_MODEL_COMPLETION_SUMMARY_TEMPLATE_STR = (
    "Based on the provided Policy Text, to provide a summary of this part of CJIS policy section '{section_code}: {section_title}' "
    "(Part {chunk_num} of {total_chunks}), the core information is: {policy_chunk}"
)
RAG_FT_MODEL_COMPLETION_HOW_OFTEN_KW_FOUND_TEMPLATE_STR = (
    "Based on the provided Policy Text, for Part {chunk_num} of {total_chunks} of section '{full_section_heading}', "
    "frequencies or time-related specifications mentioned include: {policy_chunk}"
)
RAG_FT_MODEL_COMPLETION_HOW_OFTEN_KW_NOT_FOUND_TEMPLATE_STR = (
    "Based on the provided Policy Text for Part {chunk_num} of {total_chunks} of section '{full_section_heading}', "
    "explicit frequency keywords (e.g., 'daily', 'annually', 'periodic') were not readily identified in this specific segment. "
    "The full relevant text is: {policy_chunk}"
)
RAG_FT_MODEL_COMPLETION_REQUIREMENTS_KW_FOUND_TEMPLATE_STR = (
    "Based on the provided Policy Text, specific requirements, mandates, or obligations (often indicated by words like 'shall', 'must', 'is required') "
    "in Part {chunk_num} of {total_chunks} of section '{full_section_heading}' appear to be detailed as follows: {policy_chunk}"
)
RAG_FT_MODEL_COMPLETION_REQUIREMENTS_KW_NOT_FOUND_TEMPLATE_STR = (
    "Based on the provided Policy Text for Part {chunk_num} of {total_chunks} of section '{full_section_heading}', "
    "explicit strong requirement keywords (such as 'shall', 'must') were not prominently identified in this segment. "
    "However, the full text which may contain implicit requirements is: {policy_chunk}"
)
RAG_FT_MODEL_COMPLETION_RESPONSIBILITIES_KW_FOUND_TEMPLATE_STR = (
    "Based on the provided Policy Text, key responsibilities or duties (often indicated by phrases like 'responsible for', 'duty of') "
    "detailed in Part {chunk_num} of {total_chunks} of section '{full_section_heading}' seem to include: {policy_chunk}"
)
RAG_FT_MODEL_COMPLETION_RESPONSIBILITIES_KW_NOT_FOUND_TEMPLATE_STR = (
    "Based on the provided Policy Text for Part {chunk_num} of {total_chunks} of section '{full_section_heading}', "
    "explicit keywords for responsibilities (like 'responsible for') were not prominently identified in this segment. "
    "The full text concerning potential duties or roles is: {policy_chunk}"
)
RAG_FT_MODEL_COMPLETION_NOT_FOUND_TEMPLATE_STR = (
    "Based on the provided Policy Text for CJIS policy section '{current_section_code}: {current_section_title}' (Part {chunk_num} of {total_chunks}), "
    "information regarding '{other_section_topic}' (typically from section '{other_section_code}') is not present in this specific text."
)


# LangChain PromptTemplate objects (to be initialized)
LC_CLASSIC_USER_EXPLAIN_SINGLE: Optional[_LangchainPromptTemplate] = None
LC_CLASSIC_USER_EXPLAIN_MULTI: Optional[_LangchainPromptTemplate] = None
LC_CLASSIC_USER_QA_SINGLE: Optional[_LangchainPromptTemplate] = None
LC_CLASSIC_USER_QA_MULTI: Optional[_LangchainPromptTemplate] = None
LC_CLASSIC_USER_NOT_FOUND: Optional[_LangchainPromptTemplate] = None
LC_CLASSIC_MODEL_EXPLAIN_SINGLE: Optional[_LangchainPromptTemplate] = None
LC_CLASSIC_MODEL_EXPLAIN_MULTI: Optional[_LangchainPromptTemplate] = None
LC_CLASSIC_MODEL_QA_SINGLE: Optional[_LangchainPromptTemplate] = None
LC_CLASSIC_MODEL_QA_MULTI: Optional[_LangchainPromptTemplate] = None
LC_CLASSIC_MODEL_NOT_FOUND: Optional[_LangchainPromptTemplate] = None

LC_RAG_USER_EXPLAIN: Optional[_LangchainPromptTemplate] = None
LC_RAG_USER_QA: Optional[_LangchainPromptTemplate] = None
LC_RAG_USER_REQUIREMENTS: Optional[_LangchainPromptTemplate] = None
LC_RAG_USER_HOW_OFTEN: Optional[_LangchainPromptTemplate] = None
LC_RAG_USER_RESPONSIBILITIES: Optional[_LangchainPromptTemplate] = None
LC_RAG_USER_SUMMARY: Optional[_LangchainPromptTemplate] = None
LC_RAG_USER_NOT_FOUND: Optional[_LangchainPromptTemplate] = None

LC_RAG_MODEL_EXPLAIN: Optional[_LangchainPromptTemplate] = None
LC_RAG_MODEL_QA: Optional[_LangchainPromptTemplate] = None
LC_RAG_MODEL_SUMMARY: Optional[_LangchainPromptTemplate] = None
LC_RAG_MODEL_NOT_FOUND: Optional[_LangchainPromptTemplate] = None

LC_RAG_MODEL_HOW_OFTEN_KW_FOUND: Optional[_LangchainPromptTemplate] = None
LC_RAG_MODEL_HOW_OFTEN_KW_NOT_FOUND: Optional[_LangchainPromptTemplate] = None
LC_RAG_MODEL_REQUIREMENTS_KW_FOUND: Optional[_LangchainPromptTemplate] = None
LC_RAG_MODEL_REQUIREMENTS_KW_NOT_FOUND: Optional[_LangchainPromptTemplate] = None
LC_RAG_MODEL_RESPONSIBILITIES_KW_FOUND: Optional[_LangchainPromptTemplate] = None
LC_RAG_MODEL_RESPONSIBILITIES_KW_NOT_FOUND: Optional[_LangchainPromptTemplate] = None

RAG_FT_PromptConfigEntry = Dict[str, Any]
RAG_FT_PROMPT_CONFIG: List[RAG_FT_PromptConfigEntry] = []


def initialize_langchain_templates():
    global PromptTemplate, LANGCHAIN_IMPORTED
    global LC_CLASSIC_USER_EXPLAIN_SINGLE, LC_CLASSIC_USER_EXPLAIN_MULTI, LC_CLASSIC_USER_QA_SINGLE, LC_CLASSIC_USER_QA_MULTI, LC_CLASSIC_USER_NOT_FOUND
    global LC_CLASSIC_MODEL_EXPLAIN_SINGLE, LC_CLASSIC_MODEL_EXPLAIN_MULTI, LC_CLASSIC_MODEL_QA_SINGLE, LC_CLASSIC_MODEL_QA_MULTI, LC_CLASSIC_MODEL_NOT_FOUND
    global LC_RAG_USER_EXPLAIN, LC_RAG_USER_QA, LC_RAG_USER_REQUIREMENTS, LC_RAG_USER_HOW_OFTEN, LC_RAG_USER_RESPONSIBILITIES, LC_RAG_USER_SUMMARY, LC_RAG_USER_NOT_FOUND
    global LC_RAG_MODEL_EXPLAIN, LC_RAG_MODEL_QA, LC_RAG_MODEL_SUMMARY, LC_RAG_MODEL_NOT_FOUND
    global LC_RAG_MODEL_HOW_OFTEN_KW_FOUND, LC_RAG_MODEL_HOW_OFTEN_KW_NOT_FOUND
    global LC_RAG_MODEL_REQUIREMENTS_KW_FOUND, LC_RAG_MODEL_REQUIREMENTS_KW_NOT_FOUND
    global LC_RAG_MODEL_RESPONSIBILITIES_KW_FOUND, LC_RAG_MODEL_RESPONSIBILITIES_KW_NOT_FOUND
    global RAG_FT_PROMPT_CONFIG

    if not LANGCHAIN_IMPORTED or PromptTemplate is None:
        logger.error("LangChain PromptTemplate not available. Cannot initialize LangChain templates.")
        return False

    try:
        # Classic FT Prompts
        LC_CLASSIC_USER_EXPLAIN_SINGLE = PromptTemplate.from_template(CLASSIC_FT_USER_PROMPT_EXPLAIN_SINGLE_STR)
        LC_CLASSIC_USER_EXPLAIN_MULTI = PromptTemplate.from_template(CLASSIC_FT_USER_PROMPT_EXPLAIN_MULTI_STR)
        LC_CLASSIC_USER_QA_SINGLE = PromptTemplate.from_template(CLASSIC_FT_USER_PROMPT_QA_SINGLE_STR)
        LC_CLASSIC_USER_QA_MULTI = PromptTemplate.from_template(CLASSIC_FT_USER_PROMPT_QA_MULTI_STR)
        LC_CLASSIC_USER_NOT_FOUND = PromptTemplate.from_template(CLASSIC_FT_USER_PROMPT_NOT_FOUND_STR)
        LC_CLASSIC_MODEL_EXPLAIN_SINGLE = PromptTemplate.from_template(CLASSIC_FT_MODEL_COMPLETION_EXPLAIN_SINGLE_STR)
        LC_CLASSIC_MODEL_EXPLAIN_MULTI = PromptTemplate.from_template(CLASSIC_FT_MODEL_COMPLETION_EXPLAIN_MULTI_STR)
        LC_CLASSIC_MODEL_QA_SINGLE = PromptTemplate.from_template(CLASSIC_FT_MODEL_COMPLETION_QA_SINGLE_STR)
        LC_CLASSIC_MODEL_QA_MULTI = PromptTemplate.from_template(CLASSIC_FT_MODEL_COMPLETION_QA_MULTI_STR)
        LC_CLASSIC_MODEL_NOT_FOUND = PromptTemplate.from_template(CLASSIC_FT_MODEL_COMPLETION_NOT_FOUND_STR)

        # RAG-style FT Prompts (User side)
        LC_RAG_USER_EXPLAIN = PromptTemplate.from_template(RAG_FT_USER_PROMPT_EXPLAIN_TEMPLATE_STR)
        LC_RAG_USER_QA = PromptTemplate.from_template(RAG_FT_USER_PROMPT_QA_TEMPLATE_STR)
        LC_RAG_USER_REQUIREMENTS = PromptTemplate.from_template(RAG_FT_USER_PROMPT_REQUIREMENTS_TEMPLATE_STR)
        LC_RAG_USER_HOW_OFTEN = PromptTemplate.from_template(RAG_FT_USER_PROMPT_HOW_OFTEN_TEMPLATE_STR)
        LC_RAG_USER_RESPONSIBILITIES = PromptTemplate.from_template(RAG_FT_USER_PROMPT_RESPONSIBILITIES_TEMPLATE_STR)
        LC_RAG_USER_SUMMARY = PromptTemplate.from_template(RAG_FT_USER_PROMPT_SUMMARY_TEMPLATE_STR)
        LC_RAG_USER_NOT_FOUND = PromptTemplate.from_template(RAG_FT_USER_PROMPT_NOT_FOUND_TEMPLATE_STR)

        # RAG-style FT Prompts (Model side - with nuance)
        LC_RAG_MODEL_EXPLAIN = PromptTemplate.from_template(RAG_FT_MODEL_COMPLETION_EXPLAIN_TEMPLATE_STR)
        LC_RAG_MODEL_QA = PromptTemplate.from_template(RAG_FT_MODEL_COMPLETION_QA_TEMPLATE_STR)
        LC_RAG_MODEL_SUMMARY = PromptTemplate.from_template(RAG_FT_MODEL_COMPLETION_SUMMARY_TEMPLATE_STR)
        LC_RAG_MODEL_NOT_FOUND = PromptTemplate.from_template(RAG_FT_MODEL_COMPLETION_NOT_FOUND_TEMPLATE_STR)

        LC_RAG_MODEL_HOW_OFTEN_KW_FOUND = PromptTemplate.from_template(RAG_FT_MODEL_COMPLETION_HOW_OFTEN_KW_FOUND_TEMPLATE_STR)
        LC_RAG_MODEL_HOW_OFTEN_KW_NOT_FOUND = PromptTemplate.from_template(RAG_FT_MODEL_COMPLETION_HOW_OFTEN_KW_NOT_FOUND_TEMPLATE_STR)
        LC_RAG_MODEL_REQUIREMENTS_KW_FOUND = PromptTemplate.from_template(RAG_FT_MODEL_COMPLETION_REQUIREMENTS_KW_FOUND_TEMPLATE_STR)
        LC_RAG_MODEL_REQUIREMENTS_KW_NOT_FOUND = PromptTemplate.from_template(RAG_FT_MODEL_COMPLETION_REQUIREMENTS_KW_NOT_FOUND_TEMPLATE_STR)
        LC_RAG_MODEL_RESPONSIBILITIES_KW_FOUND = PromptTemplate.from_template(RAG_FT_MODEL_COMPLETION_RESPONSIBILITIES_KW_FOUND_TEMPLATE_STR)
        LC_RAG_MODEL_RESPONSIBILITIES_KW_NOT_FOUND = PromptTemplate.from_template(RAG_FT_MODEL_COMPLETION_RESPONSIBILITIES_KW_NOT_FOUND_TEMPLATE_STR)

        # Populate RAG_FT_PROMPT_CONFIG
        RAG_FT_PROMPT_CONFIG = [
            {
                "user_template": LC_RAG_USER_EXPLAIN,
                "model_template_default": LC_RAG_MODEL_EXPLAIN,
            },
            {
                "user_template": LC_RAG_USER_QA,
                "model_template_default": LC_RAG_MODEL_QA,
            },
            {
                "user_template": LC_RAG_USER_SUMMARY,
                "model_template_default": LC_RAG_MODEL_SUMMARY,
            },
            {
                "user_template": LC_RAG_USER_REQUIREMENTS,
                "model_template_kw_found": LC_RAG_MODEL_REQUIREMENTS_KW_FOUND,
                "model_template_kw_not_found": LC_RAG_MODEL_REQUIREMENTS_KW_NOT_FOUND,
                "keyword_check_fn": check_for_requirement_keywords
            },
            {
                "user_template": LC_RAG_USER_HOW_OFTEN,
                "model_template_kw_found": LC_RAG_MODEL_HOW_OFTEN_KW_FOUND,
                "model_template_kw_not_found": LC_RAG_MODEL_HOW_OFTEN_KW_NOT_FOUND,
                "keyword_check_fn": check_for_frequency_keywords
            },
            {
                "user_template": LC_RAG_USER_RESPONSIBILITIES,
                "model_template_kw_found": LC_RAG_MODEL_RESPONSIBILITIES_KW_FOUND,
                "model_template_kw_not_found": LC_RAG_MODEL_RESPONSIBILITIES_KW_NOT_FOUND,
                "keyword_check_fn": check_for_responsibility_keywords
            }
        ]
        logger.info(f"Successfully initialized LangChain PromptTemplates and RAG FT Config (version: {SCRIPT_VERSION}).")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize LangChain PromptTemplates: {e}")
        return False

# --- Utility Helpers (Text Cleaning, Heading Matching, etc.) ----------------
def escape_braces(text: str) -> str:
    return text.replace('{', '{{').replace('}', '}}')

def dehyphenate(text: str) -> str:
    return re.sub(r'(\w)-[\r\n]+(\w)', r'\1\2', text)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = dehyphenate(text)
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r'(?<=[a-zA-Z0-9,;:])\n(?=[a-z])', ' ', text)
    text = re.sub(r'\bCJI\b(?![-\w])', 'CJIS', text)

    # --- START: Updated/Added Regex for Garbage Tokens ---
    text = re.sub(r'\b(?:[0-9A-F]{2}F){2,}[0-9A-F]{2}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:[0-9A-F]F){3,}[0-9A-F]?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:[0-9A-F]{2}F){2,}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:[0-9A-F]{2}F\s+){2,}(?:[0-9A-F]{2}F)?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:\d+F){2,}\d*\b', '', text, flags=re.IGNORECASE) # User request: e.g. 12F34F56 or 12F34F (ends with F)
    text = re.sub(r'\b([^a-zA-Z0-9\s]{1,3})\1{2,}\b', '', text)
    text = re.sub(r'\b(([a-zA-Z0-9]{1,2})[^a-zA-Z0-9\s]?)\1{2,}\b', '', text)
    text = re.sub(r'\b([A-Za-z0-9])-\1-\1\b', '', text)
    text = re.sub(r'\b([A-Za-z0-9])\s\1\s\1\b', '', text)
    # --- END: Updated/Added Regex for Garbage Tokens ---

    text = re.sub(r'\s*\[\s*(Figure|Table|Appendix|Section|Chapter)\s+[A-Z0-9\.\-]+(:[^\]]+)?\s*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Figure|Table|Index|Appendix|Section|Chapter)\s+[A-Z0-9\.\-]+\b:?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*(?:\d{1,2}/\d{1,2}/\d{2,4}\s*)?\[\s*(?:START|END) OF - .*?\]\s*[\r\n]*', '', text, flags=re.IGNORECASE | re.MULTILINE)

    text = re.sub(r'^(?:Page\s+\w?[\-\–\—]?\d+|\w?[\-\–\—]?\d+\s+Page)\b', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\bPage\s+\d+(?:\s*of\s*\d+)?\b', '', text, flags=re.IGNORECASE)

    text = re.sub(r'UNCLASSIFIED//FOR OFFICIAL USE ONLY', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:CJIS\s*Security\s*Policy|CJISSECPOL)\s*(?:Version|v)?\s*\d+\.\d+(?:\.\d+)?\b', '', text, flags=re.IGNORECASE)

    def replace_refer_to(match):
        keyword = match.group(1)
        doc_type = match.group(2)
        if doc_type:
            return f"{keyword} {doc_type.strip()} [details unspecified or follow elsewhere]."
        else:
            return f"{keyword} [details unspecified or follow elsewhere]."

    text = re.sub(
        r'\b(Refer to|See|Consult)\s+'
        r'((?:Section|Appendix|Chapter|Figure|Table|Page|Diagram|Document|Policy|Standard|Guideline|Below|Above|Attached|Enclosed|Following|Preceding|the\s+\w+)\b)?'
        r'(?!\s+[A-Za-z0-9](?:[\w\-\.]*[A-Za-z0-9])?)'
        r'\s*([:.]|\.{3,})?\s*$',
        replace_refer_to,
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )
    text = re.sub(
        r'\b(Refer to|See also|Consult)\s*(?!\s+[A-Za-z0-9](?:[\w\-\.]*[A-Za-z0-9])?)\s*(\.{3,}|[\s\.]*)$',
        r'\1 [details unspecified].',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    bullet_pattern = r'^\s*(?:[a-zA-Z0-9]{1,4}\.(?!\s*\.)\s*|[ivxlcdm]+\.(?!\s*\.)\s*|[•*-]\s*|➢\s*|❖\s*|▪\s*|(?:\([a-zA-Z0-9]{1,2}\)|[a-zA-Z0-9]{1,2}\))\s*)(?=\S)'
    text = re.sub(bullet_pattern, '• ', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*\|.*\|\s*$', '', text, flags=re.MULTILINE) # Remove likely table remnants
    text = re.sub(r'^\s*[\[\]()|*\-#@<>^~_]{1,3}\s*$', '', text, flags=re.MULTILINE) # Remove lines with only these symbols

    # Footnote marker line removal (e.g., "1.", "1)", "*", if they are the entire line)
    text = re.sub(r'^\s*(?:\d{1,2}[\.\)]?|[\*†‡§#AaBbCc]\.?)\s*$', '', text, flags=re.MULTILINE)


    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE) # Remove lines that are only numbers (like page numbers or remnants)

    return text.strip()


def is_footer_or_header_line(line_text: str, page_number_str: str, pdf_filename_stem: str,
                             line_y0: Optional[float] = None, line_y1: Optional[float] = None, page_height: Optional[float] = None,
                             header_margin: float = 35, footer_margin: float = 35) -> bool:
    line_lower = line_text.lower().strip()
    if not line_lower:
        return False

    if page_height and line_y0 is not None and line_y1 is not None:
        if line_y1 < header_margin:
            logger.debug(f"Positional Header Skip (y1 {line_y1:.2f} < margin {header_margin}): {line_text[:50]}")
            return True
        if line_y0 > (page_height - footer_margin):
            logger.debug(f"Positional Footer Skip (y0 {line_y0:.2f} > margin {page_height - footer_margin:.2f}): {line_text[:50]}")
            return True

    is_short_line = len(line_text) < 70

    if re.fullmatch(r'\d{1,4}', line_lower) and is_short_line: return True
    if page_number_str in line_text and len(line_text) < len(page_number_str) + 25 and '%' not in line_text : return True
    if re.search(r'(page\s+\w?[\-\–\—]?\d+|\w?[\-\–\—]?\d+\s+page)', line_lower): return True
    if re.fullmatch(r'page\s+\d+\s+of\s+\d+', line_lower, re.IGNORECASE): return True

    date_pattern = r'(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})'
    if re.fullmatch(date_pattern, line_lower) and is_short_line: return True
    if re.search(r'^(?:page\s+)?\d{1,3}\s+' + date_pattern + r'$', line_lower) and is_short_line: return True

    if (pdf_filename_stem.lower() in line_lower or "cjis security policy" in line_lower or
        ("policy" in line_lower and "version" in line_lower)) and len(line_text) < 80:
        return True
    if "version" in line_lower and re.search(r'\d+(\.\d+)+', line_lower) and len(line_text) < 50 : return True
    if "appendix" in line_lower and page_number_str in line_lower and len(line_text) < 40: return True
    if re.fullmatch(r'unclassified//for official use only(?: // policysensitive)?', line_lower, re.IGNORECASE): return True

    if (any(kw in line_lower for kw in ["table of contents", "summary", "introduction", "chapter", "appendix", "section"]) and \
        (TOC_PAGE_REF_PATTERN.search(line_lower) or \
         (re.search(r'\d+$', line_lower) or re.search(r'^[ivxlcdm]+$', line_lower.split()[-1] if line_lower.split() else ""))) and \
         is_short_line):
        logger.debug(f"Suspected ToC-like header/footer line: {line_text[:60]}")
        return True

    return False

def match_heading(line: str) -> Optional[Tuple[str, str]]:
    cleaned_line = TOC_PAGE_REF_PATTERN.sub('', line).strip()
    if '|' in cleaned_line and len(cleaned_line.split('|')) > 2 :
        logger.debug(f"Skipping potential heading match due to multiple pipe characters (likely table row): '{cleaned_line[:80]}...'")
        return None
    if not cleaned_line:
        return None
    if len(cleaned_line.split()) > 20 and not any(kw.lower() in cleaned_line.lower() for kw in ["appendix", "chapter", "section", "attachment"]):
        logger.debug(f"Line too long (>20 words) to be a typical non-primary heading: '{cleaned_line[:80]}...'")
        return None

    structural_keywords_map_for_code_gen = {
        "INTRODUCTION": "INTRODUCTION", "SUMMARY": "SUMMARY", "CONCLUSION": "CONCLUSION",
        "OVERVIEW": "OVERVIEW", "REFERENCES": "REFERENCES", "FOREWORD": "FOREWORD",
        "PREFACE": "PREFACE", "ACKNOWLEDGEMENTS": "ACKNOWLEDGEMENTS",
        "POLICY INFORMATION": "POLICY_INFO",
        "TERM DEFINITIONS": "TERM_DEFINITIONS",
        "CRIMINAL JUSTICE INFORMATION": "CJI_HEADING",
        "SECURITY AND MANAGEMENT CONTROL OUTSOURCING STANDARD": "SMC_OUTSOURCING_STD",
        "TABLE OF CONTENTS": "TOC_HEADING"
    }

    for pattern_idx, pattern in enumerate(HEADING_PATTERNS):
        match = pattern.match(cleaned_line)
        if not match:
            continue

        groups = match.groupdict()
        final_code = ""
        final_title = ""
        p_code_str_original = (groups.get('code') or "").strip() # Get original code string, handle None

        if pattern_idx == 0: # Title (CODE)
            if p_code_str_original and not STRICT_CODE_PATTERN_FOR_MATCH.match(p_code_str_original):
                logger.debug(f"Pattern 0: Code '{p_code_str_original}' does not match strict format. Skipping. Line: '{cleaned_line[:80]}'")
                continue
            final_title = (groups.get('title') or "").strip().rstrip('.:-— ')
            final_code = p_code_str_original.upper()

        elif pattern_idx == 1: # CODE TITLE, more specific
            if p_code_str_original and not STRICT_CODE_PATTERN_FOR_MATCH.match(p_code_str_original):
                logger.debug(f"Pattern 1: Code '{p_code_str_original}' does not match strict format. Skipping. Line: '{cleaned_line[:80]}'")
                continue
            final_code = p_code_str_original.upper()
            final_title = (groups.get('title') or "").strip().rstrip('.:-— ')
            # Existing logic for acronym in title for pattern_idx 1
            title_acronym_match = re.search(r'\s+\((?P<acronym>[A-Z0-9]{1,6}(?:[\-\.\s][A-Z0-9]{1,6})*)\)$', final_title)
            if title_acronym_match:
                acronym = title_acronym_match.group("acronym").upper()
                base_title = final_title[:title_acronym_match.start()].strip()
                is_specific_existing_code = bool(re.match(r'^[A-Z0-9]{2,}([\.\-][A-Z0-9]+)+$', final_code)) or len(final_code) > 5
                if not is_specific_existing_code and len(acronym) <= 6 and acronym == final_code:
                    final_title = base_title
                # else: p_code and p_title are already set if acronym logic doesn't simplify

        elif pattern_idx == 2: # CODE TITLE, more general
            # No strict code pattern check for this more general case
            p_code = (groups.get('code') or "").strip().rstrip('.:-— ').upper()
            p_title = (groups.get('title') or "").strip().rstrip('.:-— ')
            title_acronym_match = re.search(r'\s+\((?P<acronym>[A-Z0-9]{1,6}(?:[\-\.\s][A-Z0-9]{1,6})*)\)$', p_title)
            if title_acronym_match:
                acronym = title_acronym_match.group("acronym").upper()
                base_title = p_title[:title_acronym_match.start()].strip()
                is_specific_existing_code = bool(re.match(r'^[A-Z0-9]{2,}([\.\-][A-Z0-9]+)+$', p_code)) or len(p_code) > 5
                if not is_specific_existing_code and len(acronym) <= 6 and acronym == p_code:
                    final_code = p_code
                    final_title = base_title
                else:
                    final_code = p_code
                    final_title = p_title
            else:
                final_code = p_code
                final_title = p_title
            if final_title :
                words_in_title = final_title.split()
                if len(words_in_title) > 7 and sum(1 for w in words_in_title if w.islower()) > len(words_in_title) / 2:
                    logger.debug(f"Pattern 2: Title '{final_title[:60]}' looks like a sentence. Skipping.")
                    continue
        elif pattern_idx in [3, 4]: # Appendix/Chapter/Section CODE Title or just CODE
            final_code = (groups.get('code') or "").strip().rstrip('.:-— ').upper().replace(" ", "_")
            final_title = (groups.get('title') or "").strip().rstrip('.:-— ')
        elif pattern_idx == 5: # Common headings like Introduction
            p_title = (groups.get('title') or "").strip().rstrip('.:-— ')
            final_title = p_title
            title_upper_key = p_title.upper()
            if title_upper_key in structural_keywords_map_for_code_gen:
                final_code = structural_keywords_map_for_code_gen[title_upper_key]
            else:
                final_code = title_upper_key.replace(" ", "_")
                final_code = re.sub(r'[^A-Z0-9_]+', '', final_code)
        elif pattern_idx == 6: # Title Base (CODE)
            title_from_base = (groups.get('title_base') or "").strip().rstrip('.:-— ')
            acronym_from_code_group = (groups.get('code') or "").strip().rstrip('.:-— ').upper()

            if acronym_from_code_group: # Code was explicitly in parentheses
                if not STRICT_CODE_PATTERN_FOR_MATCH.match(acronym_from_code_group):
                    # If code in parens doesn't meet strict, it's probably not a formal code for this pattern
                    logger.debug(f"Pattern 6 (Title Base (CODE)): Code '{acronym_from_code_group}' in parens does not match strict format. Treating as part of title_base or ignoring. Line: '{cleaned_line[:80]}'")
                    # Fallback to treat title_from_base as the full title, and try to derive code from it
                    # This prevents a good title like "Access Control (AC)" from being skipped if just "AC" is not matched by strict
                    title_upper_key = title_from_base.upper() # Use the original combined title for code gen
                    num_words = len(title_from_base.split())
                    if title_upper_key in structural_keywords_map_for_code_gen:
                        final_code = structural_keywords_map_for_code_gen[title_upper_key]
                        final_title = title_from_base
                    else: # logic for generating code from title_from_base if it's a valid header
                        is_all_caps = title_from_base.isupper()
                        is_title_case = title_from_base.istitle() and num_words > 0
                        if num_words >= 1 and num_words <= 7 and (is_all_caps or is_title_case):
                            final_title = title_from_base
                            temp_code = title_upper_key.replace(" ", "_")
                            temp_code = re.sub(r'[^A-Z0-9_]+', '', temp_code)
                            if not temp_code: continue
                            final_code = temp_code
                        else: # too complex, skip this pattern instance
                            continue

                else: # Acronym from code group IS a strict code
                    final_code = acronym_from_code_group
                    final_title = title_from_base

            elif title_from_base: # No (CODE) part, only title_base matched
                title_upper_key = title_from_base.upper()
                num_words = len(title_from_base.split())
                if title_upper_key in structural_keywords_map_for_code_gen:
                    final_code = structural_keywords_map_for_code_gen[title_upper_key]
                    final_title = title_from_base
                else:
                    is_all_caps = title_from_base.isupper()
                    is_title_case = title_from_base.istitle() and num_words > 0
                    if num_words >= 1 and num_words <= 7 and (is_all_caps or is_title_case):
                        final_title = title_from_base
                        temp_code = title_upper_key.replace(" ", "_")
                        temp_code = re.sub(r'[^A-Z0-9_]+', '', temp_code)
                        if not temp_code: continue
                        final_code = temp_code
                    elif num_words > 4 or len(title_from_base) > 30:
                        lowercase_words = sum(1 for word_idx, word in enumerate(title_from_base.split()) if any(c.islower() for c in word) and not (word_idx == 0 and word.istitle()))
                        if num_words > 5 and lowercase_words >= (num_words / 2.5) and not is_title_case and not is_all_caps:
                            logger.debug(f"Pattern 6 (Title Base Only): title '{title_from_base[:60]}' appears sentence-like. Skipping.")
                            continue
                        final_title = title_from_base
                        first_word = title_upper_key.split()[0] if title_from_base.split() else "HEADING"
                        if first_word.lower() in ["the", "an", "a", "of", "to", "in", "on", "at", "is", "for"] and len(title_upper_key.split()) > 1:
                            first_word = title_upper_key.split()[1] if len(title_upper_key.split()) > 1 else "SUB_HEADING"
                        final_code = "HEADING_" + re.sub(r'[^A-Z0-9]', '', first_word[:10])
                    else:
                        continue
            else: # Neither title_base nor code in parens matched
                continue

        elif pattern_idx == 7: # Just a CODE
            final_code = (groups.get('code') or "").strip().rstrip('.:-— ').upper()
            if not STRICT_CODE_PATTERN_FOR_MATCH.match(final_code):
                logger.debug(f"Pattern 7: Code '{final_code}' does not match strict format. Skipping. Line: '{cleaned_line[:80]}'")
                continue
            final_title = final_code
            if cleaned_line != final_code: continue # Ensure entire line was just the code
            # (Additional checks for this pattern type remain)
            is_complex_code_format = bool(re.search(r'[\.\-_]', final_code))
            is_short_alphanum_code = len(final_code) >= 2 and len(final_code) <= 15 and final_code.isalnum() and not final_code.isdigit()
            is_potentially_common_word_as_code = len(final_code) >=3 and len(final_code) <=10 and final_code.isalpha() and final_code.isupper()
            if not (is_complex_code_format or is_short_alphanum_code or is_potentially_common_word_as_code):
                continue

        if final_title:
            words_in_title_list = final_title.split()
            if len(words_in_title_list) > 1 and all(w.strip().lower() == words_in_title_list[0].strip().lower() for w in words_in_title_list):
                logger.debug(f"Detected repetitive title: '{final_title}'. Using first word: '{words_in_title_list[0]}'")
                final_title = words_in_title_list[0].strip()

        if not final_code and final_title:
            title_upper_key_temp = final_title.upper()
            if title_upper_key_temp in structural_keywords_map_for_code_gen:
                final_code = structural_keywords_map_for_code_gen[title_upper_key_temp]
            else:
                temp_c = final_title.replace(" ", "_").upper()
                temp_c = re.sub(r'[^A-Z0-9_]+', '', temp_c)
                if temp_c and (len(final_title.split()) < 5 and len(final_title) < 35):
                    final_code = temp_c
                else:
                    logger.debug(f"Pattern {pattern_idx}: Title-only '{final_title[:60]}' too long/complex for auto-code. Skipping.")
                    continue

        if not final_title and final_code: final_title = final_code
        if not final_code.strip() and not final_title.strip(): continue

        final_code = final_code if final_code.strip() else "CODE_EMPTY_" + str(uuid.uuid4())[:4]
        final_title = final_title if final_title.strip() else final_code

        if final_title:
            title_lower_check = final_title.lower()
            if re.search(r';\s*(and|or|but|for|nor|so|yet)$', title_lower_check) or \
               re.fullmatch(r'(and|or|but|for|nor|so|yet|the|a|is|of|to|in|on|at)\b', title_lower_check) or \
               not re.search(r'[a-zA-Z0-9]', final_title):
                logger.debug(f"Pattern {pattern_idx}: Title '{final_title}' seems invalid. Defaulting to code. Line: '{line[:80]}'")
                final_title = final_code
                if not re.search(r'[a-zA-Z0-9]', final_title): continue

        if len(final_code) > 60 and not any(fc_part in final_code for fc_part in ["GENERATED", "HEADING", "UNKNOWN", "CODE_EMPTY", "ERROR_"]):
            logger.debug(f"Code '{final_code}' too long (>60). Rejecting. Line: '{line[:80]}'")
            continue
        if len(final_code) < 2 and final_code.isalpha() and not (final_code in structural_keywords_map_for_code_gen.values() or "GENERATED" in final_code or "CODE_EMPTY" in final_code):
            # Allow single char codes if they passed strict pattern (e.g., Appendix A)
            if not (len(final_code) == 1 and pattern_idx in [3,4,7]): # Pattern 3,4 (Appendix A etc.) or 7 (if strictly A) can be single char
                logger.debug(f"Rejecting very short alphabetic code '{final_code}' unless specific pattern. Line: '{line[:80]}'")
                continue


        if final_code == final_title:
            is_complex_fmt = bool(re.search(r'[\.\-_]', final_code))
            if (len(final_code) > 4 and final_code.isdigit()) or \
               (len(final_code) > 4 and re.search(r'\d', final_code) and not is_complex_fmt and final_code.isupper()) or \
               re.fullmatch(r'([A-Za-z0-9]{1,4})\1{2,}', final_code):
                logger.debug(f"Code/title '{final_code}' looks like artifact. Skipping. Line: '{line[:80]}'")
                continue

        logger.debug(f"Heading matched (Pattern #{pattern_idx}): Code='{final_code}', Title='{final_title}' from clean_line='{cleaned_line[:80]}'")
        return final_code, final_title

    logger.debug(f"No heading pattern matched for line: '{cleaned_line[:80]}'")
    return None


def format_camelot_table_to_markdown(table_df: "pd.DataFrame", page_num: int, table_idx: int) -> str:
    if table_df.empty:
        logger.debug(f"Camelot Table {page_num}-{table_idx} DataFrame is empty.")
        return ""

    for col in table_df.columns:
        table_df[col] = table_df[col].apply(lambda x: str(x).replace('\n', ' ').strip() if pd.notnull(x) else "")

    try:
        markdown_table = table_df.to_markdown(index=False)
        if markdown_table.strip():
            table_info_line = f"\n[Table extracted from page {page_num}, index {table_idx}]\n"
            return table_info_line + markdown_table + "\n"
        else:
            logger.debug(f"Camelot Table {page_num}-{table_idx} resulted in empty markdown.")
            return ""
    except Exception as e:
        logger.error(f"Error converting Camelot table {page_num}-{table_idx} to Markdown: {e}")
        return ""


def split_text_into_chunks_robust(text: str, max_words: int, overlap_words: int,
                                  min_chunk_word_count: int) -> List[str]:
    text_words = text.split()
    text_word_count = len(text_words)

    if not text.strip():
        logger.debug("Text is empty, cannot chunk.")
        return []

    if text_word_count < min_chunk_word_count:
        if text_word_count > 0 and text_word_count <= max_words :
            logger.debug(f"Text too short ({text_word_count} words) for min_chunk_word_count ({min_chunk_word_count}), "
                         f"but fits max_words ({max_words}) and is not empty. Returning as single chunk. Text: '{text[:50]}...'")
            return [text.strip()]
        else:
            logger.debug(f"Text too short ({text_word_count} words) for min_chunk_word_count ({min_chunk_word_count}). "
                         f"Not returning. Text: '{text[:50]}...'")
            return []


    if text_word_count <= max_words:
        return [text.strip()]

    initial_sentences: List[str] = []
    try:
        nltk_sentences = nltk.sent_tokenize(text)
        if nltk_sentences:
            initial_sentences = [s.strip() for s in nltk_sentences if s.strip()]
        if not initial_sentences:
            logger.warning(f"NLTK sent_tokenize returned no usable sentences. Text: '{text[:100]}...'")
    except Exception as e:
        logger.warning(f"NLTK sentence tokenization failed ('{str(e).splitlines()[0]}'). Falling back. Text: '{text[:100]}...'")

    if not initial_sentences:
        paragraphs = text.split('\n\n')
        for p in paragraphs:
            if p.strip(): initial_sentences.append(p.strip())

    if not initial_sentences:
        lines = text.split('\n')
        for l_text in lines:
            if l_text.strip(): initial_sentences.append(l_text.strip())

    if not initial_sentences:
        logger.warning(f"All tokenization fallbacks failed. Splitting by max_words. Text: '{text[:100]}...'")
        current_pos = 0
        while current_pos < text_word_count:
            end_pos = min(current_pos + max_words, text_word_count)
            pseudo_sent = " ".join(text_words[current_pos:end_pos])
            if pseudo_sent.strip(): initial_sentences.append(pseudo_sent)
            current_pos = end_pos

    if not initial_sentences:
        logger.error(f"Could not tokenize text into any segments: '{text[:100]}...'")
        return []


    processed_sentences: List[str] = []
    for sentence in initial_sentences:
        words_in_sentence = sentence.split()
        num_words_in_sentence = len(words_in_sentence)
        if num_words_in_sentence == 0: continue

        if num_words_in_sentence > max_words:
            logger.debug(f"Sentence ({num_words_in_sentence} words) > max_words ({max_words}). Sub-splitting: '{sentence[:80]}...'")
            start_idx = 0
            while start_idx < num_words_in_sentence:
                end_idx = min(start_idx + max_words, num_words_in_sentence)
                sub_sentence_part = " ".join(words_in_sentence[start_idx:end_idx])
                if sub_sentence_part.strip(): processed_sentences.append(sub_sentence_part)

                if end_idx == num_words_in_sentence: break
                start_idx = max(start_idx + 1, end_idx - overlap_words if overlap_words < max_words else end_idx - (max_words // 2))
                if start_idx >= end_idx : start_idx = end_idx
        else:
            processed_sentences.append(sentence)

    if not processed_sentences:
        logger.debug(f"No processable sentences after pre-splitting long ones for: '{text[:100]}...'")
        return []


    final_chunks: List[str] = []
    current_chunk_word_list: List[str] = []

    for i, sentence_text in enumerate(processed_sentences):
        sentence_words = sentence_text.split()
        num_sentence_words = len(sentence_words)
        if num_sentence_words == 0: continue

        if current_chunk_word_list and (len(current_chunk_word_list) + num_sentence_words > max_words):
            chunk_to_add_text = " ".join(current_chunk_word_list)
            if len(current_chunk_word_list) >= min_chunk_word_count:
                final_chunks.append(chunk_to_add_text)

            overlap_word_count_actual = min(overlap_words, len(current_chunk_word_list))
            words_for_overlap: List[str] = current_chunk_word_list[-overlap_word_count_actual:] if overlap_word_count_actual > 0 else []
            current_chunk_word_list = words_for_overlap

            if current_chunk_word_list and (len(current_chunk_word_list) + num_sentence_words > max_words):
                if len(current_chunk_word_list) >= min_chunk_word_count:
                    overlap_only_chunk_text = " ".join(current_chunk_word_list)
                    final_chunks.append(overlap_only_chunk_text)
                current_chunk_word_list = list(sentence_words)
            elif current_chunk_word_list is not sentence_words:
                current_chunk_word_list.extend(sentence_words)
            elif not current_chunk_word_list: # Should not happen if first condition was true and then this one
                current_chunk_word_list = list(sentence_words)

        else:
            if not current_chunk_word_list:
                current_chunk_word_list = list(sentence_words)
            elif current_chunk_word_list is not sentence_words: # Make sure we don't extend if it's somehow the same list
                current_chunk_word_list.extend(sentence_words)


    if current_chunk_word_list:
        last_chunk_text = " ".join(current_chunk_word_list)
        if len(current_chunk_word_list) >= min_chunk_word_count:
            final_chunks.append(last_chunk_text)
        elif last_chunk_text.strip(): # Only log if there was actual text and it was discarded
            logger.debug(f"Final segment ({len(current_chunk_word_list)} words) < min_chunk_word_count ({min_chunk_word_count}). Discarding. Text: '{last_chunk_text[:50]}...'")

    return [chunk for chunk in final_chunks if chunk and len(chunk.split()) >= min_chunk_word_count]


def initialize_google_ai_sdk() -> bool:
    if not genai:
        logger.error("Google Generative AI SDK (genai) not available. Cannot initialize.")
        return False
    try:
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            logger.info(f"Configured Google Generative AI SDK with GOOGLE_API_KEY using model {VERTEX_EMBEDDING_MODEL}.")
        else:
            logger.info(f"GOOGLE_API_KEY not found. Attempting to use Application Default Credentials for model {VERTEX_EMBEDDING_MODEL}.")
        return True
    except Exception as e:
        logger.error(f"Failed to configure Google Generative AI SDK: {e}. "
                     f"Ensure GOOGLE_APPLICATION_CREDENTIALS is set correctly for ADC, "
                     f"or that GOOGLE_API_KEY is valid and has the 'Vertex AI API' or 'Generative Language API' enabled.")
        return False

def extract_simple_keywords(text: str, stop_words: Set[str], top_n: int = 10, ngram_range: Tuple[int, int] = (1, 2)) -> List[str]:
    if not text:
        return []

    words = [word.lower() for word in nltk.word_tokenize(text) if word.isalpha() and word.lower() not in stop_words and len(word) > 1]

    all_ngrams: List[Tuple[str, ...]] = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        if len(words) >= n :
            all_ngrams.extend(list(nltk.ngrams(words, n)))

    if not all_ngrams:
        return []

    ngram_counts = Counter(all_ngrams)
    most_common_ngrams_tuples = ngram_counts.most_common(top_n)

    keywords = [" ".join(ngram_tuple) for ngram_tuple, count in most_common_ngrams_tuples]
    return keywords

def get_immediate_parent_info(section_code: str, code_to_title_map: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    if not section_code:
        return None, None

    parent_code_str: Optional[str] = None
    parts = section_code.split('.')
    if len(parts) > 1:
        parent_code_str = ".".join(parts[:-1])
    else:
        parts_underscore = section_code.split('_')
        if len(parts_underscore) > 1:
            last_part = parts_underscore[-1]
            # Heuristic: if the last part after '_' is short, numeric, or single char, assume it's a sub-identifier
            if len(last_part) <= 2 or last_part.isdigit() or (len(last_part)==1 and last_part.isalpha()):
                parent_code_str = "_".join(parts_underscore[:-1])

    if parent_code_str:
        parent_title = code_to_title_map.get(parent_code_str)
        return parent_code_str, parent_title
    return None, None


def get_full_hierarchical_path(section_code: str, code_to_title_map: Dict[str, str]) -> str:
    path_titles_reversed: List[str] = []
    current_child_code_optional: Optional[str] = section_code
    max_depth = 10
    count = 0

    if current_child_code_optional:
        initial_title = code_to_title_map.get(current_child_code_optional, current_child_code_optional)
        path_titles_reversed.append(initial_title)

    while current_child_code_optional and count < max_depth:
        parent_s_code, parent_s_title = get_immediate_parent_info(current_child_code_optional, code_to_title_map)

        if parent_s_code:
            title_for_path = parent_s_title if parent_s_title else parent_s_code
            if not path_titles_reversed or path_titles_reversed[-1] != title_for_path: # Avoid adding identical parent if logic somehow results in that
                path_titles_reversed.append(title_for_path)


            if parent_s_code == current_child_code_optional: # safety break for self-parenting
                logger.warning(f"Self-parenting detected for {parent_s_code} during path generation for {section_code}. Path construction stopped.")
                break
            current_child_code_optional = parent_s_code
        else:
            break # No more parents

        count += 1
        if count >= max_depth and parent_s_code: # Check if parent_s_code is still valid, implies more depth
            logger.warning(f"Max depth ({max_depth}) reached for hierarchical path generation, "
                           f"starting from {section_code}. Path might be incomplete.")
            break

    return " > ".join(reversed(path_titles_reversed))

# --- Helper functions for nuanced RAG FT model completions ---
def check_for_frequency_keywords(text: str) -> bool:
    if not text: return False
    pattern = re.compile(
        r'\b('
        r'dai(?:ly)?|week(?:ly)?|month(?:ly)?|quarter(?:ly)?|annuall?y|year(?:ly)?|'
        r'periodic(?:ally)?|regular(?:ly)?|routinely?|'
        r'ad-hoc|as needed|upon\s+\w+|continu(?:ous|ally)|ongoing|'
        r'\d+\s+(?:hour|day|week|month|year)s?|'
        r'once\s+a\s+(?:day|week|month|year)|'
        r'every\s+(?:\d+\s+)?(?:hour|day|week|month|year)s?|'
        r'at least\s+(?:once|daily|annually|\d+\s+times)|'
        r'no less than|no more than'
        r')\b',
        re.IGNORECASE
    )
    return bool(pattern.search(text))

def check_for_requirement_keywords(text: str) -> bool:
    if not text: return False
    pattern = re.compile(
        r'\b('
        r'shall(?!\s+not)|must(?!\s+not)|'
        r'(?:is|are|was|were)\s+required|'
        r'mandatory|mandate[ds]?|'
        r'needs\s+to|ensure\s+that|obligat(?:ed|ion)|'
        r'responsible\s+for\s+ensuring' # More specific than generic "responsible for"
        r')\b',
        re.IGNORECASE
    )
    return bool(pattern.search(text))

def check_for_responsibility_keywords(text: str) -> bool:
    if not text: return False
    pattern = re.compile(
        r'\b('
        r'responsible\s+for(?!.*ensuring)|responsibilit(?:y|ies)\s+(?:of|lies\s+with)|'
        r'duty\s+of|duties\s+include|'
        r'accountable\s+for|'
        r'assigned\s+to|tasked\s+with|'
        r'role\s+of|function\s+of'
        r')\b',
        re.IGNORECASE
    )
    return bool(pattern.search(text))


# Global set for sentence deduplication per document run
global_seen_sentence_hashes: Set[str] = set()
# Global NLTK stopwords
NLTK_STOPWORDS: Set[str] = set()


def main(args):
    global MIN_FT_CHUNK_WORD_COUNT, NUM_NOT_FOUND_EXAMPLES_PER_CHUNK, GHOSTSCRIPT_AVAILABLE, MIN_RAG_CHUNK_WORD_COUNT
    global global_seen_sentence_hashes, NLTK_STOPWORDS, RAG_FT_PROMPT_CONFIG

    global_seen_sentence_hashes = set() # Reset for each run
    try:
        NLTK_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
    except LookupError:
        logger.warning("NLTK stopwords resource not found during main execution. Keyword extraction quality may be reduced.")
        NLTK_STOPWORDS = set()


    MIN_FT_CHUNK_WORD_COUNT = max(10, args.ft_word_overlap_for_chunks // 2 if args.ft_word_overlap_for_chunks > 0 else 10)
    MIN_RAG_CHUNK_WORD_COUNT = max(15, args.rag_word_overlap_for_chunks // 2 if args.rag_word_overlap_for_chunks > 0 else 15)
    logger.info(f"MIN_FT_CHUNK_WORD_COUNT set to: {MIN_FT_CHUNK_WORD_COUNT}")
    logger.info(f"MIN_RAG_CHUNK_WORD_COUNT set to: {MIN_RAG_CHUNK_WORD_COUNT}")

    NUM_NOT_FOUND_EXAMPLES_PER_CHUNK = args.num_not_found_examples_per_chunk

    GHOSTSCRIPT_AVAILABLE = check_ghostscript()
    if not GHOSTSCRIPT_AVAILABLE and CAMELOT_IMPORTED and args.extract_tables:
        logger.warning("Ghostscript seems unavailable. Camelot table extraction may fail or produce poor results.")

    if not LANGCHAIN_IMPORTED:
        logger.critical("LangChain library is not available. Prompt generation for fine-tuning will fail. Exiting.")
        return
    if not initialize_langchain_templates():
        logger.critical("Failed to initialize LangChain prompt templates. Exiting.")
        return

    if not PYMUPDF_IMPORTED:
        logger.critical("PyMuPDF (fitz) is not installed or failed to import. Cannot extract text. Exiting.")
        return
    if not CAMELOT_IMPORTED and args.extract_tables:
        logger.warning("Camelot library is not available. Table extraction will be skipped.")


    logger.info(f"🟢 CJIS Policy PDF Parser ({SCRIPT_VERSION}) starting.")
    logger.info(f"Text extraction: PyMuPDF. Table extraction: {'Camelot' if CAMELOT_IMPORTED and args.extract_tables else 'Skipped'}. Prompt Generation: LangChain.")
    logger.info(f"Keyword Extraction for RAG: {'Enabled' if args.extract_keywords_for_rag else 'Disabled'}")


    if genai is None and args.generate_embeddings_for_rag:
        logger.error("Google Generative AI SDK (genai) not loaded. Cannot generate embeddings. Run: pip install google-generativeai")
        return

    for out_file in [args.output_jsonl_finetune_file, args.output_jsonl_rag_file]:
        output_dir = os.path.dirname(out_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

    pdf_filename_stem = os.path.splitext(os.path.basename(args.input_file))[0]
    source_doc_name_for_metadata = os.path.basename(args.input_file)

    parsed_sections_accumulator: List[Dict[str, Any]] = []
    _active_section_code: Optional[str] = None
    _active_section_title: Optional[str] = None
    _active_section_start_page: Optional[str] = None
    _active_section_last_content_page_num: Optional[str] = None # NEW: Tracks last page content was added for a section
    _active_section_content_items: List[str] = []

    logger.info(f"Starting PDF parsing: {args.input_file} (Pages: {args.page_start}-{args.page_end})")
    doc = None
    last_processed_page_for_final_section = str(args.page_start) # Fallback for error cases or no content
    try:
        doc = fitz.open(args.input_file)
        actual_start_page_idx = args.page_start - 1
        actual_end_page_idx = min(args.page_end, doc.page_count)

        if actual_start_page_idx < 0:
            logger.warning(f"Adjusted start page from {args.page_start} to 1 (0-indexed: 0).")
            actual_start_page_idx = 0
        if actual_start_page_idx >= doc.page_count:
            logger.error(f"Start page {args.page_start} is beyond document end ({doc.page_count} pages). Exiting.")
            if doc is not None: doc.close()
            doc = None
            return
        if actual_start_page_idx >= actual_end_page_idx :
             logger.error(f"Start page {args.page_start} not before end page {args.page_end}. Effective 0-idx range: {actual_start_page_idx} to {actual_end_page_idx-1}. Exiting.")
             if doc is not None: doc.close()
             doc = None
             return

        logger.info(f"Effective 0-indexed page range for processing: {actual_start_page_idx} to {actual_end_page_idx-1}")
        _active_section_start_page = str(actual_start_page_idx + 1) # Initialize with overall start
        _active_section_last_content_page_num = str(actual_start_page_idx + 1)


        for pg_num_0_indexed in tqdm(range(actual_start_page_idx, actual_end_page_idx), desc="Processing PDF Pages"):
            page = doc.load_page(pg_num_0_indexed)
            page_number_for_display = str(pg_num_0_indexed + 1)
            last_processed_page_for_final_section = page_number_for_display # Keep track for final section if loop ends
            current_page_text_items: List[str] = []
            page_rect_height = page.rect.height
            page_content_added_this_iter = False # Flag if content was added from this page

            page_lines_with_coords: List[Dict[str, Any]] = []
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT | fitz.TEXTFLAGS_SEARCH, sort=True)
            for block in blocks.get("blocks", []): # Access the "blocks" key, default to empty if not present
                if block.get("type") == 0: # Text block
                    for line_dict in block.get("lines", []):
                        line_text_parts = [span.get("text","") for span in line_dict.get("spans",[])]
                        line_text_full = " ".join(line_text_parts).strip()
                        if line_text_full:
                            bbox = line_dict.get("bbox", (0,0,0,0))
                            y0, y1 = bbox[1], bbox[3]
                            page_lines_with_coords.append({
                                "text": line_text_full,
                                "y0": y0,
                                "y1": y1
                            })

            if not page_lines_with_coords:
                logger.debug(f"P{page_number_for_display}: No text lines extracted using get_text('dict'). Page might be image-based or empty.")

            for line_obj in page_lines_with_coords:
                line_text_raw, line_y0, line_y1 = line_obj["text"], line_obj["y0"], line_obj["y1"]
                line_cleaned_for_eval = line_text_raw.strip()

                if not line_cleaned_for_eval: continue

                if '|' in line_cleaned_for_eval and line_cleaned_for_eval.count('|') > 2 and len(line_cleaned_for_eval) > 20:
                    logger.debug(f"P{page_number_for_display}: Skipping text seeming like table data based on pipes: '{line_cleaned_for_eval[:60]}...'")
                    continue

                if is_footer_or_header_line(line_cleaned_for_eval, page_number_for_display, pdf_filename_stem,
                                            line_y0, line_y1, page_rect_height,
                                            header_margin=args.header_footer_margin, footer_margin=args.header_footer_margin):
                    logger.debug(f"P{page_number_for_display}: Skipping H/F line: '{line_cleaned_for_eval[:60]}...' (y0:{line_y0:.1f}, y1:{line_y1:.1f}, H:{page_rect_height:.1f})")
                    continue

                heading_match_result = match_heading(line_cleaned_for_eval)
                if heading_match_result:
                    new_code, new_title = heading_match_result
                    _active_section_content_items.extend(current_page_text_items) # Add pending items from current page
                    if current_page_text_items: # If any text from current page was added *before* this new heading
                        _active_section_last_content_page_num = page_number_for_display
                    current_page_text_items = [] # Reset for this page after heading

                    if _active_section_code is not None or _active_section_title is not None: # If there was a previous active section
                        if _active_section_content_items: # Only archive if previous section had content
                            parsed_sections_accumulator.append({
                                "code": _active_section_code or "UNKNOWN_CODE_PREV",
                                "title": _active_section_title or "Untitled Section (Prev)",
                                "content_items": list(_active_section_content_items), # make a copy
                                "source_page_start": _active_section_start_page or "UNKNOWN_PAGE_PREV",
                                "source_page_end": _active_section_last_content_page_num or _active_section_start_page or "UNKNOWN_PAGE_PREV"
                            })
                            logger.debug(f"P{page_number_for_display}: Archived previous section: '{_active_section_code} - {_active_section_title}' "
                                         f"(P{_active_section_start_page}-{_active_section_last_content_page_num}) with {len(_active_section_content_items)} items.")
                        else:
                             logger.debug(f"P{page_number_for_display}: Prev section '{_active_section_code} - {_active_section_title}' had no content. Not archiving.")

                    _active_section_content_items = [] # Reset for the new section
                    _active_section_code = new_code
                    _active_section_title = new_title
                    _active_section_start_page = page_number_for_display
                    _active_section_last_content_page_num = page_number_for_display # Heading page is initially start & end
                    logger.info(f"P{page_number_for_display}: New section detected: Code='{_active_section_code}', Title='{_active_section_title}'")
                    page_content_added_this_iter = True # Heading itself means something started on this page
                    continue # Finished processing this line as a heading
                else: # Not a heading, add to current page's text items for the active section
                    current_page_text_items.append(line_text_raw)
                    page_content_added_this_iter = True

            _active_section_content_items.extend(current_page_text_items) # Add any remaining text from this page to the active section
            if page_content_added_this_iter or current_page_text_items : # if any text was added from this page
                _active_section_last_content_page_num = page_number_for_display

            if args.extract_tables and CAMELOT_IMPORTED and camelot and GHOSTSCRIPT_AVAILABLE:
                try:
                    logger.debug(f"P{page_number_for_display}: Attempting table extraction with Camelot...")
                    tables_lattice = camelot.read_pdf(args.input_file, pages=page_number_for_display, flavor='lattice', suppress_stdout=True, line_scale=40)
                    camelot_tables = tables_lattice
                    flavor_used = 'lattice'

                    if tables_lattice.n == 0:
                        logger.debug(f"P{page_number_for_display}: Camelot (lattice) found 0 tables. Trying 'stream' flavor.")
                        tables_stream = camelot.read_pdf(args.input_file, pages=page_number_for_display, flavor='stream', suppress_stdout=True, edge_tol=500, row_tol=10)
                        camelot_tables = tables_stream
                        flavor_used = 'stream'
                        if tables_stream.n > 0:
                            logger.info(f"P{page_number_for_display}: Camelot (stream) found {tables_stream.n} table(s).")
                    elif tables_lattice.n > 0:
                         logger.info(f"P{page_number_for_display}: Camelot (lattice) found {tables_lattice.n} table(s).")


                    if camelot_tables.n > 0:
                        for table_idx, ct_table in enumerate(camelot_tables):
                            logger.debug(f"P{page_number_for_display}: Processing Camelot table {table_idx+1}/{camelot_tables.n}. Flavor: {ct_table.flavor} (Attempted: {flavor_used}). Accuracy: {ct_table.accuracy:.2f}%, Whitespace: {ct_table.whitespace:.2f}%")
                            if ct_table.df.empty:
                                logger.debug(f"P{page_number_for_display}: Camelot table {table_idx+1} DataFrame is empty, skipping.")
                                continue
                            md_table = format_camelot_table_to_markdown(ct_table.df, int(page_number_for_display), table_idx + 1)
                            if md_table.strip():
                                _active_section_content_items.append(md_table)
                                _active_section_last_content_page_num = page_number_for_display # Table added from this page
                                logger.debug(f"Added Camelot table {table_idx+1} from P{page_number_for_display} to section '{_active_section_code if _active_section_code else 'INITIAL_BUFFER'}'.")
                            else:
                                logger.debug(f"P{page_number_for_display}: Camelot table {table_idx+1} was empty after formatting.")
                except Exception as e:
                    if "Ghostscript" in str(e) or "GS" in str(e).upper():
                        logger.error(f"P{page_number_for_display}: Ghostscript error during Camelot table extraction. Please ensure Ghostscript is installed and in PATH. Error: {e}")
                        GHOSTSCRIPT_AVAILABLE = False # Disable further attempts if it definitively failed due to GS
                    else:
                        logger.warning(f"P{page_number_for_display}: Error during Camelot table extraction ({type(e).__name__}: {str(e)[:150]}). Skipping table proc for this page.")
            elif args.extract_tables and not CAMELOT_IMPORTED:
                logger.debug(f"P{page_number_for_display}: Camelot not imported. Skipping table extraction.")
            elif args.extract_tables and CAMELOT_IMPORTED and not GHOSTSCRIPT_AVAILABLE:
                logger.debug(f"P{page_number_for_display}: Ghostscript unavailable. Skipping Camelot table extraction.")

    except FileNotFoundError:
        logger.critical(f"Input PDF file not found: {args.input_file}. Exiting.")
        if doc is not None: doc.close()
        return
    except Exception as e:
        logger.critical(f"An error occurred during PDF processing: {type(e).__name__}: {e}", exc_info=True)
        # Save partial last section if possible upon error
        if _active_section_code or _active_section_title:
            if _active_section_content_items: # Only if it had some content
                current_end_page = _active_section_last_content_page_num or _active_section_start_page or last_processed_page_for_final_section or "ERROR_PAGE"
                parsed_sections_accumulator.append({
                    "code": _active_section_code or "ERROR_CODE",
                    "title": _active_section_title or "Error Section",
                    "content_items": list(_active_section_content_items),
                    "source_page_start": _active_section_start_page or "ERROR_PAGE",
                    "source_page_end": current_end_page
                })
                logger.info(f"Saved partial last active section due to error: [{_active_section_code}] (P{_active_section_start_page}-{current_end_page})")
        if doc is not None: doc.close()
        return
    finally:
        if doc is not None:
            doc.close()

    # Finalize the last active section after loop completion
    final_section_start_page = _active_section_start_page or last_processed_page_for_final_section or str(args.page_start)
    final_section_end_page = _active_section_last_content_page_num or final_section_start_page

    if _active_section_code or _active_section_title:
        if _active_section_content_items : # Only add if there are content items
            parsed_sections_accumulator.append({
                "code": _active_section_code or "FINAL_CODE_UNKNOWN",
                "title": _active_section_title or "Final Section Untitled",
                "content_items": list(_active_section_content_items), # make a copy
                "source_page_start": final_section_start_page,
                "source_page_end": final_section_end_page
            })
            logger.info(f"Finalized last section: [{_active_section_code}] {_active_section_title} "
                        f"(P{final_section_start_page}-{final_section_end_page}) with {len(_active_section_content_items)} items.")
        else:
            logger.info(f"Last active section [{_active_section_code}] {_active_section_title} had no content. Not adding.")
    elif _active_section_content_items: # If content exists but no code/title was ever formally set (e.g. content before first heading)
        logger.warning(f"Found {len(_active_section_content_items)} content items buffered but no formal section detected globally. "
                       "Saving as 'ORPHANED_CONTENT'.")
        parsed_sections_accumulator.append({
            "code": "ORPHANED_CONTENT_NO_HEADINGS",
            "title": "Orphaned Content (No Headings Detected or Content Before First Heading)",
            "content_items": list(_active_section_content_items),
            "source_page_start": str(args.page_start), # Page range of the doc
            "source_page_end": last_processed_page_for_final_section # This would be the last page processed
        })

    logger.info(f"PDF parsing complete. Extracted {len(parsed_sections_accumulator)} raw sections.")
    if not parsed_sections_accumulator:
        logger.error("No sections extracted. Check HEADING_PATTERNS, page range, PDF text layer, or H/F margins. Exiting.")
        return

    all_valid_section_details_for_not_found = []
    code_to_title_map: Dict[str, str] = {}
    for sec_data in parsed_sections_accumulator:
        temp_code = sec_data["code"]
        temp_title = sec_data["title"]
        code_to_title_map[temp_code] = temp_title

        temp_raw_body = "\n".join(str(item) for item in sec_data.get("content_items", []))
        temp_cleaned_body_for_val = clean_text(temp_raw_body)

        is_skip_code = str(temp_code).upper() in SKIP_SECTION_CODES
        is_skip_title = any(pattern.search(str(temp_title or "")) for pattern in SKIP_SECTION_TITLE_PATTERNS)
        is_too_short_or_generic = (not temp_cleaned_body_for_val or
                                   len(temp_cleaned_body_for_val.split()) < MIN_SECTION_WORD_COUNT or
                                   any(gc in temp_code for gc in ["ORPHANED", "UNKNOWN", "GENERATED", "CODE_EMPTY", "ERROR_"]))

        if not (is_skip_code or is_skip_title or is_too_short_or_generic):
            all_valid_section_details_for_not_found.append({"code": temp_code, "title": temp_title})

    if not all_valid_section_details_for_not_found:
        logger.warning("No suitable 'other' sections found for 'Not Found' FT examples. This can happen with short docs or strict filtering.")
    else:
        logger.info(f"Collected {len(all_valid_section_details_for_not_found)} distinct valid sections for 'Not Found' example generation.")
    logger.info(f"Built code-to-title map with {len(code_to_title_map)} entries for hierarchical metadata.")


    total_finetune_records = 0
    total_rag_chunks = 0
    sdk_ok_for_embeddings = False

    if args.generate_embeddings_for_rag:
        if genai:
            sdk_ok_for_embeddings = initialize_google_ai_sdk()
            if not sdk_ok_for_embeddings:
                logger.warning("Google AI SDK init failed. Embeddings for RAG will NOT be generated.")
        else: # genai itself is None
            args.generate_embeddings_for_rag = False # Force disable if SDK isn't even loaded

    with open(args.output_jsonl_finetune_file, "w", encoding="utf-8", newline='\n') as ft_out_f, \
         open(args.output_jsonl_rag_file, "w", encoding="utf-8", newline='\n') as rag_out_f:

        for section_data in tqdm(parsed_sections_accumulator, desc="Formatting Sections for FT & RAG"):
            section_code = section_data["code"]
            section_title = section_data["title"]
            source_page_start = section_data["source_page_start"]
            source_page_end = section_data.get("source_page_end", source_page_start) # Use start_page if end not present (e.g. single pg section)


            if str(section_code).upper() in SKIP_SECTION_CODES or \
               any(pattern.search(str(section_title or "")) for pattern in SKIP_SECTION_TITLE_PATTERNS):
                logger.info(f"Skipping section marked for exclusion: Code='{section_code}', Title='{section_title}'")
                continue

            raw_body_from_items = "\n".join(str(item) for item in section_data.get("content_items", []))
            cleaned_body_initial = clean_text(raw_body_from_items)

            cleaned_body = ""
            if cleaned_body_initial:
                try:
                    sentences_for_dedup = nltk.sent_tokenize(cleaned_body_initial)
                except Exception as e:
                    logger.warning(f"NLTK sent_tokenize failed for section '{section_code}' body: {e}. Using raw body for this section's deduplication.")
                    sentences_for_dedup = [cs.strip() for cs in cleaned_body_initial.splitlines() if cs.strip()] # Fallback

                unique_sentences = []
                num_deduplicated_sentences = 0
                for sentence in sentences_for_dedup:
                    normalized_sentence_for_hash = sentence.strip().lower()
                    if not normalized_sentence_for_hash: # Skip empty sentences
                        continue
                    sentence_hash = hashlib.md5(normalized_sentence_for_hash.encode('utf-8')).hexdigest()

                    if sentence_hash not in global_seen_sentence_hashes:
                        unique_sentences.append(sentence.strip())
                        global_seen_sentence_hashes.add(sentence_hash)
                    else:
                        num_deduplicated_sentences += 1

                if num_deduplicated_sentences > 0:
                    logger.debug(f"Section '{section_code}': Deduplicated {num_deduplicated_sentences} sentences from this section's content (already seen globally).")
                cleaned_body = "\n".join(unique_sentences) # Rejoin unique sentences
            else: # cleaned_body_initial was empty
                cleaned_body = ""

            if not cleaned_body or len(cleaned_body.split()) < MIN_SECTION_WORD_COUNT:
                logger.info(f"Skipping section '[{section_code}] {section_title}' (P{source_page_start}-{source_page_end}) due to insufficient content after cleaning and deduplication ({len(cleaned_body.split())} words, need >{MIN_SECTION_WORD_COUNT}).")
                continue

            full_section_heading = f"{section_code} {section_title}".strip()

            # --- Fine-tuning Data Generation ---
            ft_text_chunks = split_text_into_chunks_robust(
                cleaned_body, args.ft_max_words_per_chunk, args.ft_word_overlap_for_chunks, MIN_FT_CHUNK_WORD_COUNT
            )
            if not ft_text_chunks:
                logger.warning(f"FT Chunking: no valid chunks for '{full_section_heading}' (P{source_page_start}-{source_page_end}). Original words after dedup: {len(cleaned_body.split())}. Skipping FT for this section.")

            if ft_text_chunks:
                num_ft_chunks = len(ft_text_chunks)
                for i, chunk_text_unescaped in enumerate(ft_text_chunks):
                    prompt_content_pairs_for_chunk: List[Tuple[str,str]] = []
                    current_chunk_system_instruction = ""

                    common_format_args = {
                        "policy_chunk": chunk_text_unescaped,
                        "body_text": chunk_text_unescaped, # For classic style
                        "section_code": section_code,
                        "section_title": section_title,
                        "chunk_num": i + 1,
                        "total_chunks": num_ft_chunks,
                        "full_section_heading": full_section_heading,
                    }

                    if args.ft_rag_style_prompts:
                        current_chunk_system_instruction = RAG_FT_SYSTEM_INSTRUCTION
                        for config_entry in RAG_FT_PROMPT_CONFIG:
                            user_tpl = config_entry["user_template"]
                            model_tpl_effective = config_entry.get("model_template_default")
                            keyword_check_fn = config_entry.get("keyword_check_fn")

                            if keyword_check_fn:
                                if keyword_check_fn(chunk_text_unescaped):
                                    model_tpl_effective = config_entry.get("model_template_kw_found", model_tpl_effective)
                                else:
                                    model_tpl_effective = config_entry.get("model_template_kw_not_found", model_tpl_effective)

                            if user_tpl and model_tpl_effective:
                                user_content = user_tpl.format(**common_format_args)
                                model_content = model_tpl_effective.format(**common_format_args)
                                prompt_content_pairs_for_chunk.append((user_content, model_content))
                            elif user_tpl and not model_tpl_effective:
                                logger.error(f"RAG FT: User template exists but effective model template for user prompt type '{user_tpl.template[:30]}...' is None/Not found in config. Skipping this pair for {section_code} chunk {i+1}.")

                    else: # Classic FT prompts
                        current_chunk_system_instruction = CLASSIC_FT_SYSTEM_INSTRUCTION
                        user_tpl_exp = LC_CLASSIC_USER_EXPLAIN_MULTI if num_ft_chunks > 1 else LC_CLASSIC_USER_EXPLAIN_SINGLE
                        model_tpl_exp = LC_CLASSIC_MODEL_EXPLAIN_MULTI if num_ft_chunks > 1 else LC_CLASSIC_MODEL_EXPLAIN_SINGLE
                        if user_tpl_exp and model_tpl_exp:
                            prompt_content_pairs_for_chunk.append((
                                user_tpl_exp.format(**common_format_args),
                                model_tpl_exp.format(**common_format_args)
                            ))

                        user_tpl_qa = LC_CLASSIC_USER_QA_MULTI if num_ft_chunks > 1 else LC_CLASSIC_USER_QA_SINGLE
                        model_tpl_qa = LC_CLASSIC_MODEL_QA_MULTI if num_ft_chunks > 1 else LC_CLASSIC_MODEL_QA_SINGLE
                        if user_tpl_qa and model_tpl_qa:
                            prompt_content_pairs_for_chunk.append((
                                user_tpl_qa.format(**common_format_args),
                                model_tpl_qa.format(**common_format_args)
                            ))

                    for user_c, model_c in prompt_content_pairs_for_chunk:
                        gemini_ft_record: Dict[str, Any] = {}
                        if current_chunk_system_instruction:
                            gemini_ft_record["systemInstruction"] = {
                                "role": "system",
                                "parts": [{"text": current_chunk_system_instruction}]
                            }
                        gemini_ft_record["contents"] = [
                            {"role": "user", "parts": [{"text": user_c}]},
                            {"role": "model", "parts": [{"text": model_c}]}
                        ]
                        ft_out_f.write(json.dumps(gemini_ft_record, ensure_ascii=False) + "\n")
                        total_finetune_records += 1

                    if all_valid_section_details_for_not_found and NUM_NOT_FOUND_EXAMPLES_PER_CHUNK > 0:
                        other_sections_options = [s for s in all_valid_section_details_for_not_found if s["code"] != section_code]
                        if not other_sections_options and len(all_valid_section_details_for_not_found) >=1 : # If current section is the ONLY valid one, sample from itself (less ideal, but better than nothing)
                            other_sections_options = all_valid_section_details_for_not_found

                        if other_sections_options: # Check again, as it might still be empty if all_valid_section_details_for_not_found was empty
                            num_to_sample = min(NUM_NOT_FOUND_EXAMPLES_PER_CHUNK, len(other_sections_options))
                            if num_to_sample > 0:
                                sampled_other_sections = random.sample(other_sections_options, num_to_sample)
                                for other_sec in sampled_other_sections:
                                    not_found_format_args = {
                                        **common_format_args, # Includes policy_chunk for RAG style
                                        "other_section_topic": other_sec["title"],
                                        "other_section_code": other_sec["code"],
                                        "current_section_code": section_code,
                                        "current_section_title": section_title,
                                    }

                                    system_instruction_nf_text = ""
                                    user_tpl_nf, model_tpl_nf = None, None

                                    if args.ft_rag_style_prompts:
                                        system_instruction_nf_text = RAG_FT_SYSTEM_INSTRUCTION
                                        user_tpl_nf, model_tpl_nf = LC_RAG_USER_NOT_FOUND, LC_RAG_MODEL_NOT_FOUND
                                    else:
                                        system_instruction_nf_text = CLASSIC_FT_SYSTEM_INSTRUCTION
                                        user_tpl_nf, model_tpl_nf = LC_CLASSIC_USER_NOT_FOUND, LC_CLASSIC_MODEL_NOT_FOUND

                                    if user_tpl_nf and model_tpl_nf:
                                        user_content_nf = user_tpl_nf.format(**not_found_format_args)
                                        model_content_nf = model_tpl_nf.format(**not_found_format_args)
                                        gemini_ft_record_nf: Dict[str, Any] = {}
                                        if system_instruction_nf_text:
                                            gemini_ft_record_nf["systemInstruction"] = {
                                                "role": "system",
                                                "parts": [{"text": system_instruction_nf_text}]
                                            }
                                        gemini_ft_record_nf["contents"] = [
                                            {"role": "user", "parts": [{"text": user_content_nf}]},
                                            {"role": "model", "parts": [{"text": model_content_nf}]}
                                        ]
                                        ft_out_f.write(json.dumps(gemini_ft_record_nf, ensure_ascii=False) + "\n")
                                        total_finetune_records += 1

            # --- RAG Data Generation ---
            rag_text_chunks_initial = split_text_into_chunks_robust(
                cleaned_body, args.rag_max_words_per_chunk, args.rag_word_overlap_for_chunks, MIN_RAG_CHUNK_WORD_COUNT
            )

            rag_text_chunks = [] # Final list after potential merge
            if len(rag_text_chunks_initial) >= 2:
                last_chunk = rag_text_chunks_initial[-1]
                penultimate_chunk = rag_text_chunks_initial[-2]
                if len(last_chunk.split()) < MIN_RAG_CHUNK_WORD_COUNT : # User request: Merge small tail
                    merged_chunk_text = penultimate_chunk + "\n\n" + last_chunk # Or just " " if desired
                    if len(merged_chunk_text.split()) <= args.rag_max_words_per_chunk:
                        logger.debug(f"RAG: Merging small tail chunk ({len(last_chunk.split())} words) from section '{section_code}' into previous chunk.")
                        rag_text_chunks.extend(rag_text_chunks_initial[:-2]) # Add all but last two
                        rag_text_chunks.append(merged_chunk_text) # Add merged chunk
                    else:
                        logger.warning(f"RAG: Small tail chunk ({len(last_chunk.split())} words) for section '{section_code}' could not be merged as it would exceed max_words. Keeping separate if valid, or it might be dropped by splitter.")
                        rag_text_chunks.extend(rag_text_chunks_initial) # Keep as is, splitter might drop tail
                else:
                    rag_text_chunks.extend(rag_text_chunks_initial) # Last chunk is fine
            elif rag_text_chunks_initial: # Only one chunk, or zero
                rag_text_chunks.extend(rag_text_chunks_initial)


            if not rag_text_chunks: # If still no chunks after potential merge
                logger.warning(f"RAG Chunking: no valid chunks for '{full_section_heading}' (P{source_page_start}-{source_page_end}). Original words after dedup: {len(cleaned_body.split())}. Skipping RAG for this section.")

            if rag_text_chunks:
                for rag_chunk_idx, rag_chunk_text in enumerate(rag_text_chunks):
                    chunk_id = str(uuid.uuid4())
                    final_keywords_for_metadata: List[str] = []
                    if args.extract_keywords_for_rag:
                        code_kw = [section_code.lower()] if section_code else []
                        title_kws = []
                        if section_title:
                            try:
                                title_tokens = nltk.word_tokenize(section_title)
                                title_kws = [w.lower() for w in title_tokens if w.isalpha() and w.lower() not in NLTK_STOPWORDS and len(w) > 1]
                            except Exception as e :
                                logger.warning(f"NLTK word_tokenize failed for title '{section_title}': {e}. Skipping title keywords for this chunk.")
                        text_kws = extract_simple_keywords(rag_chunk_text, NLTK_STOPWORDS, top_n=args.num_keywords, ngram_range=(1, args.keyword_ngram_max))

                        combined_kws_ordered = []
                        seen_kws = set()
                        for kw_list in [code_kw, title_kws, text_kws]:
                            for kw in kw_list:
                                if kw not in seen_kws:
                                    combined_kws_ordered.append(kw)
                                    seen_kws.add(kw)
                        final_keywords_for_metadata = combined_kws_ordered[:args.num_keywords]

                    imm_parent_code, imm_parent_title = get_immediate_parent_info(section_code, code_to_title_map)
                    full_hierarchical_path_str = get_full_hierarchical_path(section_code, code_to_title_map)

                    metadata = {
                        "source_document_name": source_doc_name_for_metadata,
                        "source_document_path": os.path.abspath(args.input_file),
                        "source_page_start": str(source_page_start),
                        "source_page_end": str(source_page_end), # Added source_page_end
                        "section_code": str(section_code),
                        "section_title": str(section_title),
                        "full_section_heading": full_section_heading,
                        "immediate_parent_code": imm_parent_code,
                        "immediate_parent_title": imm_parent_title,
                        "full_hierarchical_path": full_hierarchical_path_str,
                        "chunk_index_in_section": rag_chunk_idx + 1,
                        "total_chunks_in_section": len(rag_text_chunks),
                        "rag_chunk_word_count": len(rag_chunk_text.split()),
                        "parser_script_version": SCRIPT_VERSION,
                        "keywords": final_keywords_for_metadata
                    }

                    embedding_vector = None

                    if args.generate_embeddings_for_rag and genai and sdk_ok_for_embeddings:
                        try:
                            if not rag_chunk_text.strip():
                                logger.warning(f"Skipping embedding for RAG chunk ID {chunk_id} (section '{section_code}') due to empty content.")
                            else:
                                current_embedding_model = args.vertex_embedding_model # Use the one from args
                                logger.debug(f"Embedding RAG chunk ID {chunk_id} ({len(rag_chunk_text.split())} words) using {current_embedding_model}...")
                                result = genai.embed_content(
                                    model=current_embedding_model,
                                    content=rag_chunk_text,
                                    task_type="RETRIEVAL_DOCUMENT",
                                )
                                if result and isinstance(result.get('embedding'), list) and result['embedding']:
                                    embedding_vector = result['embedding']
                                else:
                                    logger.error(f"Embedding for RAG ID {chunk_id} returned invalid/empty response. Text: '{rag_chunk_text[:60]}...' Result: {result}")
                        except KeyboardInterrupt:
                            logger.warning(f"KeyboardInterrupt during embedding for chunk {chunk_id}. Halting further embeddings.")
                            args.generate_embeddings_for_rag = False # Stop trying for subsequent chunks
                            sdk_ok_for_embeddings = False # Reflect that we stopped
                        except Exception as e:
                            error_msg = f"Error generating embedding for RAG ID {chunk_id} ({len(rag_chunk_text)} chars): {type(e).__name__}: {e}. Text: '{rag_chunk_text[:60]}...'"
                            if GOOGLE_API_EXCEPTIONS_IMPORTED and google_api_exceptions:
                                if isinstance(e, (google_api_exceptions.DeadlineExceeded, google_api_exceptions.ResourceExhausted, google_api_exceptions.InvalidArgument)):
                                    logger.error(error_msg + " This might be due to chunk size, API limits, or invalid input.")
                                elif isinstance(e, google_api_exceptions.GoogleAPIError): # Catch more general Google API errors
                                    logger.error(f"Google API Error (non-specific): {error_msg}")
                                else:
                                    logger.error(f"Unexpected non-Google Exception: {error_msg}")
                            else: # google_api_exceptions not imported, so treat as generic
                                logger.error(f"Generic Exception: {error_msg}")

                    rag_record = {"chunk_id": chunk_id, "text": rag_chunk_text, "embedding": embedding_vector, "metadata": metadata}
                    rag_out_f.write(json.dumps(rag_record, ensure_ascii=False) + "\n")
                    total_rag_chunks += 1

    logger.info(f"✅ Fine-tuning data: {total_finetune_records} records written to {args.output_jsonl_finetune_file} (Style: {'RAG-Style Prompts (with nuance)' if args.ft_rag_style_prompts else 'Classic-Style Prompts'}, Format: Gemini)")
    logger.info(f"✅ RAG data: {total_rag_chunks} chunks written to {args.output_jsonl_rag_file}")

    original_generate_embeddings_flag_default = parser.get_default("generate_embeddings_for_rag") # Get the default value of the flag
    embedding_was_intended_or_active = args.generate_embeddings_for_rag or \
                                    (not args.generate_embeddings_for_rag and sdk_ok_for_embeddings and original_generate_embeddings_flag_default)


    if embedding_was_intended_or_active: # Only log details if embeddings were on, attempted, or default on
        num_embedded_chunks = 0
        if total_rag_chunks > 0 and os.path.exists(args.output_jsonl_rag_file): # Check if file exists before trying to read
            try:
                with open(args.output_jsonl_rag_file, "r", encoding="utf-8") as rag_in_f_check:
                    for line_num, line_content in enumerate(rag_in_f_check):
                        try:
                            record_check = json.loads(line_content)
                            if record_check.get("embedding") and isinstance(record_check.get("embedding"), list) and len(record_check.get("embedding")) > 0:
                                num_embedded_chunks +=1
                        except json.JSONDecodeError:
                            logger.warning(f"Malformed JSON in RAG output check (line {line_num+1}): {line_content[:100]}...")
            except FileNotFoundError: # Should be caught by os.path.exists, but defensive
                    logger.error(f"RAG output file {args.output_jsonl_rag_file} not found for embedding count check, though it should exist.")


        if genai and sdk_ok_for_embeddings and args.generate_embeddings_for_rag: # If SDK is OK and embeddings were *still* on at the end
            if num_embedded_chunks > 0:
                logger.info(f"Successfully embedded {num_embedded_chunks}/{total_rag_chunks} RAG chunks (model '{args.vertex_embedding_model}').")
                if num_embedded_chunks < total_rag_chunks:
                     logger.warning(f"{total_rag_chunks - num_embedded_chunks} RAG chunks MISSING embeddings. Check logs for API errors or empty chunks.")
            elif total_rag_chunks > 0 : # No chunks embedded despite trying
                logger.warning(f"0/{total_rag_chunks} RAG chunks embedded. Embeddings were active with '{args.vertex_embedding_model}'. Check logs for API errors/auth or empty chunks.")
            elif total_rag_chunks == 0:
                logger.info(f"RAG Embeddings enabled ('{args.vertex_embedding_model}'), but no RAG chunks were produced to embed.")

        elif not sdk_ok_for_embeddings and original_generate_embeddings_flag_default: # SDK failed, but embeddings were originally intended (default=True)
            logger.warning(f"Embeddings for RAG ('{args.vertex_embedding_model}') were REQUESTED/DEFAULTED ON but FAILED due to SDK/Auth issues. {total_rag_chunks} RAG chunks lack embeddings.")
        elif not args.generate_embeddings_for_rag and original_generate_embeddings_flag_default and sdk_ok_for_embeddings: # Embeddings were turned off mid-process (e.g. Ctrl+C) but SDK was initially okay
            logger.info(f"Embedding for RAG ('{args.vertex_embedding_model}') was default ON but became DISABLED (user flag/runtime issue, but SDK was OK). {total_rag_chunks} RAG chunks lack embeddings.")

    else: # Embeddings were explicitly disabled from the start or due to no SDK
        logger.info(f"Embedding for RAG disabled by user or SDK unavailability. {total_rag_chunks} RAG chunks in '{args.output_jsonl_rag_file}' have no embeddings.")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        if load_dotenv(): logger.info("Loaded .env file if present.")
        else: logger.debug(".env file not found/loaded (ok if vars set globally).")
    except ImportError:
        logger.info("python-dotenv not installed. Skipping .env load. Set env vars manually if needed.")

    parser = argparse.ArgumentParser(
        description=f"CJIS Policy PDF Parser ({SCRIPT_VERSION})\n"
                    "Parses a PDF policy document into JSONL for Gemini fine-tuning and RAG.\n"
                    "Fine-tuning output is formatted for Gemini (systemInstruction + contents).\n"
                    "RAG-style fine-tuning model completions are nuanced based on question type and content heuristics.\n"
                    "Text Extraction: PyMuPDF\n"
                    "Table Extraction: Camelot (requires Ghostscript, opencv-python, pandas, tk)\n"
                    "Prompt Generation: LangChain\n"
                    "Keyword Extraction: NLTK (optional for RAG metadata)\n"
                    "Hierarchical Metadata: Added immediate parent and full hierarchical path for RAG.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    io_group = parser.add_argument_group('Input/Output Files')
    io_group.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE, help="Path to input PDF.")
    io_group.add_argument("--output_jsonl_finetune_file", type=str, default=DEFAULT_OUTPUT_JSONL_FINETUNE_FILE, help="Path for fine-tuning JSONL.")
    io_group.add_argument("--output_jsonl_rag_file", type=str, default=DEFAULT_OUTPUT_JSONL_RAG_FILE, help="Path for RAG JSONL.")

    pdf_opts_group = parser.add_argument_group('PDF Parsing Options')
    pdf_opts_group.add_argument("--page_start", type=int, default=DEFAULT_PAGE_START, help=f"Start page (1-indexed). Default: {DEFAULT_PAGE_START}.")
    pdf_opts_group.add_argument("--page_end", type=int, default=DEFAULT_PAGE_END, help=f"End page (inclusive, 1-indexed). Default: {DEFAULT_PAGE_END} (to end).")
    pdf_opts_group.add_argument("--header_footer_margin", type=int, default=35, help="Pixel margin for H/F detection. Default: 35.")
    pdf_opts_group.add_argument("--extract_tables", action=argparse.BooleanOptionalAction, default=True, help="Attempt to extract tables using Camelot (default: True). Use --no-extract-tables to disable.")


    chunk_opts_group = parser.add_argument_group('Text Chunking Options')
    chunk_opts_group.add_argument("--ft_max_words_per_chunk", type=int, default=DEFAULT_FT_MAX_WORDS_PER_CHUNK, help=f"Max words/chunk for FT data. Default: {DEFAULT_FT_MAX_WORDS_PER_CHUNK}.")
    chunk_opts_group.add_argument("--ft_word_overlap_for_chunks", type=int, default=DEFAULT_FT_WORD_OVERLAP_FOR_CHUNKS, help=f"Word overlap for FT chunks. Default: {DEFAULT_FT_WORD_OVERLAP_FOR_CHUNKS}.")
    chunk_opts_group.add_argument("--rag_max_words_per_chunk", type=int, default=DEFAULT_RAG_MAX_WORDS_PER_CHUNK, help=f"Max words/chunk for RAG data. Default: {DEFAULT_RAG_MAX_WORDS_PER_CHUNK}.")
    chunk_opts_group.add_argument("--rag_word_overlap_for_chunks", type=int, default=DEFAULT_RAG_WORD_OVERLAP_FOR_CHUNKS, help=f"Word overlap for RAG chunks. Default: {DEFAULT_RAG_WORD_OVERLAP_FOR_CHUNKS}.")

    ai_opts_group = parser.add_argument_group('AI, Fine-tuning, and RAG Options')
    ai_opts_group.add_argument("--vertex_embedding_model", type=str, default=VERTEX_EMBEDDING_MODEL, help=f"Vertex AI/Google AI Studio embedding model. Default: {VERTEX_EMBEDDING_MODEL}.")
    ai_opts_group.add_argument("--generate_embeddings_for_rag", action=argparse.BooleanOptionalAction, default=True, help="Generate embeddings for RAG (default: True). Use --no-generate-embeddings-for-rag to disable.")
    ai_opts_group.add_argument("--ft_rag_style_prompts", action=argparse.BooleanOptionalAction, default=True, help="Use RAG-style fine-tuning prompts (Policy Text in user query). Default: True. Use --no-ft-rag-style-prompts for classic style.")
    ai_opts_group.add_argument("--num_not_found_examples_per_chunk", type=int, default=NUM_NOT_FOUND_EXAMPLES_PER_CHUNK, help=f"Num 'not found' FT examples per content chunk. Default: {NUM_NOT_FOUND_EXAMPLES_PER_CHUNK}")
    ai_opts_group.add_argument("--extract_keywords_for_rag", action=argparse.BooleanOptionalAction, default=True, help="Extract keywords for RAG metadata (default: True). Use --no-extract-keywords-for-rag to disable.")
    ai_opts_group.add_argument("--num_keywords", type=int, default=10, help=f"Number of keywords to include in RAG metadata if enabled. Default: 10.")
    ai_opts_group.add_argument("--keyword_ngram_max", type=int, default=2, choices=[1,2,3], help=f"Max N for N-gram keyword extraction from text (1=unigrams, 2=uni+bigrams, 3=uni+bi+trigrams). Default: 2.")


    general_group = parser.add_argument_group('General Script Behavior')
    general_group.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logging.root.handlers: handler.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled.")
    elif logger.level != logging.INFO : # Ensure it's INFO if not debug
        logger.setLevel(logging.INFO)
        for handler in logging.root.handlers: handler.setLevel(logging.INFO)


    if args.generate_embeddings_for_rag:
        if not GOOGLE_API_KEY and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.warning("IMPORTANT: NEITHER GOOGLE_API_KEY NOR GOOGLE_APPLICATION_CREDENTIALS set. "
                           "Embedding calls LIKELY TO FAIL unless in pre-configured GCP env (e.g., Vertex AI Workbench).")
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            gac_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if gac_path and os.path.exists(gac_path):
                logger.info(f"Using GOOGLE_APPLICATION_CREDENTIALS: {gac_path}")
            elif gac_path: # Path is set but does not exist
                 logger.error(f"ERROR: GOOGLE_APPLICATION_CREDENTIALS='{gac_path}', but file NOT FOUND.")
        elif GOOGLE_API_KEY: # Only log if GAC wasn't successfully used
            logger.info("Using GOOGLE_API_KEY for Google Generative AI SDK.")
        logger.info(f"Embeddings for RAG data will be ATTEMPTED with model: {args.vertex_embedding_model}")
    else:
        logger.info("Embedding for RAG data IS DISABLED (--no-generate-embeddings-for-rag).")

    if args.page_start > args.page_end and args.page_end != parser.get_default("page_end"): # Check if page_end was explicitly set to be smaller
        logger.error(f"Error: --page_start ({args.page_start}) > --page_end ({args.page_end}) is invalid. Exiting.")
        exit(1)
    if args.page_start <= 0:
        logger.warning(f"Warning: --page_start ({args.page_start}) corrected to 1.")
        args.page_start = 1

    if not PYMUPDF_IMPORTED:
        logger.critical("PyMuPDF (fitz) is essential and not imported successfully. Exiting.")
        exit(1)
    if not LANGCHAIN_IMPORTED and PromptTemplate is None : # Check PromptTemplate as well, as LANGCHAIN_IMPORTED might be true but PromptTemplate is None hypothetically
        logger.critical("LangChain is essential for prompt generation and not imported successfully. Exiting.")
        exit(1)
    if args.extract_tables and not CAMELOT_IMPORTED:
        logger.warning("Camelot for table extraction is not available. Tables will not be extracted. To silence this, use --no-extract-tables.")
    if args.extract_keywords_for_rag: # Confirm NLTK resources if keyword extraction is enabled
        try:
            nltk.data.find('corpora/stopwords.zip', paths=[NLTK_DATA_PATH_SELECTED] + nltk.data.path)
            nltk.data.find('tokenizers/punkt.zip', paths=[NLTK_DATA_PATH_SELECTED] + nltk.data.path)
            nltk.data.find('taggers/averaged_perceptron_tagger.zip', paths=[NLTK_DATA_PATH_SELECTED] + nltk.data.path)
        except LookupError as e:
            logger.error(f"NLTK resources required for keyword extraction (stopwords, punkt, averaged_perceptron_tagger) are missing and could not be downloaded: {e}. "
                         "Please install them manually or disable keyword extraction with --no-extract-keywords-for-rag. Functionality will be impaired.")


    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("\n🛑 Script interrupted by user (Ctrl+C). Partial outputs may exist. Exiting.")
        try: exit(130) # Standard exit code for SIGINT
        except SystemExit: os._exit(130) # If SystemExit is already being handled
    except Exception as e:
        logger.critical(f"🆘 Unhandled critical error in main execution: {type(e).__name__}: {e}", exc_info=True)
        exit(1)

