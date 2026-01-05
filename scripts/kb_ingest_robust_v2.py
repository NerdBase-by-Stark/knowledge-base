#!/usr/bin/env python3
"""
ROBUST KNOWLEDGE BASE INGESTION
================================
Production-grade KB ingestion with:
- Convergence detection (run until entity count stabilizes)
- Thinking model support (Qwen3, DeepSeek-R1, etc.)
- Built-in entity cleanup and stoplist
- Multi-pass extraction with focused prompts
- Checkpoint/resume capability
- Config-driven operation

Usage:
    # Basic - point at directory, let it run until convergence
    python kb_ingest_robust.py --source /path/to/docs --collection my_kb

    # With specific model
    python kb_ingest_robust.py --source /path/to/docs --collection my_kb --model llama3.1:8b

    # Resume from checkpoint
    python kb_ingest_robust.py --resume /path/to/checkpoint.json

    # Dry run (no storage, just extract and show stats)
    python kb_ingest_robust.py --source /path/to/docs --dry-run

Author: Built with Claude Code
"""

import json
import time
import hashlib
import re
import argparse
import signal
import atexit
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from enum import Enum
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import ollama
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
    from rich.table import Table
    from rich.panel import Panel
    HAS_DEPS = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install ollama qdrant-client rich")
    HAS_DEPS = False

console = Console()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class IngestionConfig:
    """Configuration for KB ingestion."""

    # Source settings
    source_dir: Path = None
    file_patterns: List[str] = field(default_factory=lambda: ["*.md", "*.txt"])

    # Model settings
    model: str = "llama3.1:8b"
    temperature: float = 0.3
    max_tokens: int = 400
    timeout_seconds: int = 120
    max_retries: int = 3

    # Chunking settings
    chunk_size: int = 512  # words
    chunk_overlap: int = 75  # words

    # Convergence settings
    convergence_threshold: float = 0.05  # Stop when new entities < 5%
    max_runs: int = 10  # Safety limit
    min_runs: int = 2  # At least 2 runs

    # Entity settings
    entity_stoplist: Set[str] = field(default_factory=lambda: {
        # Lua/programming primitives
        "true", "false", "nil", "string", "table", "function", "number",
        "boolean", "userdata", "thread", "integer", "float",
        # Lua globals and internals
        "_g", "_env", "_version", "self", "arg", "args",
        # Short variable names (common noise)
        "i", "j", "k", "n", "v", "x", "y", "z", "a", "b", "c",
        "v1", "v2", "v3", "bt", "fn", "cb", "ok", "err",
        # Generic terms
        "value", "values", "data", "type", "types", "name", "names",
        "item", "items", "list", "array", "object", "objects",
        "key", "keys", "index", "result", "results", "output", "input",
        # Programming language names (too generic)
        "c", "c++", "java", "python", "javascript",
        # Boilerplate
        "copyright", "support", "resources", "help", "documentation",
        "example", "examples", "note", "notes", "warning", "tip",
        "see also", "related", "overview", "introduction", "summary",
        # Common noise
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "[[]]", "[]", "{}", "()", "...", "---", "n/a", "tbd", "todo",
    })

    # Normalize these entity type variations to standard types
    # Standard types: PRODUCT, TECHNOLOGY, CONCEPT, PARAMETER, FUNCTION,
    #                 CLASS, EVENT, RELATIONSHIP
    entity_type_normalization: Dict[str, str] = field(default_factory=lambda: {
        # Function variations
        "FUNC": "FUNCTION",
        "METHOD": "FUNCTION",
        "CALLBACK": "FUNCTION",
        "HANDLER": "FUNCTION",
        "EVENT HANDLER": "FUNCTION",
        "API CALL": "FUNCTION",
        "API": "FUNCTION",
        "HOOK": "FUNCTION",
        "COMMAND": "FUNCTION",
        "PROPERTY ACCESSOR": "FUNCTION",

        # Parameter variations
        "PARAM": "PARAMETER",
        "PROPERTY": "PARAMETER",
        "PROP": "PARAMETER",
        "CONFIG": "PARAMETER",
        "SETTING": "PARAMETER",
        "OPTION": "PARAMETER",
        "ARRAY PROPERTY": "PARAMETER",
        "VARIABLE": "PARAMETER",
        "CONSTANT": "PARAMETER",

        # Technology variations
        "TECH": "TECHNOLOGY",
        "PROTOCOL": "TECHNOLOGY",
        "PROTOCOL/INTERFACE": "TECHNOLOGY",
        "STANDARD": "TECHNOLOGY",
        "FORMAT": "TECHNOLOGY",
        "LANGUAGE": "TECHNOLOGY",
        "PROGRAMMING LANGUAGE": "TECHNOLOGY",

        # Product variations
        "PROD": "PRODUCT",
        "COMPONENT": "PRODUCT",
        "DEVICE": "PRODUCT",
        "SOFTWARE": "PRODUCT",
        "SOFTWARE COMPONENT": "PRODUCT",
        "HARDWARE": "PRODUCT",
        "CONTROLLER": "PRODUCT",
        "MODEL": "PRODUCT",
        "WINDOW": "PRODUCT",
        "INTERFACE": "PRODUCT",

        # Concept variations
        "TERM": "CONCEPT",
        "PATTERN": "CONCEPT",
        "IDEA": "CONCEPT",
        "TERMINOLOGY": "CONCEPT",
        "TECHNICAL_TERM": "CONCEPT",
        "BEST PRACTICE": "CONCEPT",
        "ARCHITECTURAL TERM": "CONCEPT",
        "DATA STRUCTURE": "CONCEPT",
        "ARRAY": "CONCEPT",
        "OBJECT": "CONCEPT",
        "UNIT OF MEASUREMENT": "CONCEPT",
        "FILE TYPE": "CONCEPT",
        "EXTENSION": "CONCEPT",
        "ERROR CODE": "CONCEPT",
        "ERROR_MESSAGE": "CONCEPT",
        "OTHER": "CONCEPT",

        # Class variations
        "PARENT CLASS": "CLASS",
        "CHILD CLASS": "CLASS",
        "BASE CLASS": "CLASS",

        # Relationship variations
        "REL": "RELATIONSHIP",
        "DEPENDENCY": "RELATIONSHIP",
        "DEPENDENCY/RESOURCE": "RELATIONSHIP",
        "LINK": "RELATIONSHIP",
        "RELATED COMPONENT": "RELATIONSHIP",
        "RELATED RESOURCE": "RELATIONSHIP",

        # Resource types -> CONCEPT (generic)
        "DOCUMENTATION": "CONCEPT",
        "REFERENCE MANUAL": "CONCEPT",
        "BOOK": "CONCEPT",
        "TRAINING/EDUCATION RESOURCE": "CONCEPT",
    })

    # Output settings
    collection_name: str = None
    output_dir: Path = None
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_auth: Tuple[str, str] = ("neo4j", "agentmemory123")

    # Processing settings
    batch_size: int = 10
    checkpoint_interval: int = 20  # chunks between checkpoints

    # Flags
    dry_run: bool = False
    skip_storage: bool = False
    skip_graph: bool = False
    verbose: bool = False

    # Cleanup settings
    cleanup_enabled: bool = True
    cleanup_model: str = None  # If None, uses same model; set to thinking model for better results
    cleanup_batch_size: int = 30

    # Noise patterns for rule-based cleanup (regex)
    noise_patterns: List[str] = field(default_factory=lambda: [
        # Single chars and primitives
        r'^[a-z]$',                    # Single letters
        r'^\d+$',                      # Pure numbers
        r'^\.+$',                      # Dots only
        r'^_+$',                       # Underscores only
        r'^[\[\]\(\)\{\}]+$',          # Brackets only
        r'^\\[nrtv]$',                 # Escape characters like \n, \t

        # Template/placeholder patterns
        r'^_\w+_$',                    # _var_, _name_, etc.
        r'^\w\[_\w+_\]',               # s[_n_], t[_i_], etc.
        r'^<\w+>$',                    # <type>, <name>, etc.

        # Too verbose (likely descriptions, not entities)
        r'^.{60,}$',                   # Over 60 chars - too long
        r'\s{2,}',                     # Multiple spaces - likely sentence
        r'^\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+',  # 6+ words - description

        # Behavior descriptions (not entities)
        r'^returns?\s+',               # "returns nil", "return value"
        r'\s+returns?\s*$',            # "function returns"
        r'unwinds?\s+',                # "error unwinds the stack"
        r'^(is|are|was|were|has|have)\s+',  # Starting with verbs
        r'\s+(is|are|was|were)\s+',    # Contains "is/are" mid-sentence

        # Meta/documentation terms
        r'^code\s+examples?',          # Meta-descriptions
        r'^(the|a|an)\s+',             # Starting with articles
        r'^(this|that|these|those)\s+', # Demonstratives
        r'^\d+\.\d+',                  # Version-like numbers (5.3, 2.1)
        r'^https?://',                 # URLs
        r'^\s*$',                      # Empty/whitespace

        # Operators and syntax
        r'^[<>]=?$',                   # Comparison operators
        r'^[+\-*/]=?$',                # Math operators
        r'^\.\.\.$',                   # Ellipsis
        r'^==$',                       # Equality
        r'^~=$',                       # Lua not equal
        r'^\w+\s+operator$',           # "== operator", "unary operator"

        # Generic/noise
        r'^n/a$',                      # N/A
        r'^\w\.\w\.\w',                # Single letter chains like a.b.c
        r'^(yes|no|true|false|nil)$',  # Boolean-like
        r'^(legal|copyright|license)', # Legal boilerplate
        r'^(todo|fixme|hack|xxx)',     # Code comments
        r'^(see|note|warning|tip)(\s|:)',  # Documentation markers

        # Lua-specific noise
        r'^lua\s+\d',                  # "lua 5.3", "lua 5.4"
        r'^(statement|expression|block)s?$',  # Grammar terms
        r'^\w+\s+statement$',          # "break statement", "return statement"
        r'^\w+\s+expression$',         # "binary expression"
    ])


# =============================================================================
# THINKING MODEL SUPPORT
# =============================================================================

# Known thinking models and their patterns
THINKING_MODELS = {
    "qwen3": {"pattern": r"<think>.*?</think>", "flags": re.DOTALL},
    "deepseek-r1": {"pattern": r"<think>.*?</think>", "flags": re.DOTALL},
    "deepseek-reasoner": {"pattern": r"<think>.*?</think>", "flags": re.DOTALL},
    "o1": {"pattern": r"", "flags": 0},  # OpenAI o1 hides thinking
    "claude-3-opus-thinking": {"pattern": r"<thinking>.*?</thinking>", "flags": re.DOTALL},
}


def is_thinking_model(model_name: str) -> bool:
    """Check if model is a known thinking model."""
    model_lower = model_name.lower()
    for known in THINKING_MODELS:
        if known in model_lower:
            return True
    # Also check for common thinking indicators
    if any(x in model_lower for x in ["think", "reason", "cot", "o1"]):
        return True
    return False


def extract_response_from_thinking(text: str, model_name: str) -> str:
    """
    Extract the actual response from a thinking model's output.

    Thinking models often output:
    <think>
    ... reasoning ...
    </think>

    Actual response here

    Some models (like Qwen3) may have junk before the think tag or
    may not properly close the tag.
    """
    if not text:
        return ""

    # First, try to find content AFTER </think> tag (the actual response)
    think_end_match = re.search(r'</think>\s*(.+)', text, re.DOTALL | re.IGNORECASE)
    if think_end_match:
        after_think = think_end_match.group(1).strip()
        if after_think:
            # Strip markdown code blocks if present (Qwen3 wraps JSON in ```json ... ```)
            after_think = re.sub(r'^```(?:json)?\s*', '', after_think)
            after_think = re.sub(r'\s*```$', '', after_think)
            return after_think.strip()

    # If no closing tag, try to extract JSON from anywhere in the response
    # This handles cases where thinking is incomplete or malformed
    json_match = re.search(r'\[[\s\S]*?\]', text)
    if json_match:
        return json_match.group(0)

    model_lower = model_name.lower()

    # Try known patterns (remove everything inside think tags)
    for model_key, config in THINKING_MODELS.items():
        if model_key in model_lower and config["pattern"]:
            pattern = re.compile(config["pattern"], config["flags"])
            cleaned = pattern.sub("", text)
            if cleaned.strip():
                return cleaned.strip()

    # Generic thinking tag removal
    patterns = [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<reason>.*?</reason>",
        r"<reasoning>.*?</reasoning>",
        r"<thought>.*?</thought>",
        r"\[thinking\].*?\[/thinking\]",
    ]

    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.DOTALL | re.IGNORECASE)

    return result.strip() if result.strip() else text


# =============================================================================
# ENTITY EXTRACTION PROMPTS
# =============================================================================

# Multi-pass extraction prompts - each focuses on different entity types
EXTRACTION_PASSES = [
    {
        "name": "standard",
        "focus": "General technical entities",
        "prompt": """Extract technical entities from this documentation. Return ONLY a valid JSON array.

Entity types: PRODUCT, TECHNOLOGY, CONCEPT, PARAMETER, FUNCTION, CLASS, EVENT

Text:
---
{chunk}
---

Return JSON like: [{{"name": "EntityName", "type": "TYPE"}}]
Extract up to 10 important entities. JSON array only, no explanation:"""
    },
    {
        "name": "functions",
        "focus": "Functions, methods, and API calls",
        "prompt": """Extract ALL function names, method names, callbacks, and API calls from this text.

Text:
---
{chunk}
---

Focus on: function names, method calls, callbacks, event handlers, API endpoints, hooks.
Return JSON like: [{{"name": "FunctionName", "type": "FUNCTION"}}]
JSON array only:"""
    },
    {
        "name": "parameters",
        "focus": "Parameters, properties, and configurations",
        "prompt": """Extract ALL parameters, properties, settings, and configuration options from this text.

Text:
---
{chunk}
---

Focus on: function parameters, object properties, config options, default values, settings.
Return JSON like: [{{"name": "paramName", "type": "PARAMETER"}}]
JSON array only:"""
    },
    {
        "name": "concepts",
        "focus": "Programming concepts and terminology",
        "prompt": """Extract programming concepts, design patterns, and technical terminology from this text.

Text:
---
{chunk}
---

Focus on: programming patterns, concepts, terminology, best practices, architectural terms.
Return JSON like: [{{"name": "ConceptName", "type": "CONCEPT"}}]
JSON array only:"""
    },
    {
        "name": "relationships",
        "focus": "Relationships and dependencies",
        "prompt": """Extract entities that relate to, connect with, inherit from, or depend on other entities.

Text:
---
{chunk}
---

Focus on: parent classes, dependencies, related components, protocols, interfaces.
Return JSON like: [{{"name": "EntityName", "type": "RELATIONSHIP"}}]
JSON array only:"""
    },
    {
        "name": "catchall",
        "focus": "Remaining technical terms",
        "prompt": """Final pass: Extract ANY remaining technical terms, identifiers, or names not yet captured.

Text:
---
{chunk}
---

Look for: variable names, constants, enums, error codes, file types, any technical term.
Return JSON like: [{{"name": "term", "type": "CONCEPT"}}]
JSON array only:"""
    },
]

# Context and summary prompts
CONTEXT_PROMPT = """Write a brief context statement (2-3 sentences) for this documentation chunk.
Document: {doc_title}

Chunk:
---
{chunk}
---

Context statement (2-3 sentences only):"""

SUMMARY_PROMPT = """Summarize this technical documentation in ONE sentence:
---
{chunk}
---

One sentence summary:"""

QA_PROMPT = """What specific questions can this documentation answer? List exactly 3 questions.
---
{chunk}
---

Questions (one per line, no numbering):"""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Entity:
    """A normalized entity."""
    name: str
    type: str

    def to_dict(self) -> Dict:
        return {"name": self.name, "type": self.type}

    @property
    def key(self) -> str:
        """Unique key for deduplication."""
        return f"{self.name.lower().strip()}:{self.type}"


@dataclass
class Chunk:
    """A document chunk with extracted metadata."""
    chunk_id: str
    doc_name: str
    doc_title: str
    chunk_index: int
    content: str
    context: str = ""
    summary: str = ""
    entities: List[Entity] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "doc_title": self.doc_title,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "context": self.context,
            "summary": self.summary,
            "entities": [e.to_dict() for e in self.entities],
            "questions": self.questions,
        }


@dataclass
class RunStats:
    """Statistics for a single run."""
    run_number: int
    new_entities: int
    total_extracted: int
    cumulative_unique: int
    convergence_pct: float
    time_seconds: float
    pass_breakdown: List[Dict] = field(default_factory=list)


@dataclass
class IngestionState:
    """Checkpoint state for resume capability."""
    config: Dict
    chunks: List[Dict]
    known_entities: List[str]
    run_stats: List[Dict]
    current_run: int
    current_pass: int
    current_chunk: int
    started_at: str
    last_checkpoint: str

    def save(self, path: Path):
        """Save checkpoint to file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "IngestionState":
        """Load checkpoint from file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# CORE INGESTION ENGINE
# =============================================================================

class IngestionEngine:
    """The main ingestion engine."""

    # Class-level reference for signal handler
    _instance = None

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.known_entities: Set[str] = set()
        self.chunks: List[Chunk] = []
        self.run_stats: List[RunStats] = []
        self.is_thinking_model = is_thinking_model(config.model)
        self._shutting_down = False
        self._current_run = 0
        self._current_pass = 0
        self._current_chunk = 0
        self._started_at = datetime.now().isoformat()

        # Initialize progress file
        self.progress_file = (config.output_dir or Path.home()) / "kb_ingest_progress.log"

        # Set up signal handlers for graceful shutdown
        IngestionEngine._instance = self
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        atexit.register(self._emergency_checkpoint)

        if self.is_thinking_model:
            self.log(f"Detected thinking model: {config.model}")

    def _handle_shutdown(self, signum, frame):
        """Handle SIGTERM/SIGINT gracefully."""
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        self.log(f"\n{'='*60}", "WARN")
        self.log(f"Received {sig_name} - saving checkpoint and shutting down...", "WARN")
        self.log(f"{'='*60}", "WARN")

        self._shutting_down = True

        # Save emergency checkpoint
        self._emergency_checkpoint()

        self.log(f"Checkpoint saved. Safe to restart with --resume", "WARN")
        sys.exit(0)

    def _emergency_checkpoint(self):
        """Save checkpoint on any exit."""
        if not self.config.output_dir or not self.chunks:
            return

        try:
            checkpoint_path = self.config.output_dir / "checkpoint_emergency.json"
            state = IngestionState(
                config=asdict(self.config),
                chunks=[c.to_dict() for c in self.chunks],
                known_entities=list(self.known_entities),
                run_stats=[asdict(s) for s in self.run_stats],
                current_run=self._current_run,
                current_pass=self._current_pass,
                current_chunk=self._current_chunk,
                started_at=self._started_at,
                last_checkpoint=datetime.now().isoformat(),
            )
            state.save(checkpoint_path)
            console.print(f"[yellow]Emergency checkpoint: {checkpoint_path}[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to save emergency checkpoint: {e}[/red]")

    @property
    def should_stop(self) -> bool:
        """Check if shutdown was requested."""
        return self._shutting_down

    def log(self, msg: str, level: str = "INFO"):
        """Log a message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {msg}"

        if self.config.verbose or level in ["ERROR", "WARN"]:
            console.print(formatted)

        with open(self.progress_file, "a") as f:
            f.write(formatted + "\n")

    def call_llm(self, prompt: str, max_tokens: int = None) -> str:
        """Call the LLM with retry logic, timeout, and thinking model support."""
        max_tokens = max_tokens or self.config.max_tokens

        # Thinking models need more tokens for reasoning + response
        if self.is_thinking_model:
            max_tokens = max(max_tokens * 3, 1000)

        def _do_call():
            """Inner function for timeout wrapper."""
            return ollama.chat(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": self.config.temperature,
                    "num_predict": max_tokens,
                }
            )

        for attempt in range(self.config.max_retries):
            try:
                # Use ThreadPoolExecutor for timeout support
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_do_call)
                    try:
                        response = future.result(timeout=self.config.timeout_seconds)
                    except FuturesTimeoutError:
                        self.log(f"LLM call timed out after {self.config.timeout_seconds}s (attempt {attempt + 1})", "WARN")
                        if attempt < self.config.max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            return f"ERROR: Timeout after {self.config.timeout_seconds}s"

                text = response["message"]["content"]

                # Handle thinking models
                if self.is_thinking_model:
                    text = extract_response_from_thinking(text, self.config.model)

                return text

            except FuturesTimeoutError:
                # Already handled above, but just in case
                pass
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait = 2 ** attempt  # Exponential backoff
                    self.log(f"LLM call failed (attempt {attempt + 1}), retrying in {wait}s: {e}", "WARN")
                    time.sleep(wait)
                else:
                    self.log(f"LLM call failed after {self.config.max_retries} attempts: {e}", "ERROR")
                    return f"ERROR: {e}"

        return "ERROR: Max retries exceeded"

    def normalize_entity_type(self, etype: str) -> str:
        """
        Normalize entity type using explicit mapping and fuzzy matching.

        Falls back to keyword-based matching for types not in the explicit map.
        """
        etype = etype.strip().upper()

        # Try explicit mapping first
        if etype in self.config.entity_type_normalization:
            return self.config.entity_type_normalization[etype]

        # If already a standard type, return as-is
        standard_types = {"PRODUCT", "TECHNOLOGY", "CONCEPT", "PARAMETER",
                         "FUNCTION", "CLASS", "EVENT", "RELATIONSHIP"}
        if etype in standard_types:
            return etype

        # Fuzzy matching based on keywords
        etype_lower = etype.lower()

        # Function-like types
        if any(k in etype_lower for k in ["func", "method", "call", "handler",
                                           "callback", "hook", "command", "api"]):
            return "FUNCTION"

        # Parameter-like types
        if any(k in etype_lower for k in ["param", "prop", "setting", "config",
                                           "option", "variable", "const", "value"]):
            return "PARAMETER"

        # Technology-like types
        if any(k in etype_lower for k in ["tech", "protocol", "standard", "format",
                                           "language", "encoding"]):
            return "TECHNOLOGY"

        # Product-like types
        if any(k in etype_lower for k in ["product", "component", "device", "software",
                                           "hardware", "tool", "system", "module"]):
            return "PRODUCT"

        # Relationship-like types
        if any(k in etype_lower for k in ["relation", "depend", "link", "connect",
                                           "inherit", "parent", "child", "base"]):
            return "RELATIONSHIP"

        # Class-like types
        if any(k in etype_lower for k in ["class", "object", "struct", "interface"]):
            return "CLASS"

        # Event-like types
        if any(k in etype_lower for k in ["event", "signal", "trigger", "notification"]):
            return "EVENT"

        # Default to CONCEPT
        return "CONCEPT"

    def parse_entities(self, text: str) -> List[Entity]:
        """Parse entities from LLM output with normalization."""
        if text.startswith("ERROR:"):
            return []

        entities = []

        try:
            # Find JSON array in response
            start = text.find("[")
            end = text.rfind("]") + 1

            if start >= 0 and end > start:
                json_str = text[start:end]
                raw_entities = json.loads(json_str)

                # Handle case where LLM returns a single dict instead of list
                if isinstance(raw_entities, dict):
                    raw_entities = [raw_entities]

                # Ensure it's a list
                if not isinstance(raw_entities, list):
                    return []

                for e in raw_entities:
                    # Skip non-dict items in list
                    if not isinstance(e, dict):
                        continue

                    name = str(e.get("name", "")).strip()
                    etype = str(e.get("type", "CONCEPT")).strip().upper()

                    # Skip empty or stoplist
                    if not name or len(name) < 2:
                        continue
                    if name.lower() in self.config.entity_stoplist:
                        continue

                    # Normalize type (uses both explicit mapping and fuzzy matching)
                    etype = self.normalize_entity_type(etype)

                    # Skip if entity name is too generic
                    if len(name) < 3 and etype not in ["FUNCTION", "PARAMETER"]:
                        continue

                    # Skip copyright and boilerplate patterns
                    if any(x in name.lower() for x in ["copyright", "©", "all rights", "reserved"]):
                        continue

                    entities.append(Entity(name=name, type=etype))

        except json.JSONDecodeError:
            # Try to extract entities from non-JSON format
            pass
        except Exception as e:
            self.log(f"Entity parse error: {e}", "WARN")

        return entities

    def parse_questions(self, text: str) -> List[str]:
        """Parse questions from LLM output."""
        if text.startswith("ERROR:"):
            return []

        questions = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering
            if len(line) > 2 and line[0].isdigit() and line[1] in ".)":
                line = line[2:].strip()
            elif line.startswith("-"):
                line = line[1:].strip()
            if line and "?" in line:
                questions.append(line)

        return questions[:3]

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text into overlapping segments."""
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + self.config.chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start = end - self.config.chunk_overlap

        return chunks

    def load_documents(self) -> List[Chunk]:
        """Load and chunk all documents from source directory."""
        if not self.config.source_dir or not self.config.source_dir.exists():
            raise ValueError(f"Source directory not found: {self.config.source_dir}")

        all_chunks = []

        for pattern in self.config.file_patterns:
            for doc_path in sorted(self.config.source_dir.glob(pattern)):
                doc_name = doc_path.name
                doc_title = doc_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")

                try:
                    text = doc_path.read_text(encoding="utf-8")
                except Exception as e:
                    self.log(f"Failed to read {doc_path}: {e}", "WARN")
                    continue

                chunks = self.chunk_text(text)

                for i, chunk_text in enumerate(chunks):
                    chunk_id = hashlib.md5(
                        f"{doc_name}:{i}:{chunk_text[:50]}".encode()
                    ).hexdigest()[:12]

                    all_chunks.append(Chunk(
                        chunk_id=chunk_id,
                        doc_name=doc_name,
                        doc_title=doc_title,
                        chunk_index=i,
                        content=chunk_text,
                    ))

        self.log(f"Loaded {len(all_chunks)} chunks from {len(set(c.doc_name for c in all_chunks))} documents")
        return all_chunks

    def strip_llm_preamble(self, text: str) -> str:
        """Remove common LLM preambles from responses."""
        if not text:
            return text

        # Common preamble patterns to strip
        preambles = [
            r"^here is a?\s*(brief\s*)?(context|summary|statement)[^:]*:\s*",
            r"^(the\s+)?(context|summary)\s+(is|for this)[^:]*:\s*",
            r"^(this|the)\s+(chunk|text|documentation)\s+(is|describes|covers)[^.]*\.\s*",
            r"^based on the (text|chunk|documentation)[^:]*:\s*",
            r"^(sure|okay|certainly)[^.]*\.\s*",
        ]

        result = text
        for pattern in preambles:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        return result.strip()

    def extract_metadata(self, chunk: Chunk):
        """Extract context, summary, and questions for a chunk."""
        # Context
        if not chunk.context:
            resp = self.call_llm(
                CONTEXT_PROMPT.format(doc_title=chunk.doc_title, chunk=chunk.content),
                max_tokens=150
            )
            if not resp.startswith("ERROR:"):
                chunk.context = self.strip_llm_preamble(resp)

        # Summary
        if not chunk.summary:
            resp = self.call_llm(
                SUMMARY_PROMPT.format(chunk=chunk.content),
                max_tokens=100
            )
            if not resp.startswith("ERROR:"):
                chunk.summary = self.strip_llm_preamble(resp)

        # Questions
        if not chunk.questions:
            resp = self.call_llm(
                QA_PROMPT.format(chunk=chunk.content),
                max_tokens=200
            )
            if not resp.startswith("ERROR:"):
                chunk.questions = self.parse_questions(resp)

    def run_extraction_pass(self, pass_config: Dict, run_num: int, pass_idx: int = 0) -> Tuple[int, int]:
        """
        Run a single extraction pass over all chunks.

        Returns: (new_entities_count, total_extracted_count)
        """
        pass_name = pass_config["name"]
        prompt_template = pass_config["prompt"]

        new_count = 0
        total_count = 0
        error_count = 0

        for idx, chunk in enumerate(self.chunks):
            # Check for shutdown request
            if self.should_stop:
                self.log(f"  [{pass_name}] Shutdown requested, stopping at chunk {idx}", "WARN")
                break

            # Update tracking for emergency checkpoints
            self._current_chunk = idx

            try:
                # Extract entities
                resp = self.call_llm(prompt_template.format(chunk=chunk.content))
                entities = self.parse_entities(resp)

                for entity in entities:
                    total_count += 1

                    if entity.key not in self.known_entities:
                        self.known_entities.add(entity.key)
                        new_count += 1

                        # Add to chunk if not already present
                        if entity.key not in [e.key for e in chunk.entities]:
                            chunk.entities.append(entity)

            except Exception as e:
                error_count += 1
                self.log(f"  [{pass_name}] Error on chunk {idx} ({chunk.doc_name}): {e}", "WARN")
                # Continue to next chunk instead of crashing
                continue

            # Progress logging
            if (idx + 1) % 20 == 0:
                self.log(f"  [{pass_name}] {idx + 1}/{len(self.chunks)} chunks, +{new_count} new")

            # Mid-pass checkpoint every N chunks
            if (idx + 1) % self.config.checkpoint_interval == 0:
                self._save_midpass_checkpoint(run_num, pass_idx, idx + 1)

        if error_count > 0:
            self.log(f"  [{pass_name}] Completed with {error_count} errors", "WARN")

        return new_count, total_count

    def run_single_complete_pass(self, run_num: int) -> RunStats:
        """Run all 6 extraction passes and return stats."""
        self._current_run = run_num

        self.log(f"\n{'='*60}")
        self.log(f"RUN {run_num} STARTING")
        self.log(f"Known entities before: {len(self.known_entities)}")
        self.log(f"{'='*60}")

        run_start = time.time()
        run_new = 0
        run_total = 0
        pass_breakdown = []

        for pass_idx, pass_config in enumerate(EXTRACTION_PASSES):
            # Check for shutdown
            if self.should_stop:
                self.log(f"Shutdown requested, stopping at pass {pass_idx}", "WARN")
                break

            self._current_pass = pass_idx
            pass_start = time.time()
            self.log(f"  Pass {pass_idx + 1}/6: {pass_config['focus']}")

            # First pass of first run also extracts metadata
            if run_num == 1 and pass_idx == 0:
                self.log("  (Also extracting context/summary/questions)")
                for chunk in self.chunks:
                    self.extract_metadata(chunk)

            new_count, total_count = self.run_extraction_pass(pass_config, run_num, pass_idx)
            run_new += new_count
            run_total += total_count

            pass_time = time.time() - pass_start
            pass_breakdown.append({
                "pass": pass_idx + 1,
                "name": pass_config["name"],
                "focus": pass_config["focus"],
                "new_entities": new_count,
                "total_extracted": total_count,
                "time_seconds": round(pass_time, 1),
            })

            self.log(f"  Pass {pass_idx + 1} complete: +{new_count} new, {total_count} total, {pass_time:.0f}s")

        run_time = time.time() - run_start

        # Calculate convergence percentage
        if run_total > 0:
            convergence_pct = run_new / run_total
        else:
            convergence_pct = 0.0

        stats = RunStats(
            run_number=run_num,
            new_entities=run_new,
            total_extracted=run_total,
            cumulative_unique=len(self.known_entities),
            convergence_pct=convergence_pct,
            time_seconds=run_time,
            pass_breakdown=pass_breakdown,
        )

        self.log(f"\nRUN {run_num} COMPLETE:")
        self.log(f"  New entities: {run_new}")
        self.log(f"  Total extracted: {run_total}")
        self.log(f"  Cumulative unique: {len(self.known_entities)}")
        self.log(f"  Convergence: {convergence_pct:.1%} new")
        self.log(f"  Time: {run_time/60:.1f} minutes")

        return stats

    def has_converged(self) -> bool:
        """Check if entity extraction has converged."""
        if len(self.run_stats) < self.config.min_runs:
            return False

        if len(self.run_stats) >= self.config.max_runs:
            self.log(f"Max runs ({self.config.max_runs}) reached")
            return True

        last_run = self.run_stats[-1]
        if last_run.convergence_pct <= self.config.convergence_threshold:
            self.log(f"Converged! New entities {last_run.convergence_pct:.1%} <= threshold {self.config.convergence_threshold:.1%}")
            return True

        return False

    def save_checkpoint(self, run_num: int):
        """Save checkpoint for resume capability."""
        if not self.config.output_dir:
            return

        checkpoint_path = self.config.output_dir / f"checkpoint_run{run_num}.json"

        state = IngestionState(
            config=asdict(self.config),
            chunks=[c.to_dict() for c in self.chunks],
            known_entities=list(self.known_entities),
            run_stats=[asdict(s) for s in self.run_stats],
            current_run=run_num,
            current_pass=6,  # Complete
            current_chunk=len(self.chunks),
            started_at=datetime.now().isoformat(),
            last_checkpoint=datetime.now().isoformat(),
        )
        state.save(checkpoint_path)
        self.log(f"Checkpoint saved: {checkpoint_path}")

    def _save_midpass_checkpoint(self, run_num: int, pass_idx: int, chunk_idx: int):
        """Save checkpoint mid-pass for crash recovery."""
        if not self.config.output_dir:
            return

        checkpoint_path = self.config.output_dir / "checkpoint_latest.json"

        try:
            state = IngestionState(
                config=asdict(self.config),
                chunks=[c.to_dict() for c in self.chunks],
                known_entities=list(self.known_entities),
                run_stats=[asdict(s) for s in self.run_stats],
                current_run=run_num,
                current_pass=pass_idx,
                current_chunk=chunk_idx,
                started_at=getattr(self, '_started_at', datetime.now().isoformat()),
                last_checkpoint=datetime.now().isoformat(),
            )
            state.save(checkpoint_path)
        except Exception as e:
            self.log(f"Failed to save mid-pass checkpoint: {e}", "WARN")

    def store_to_qdrant(self):
        """Store chunks to Qdrant vector database."""
        if self.config.dry_run or self.config.skip_storage:
            self.log("Skipping Qdrant storage (dry run or skip_storage)")
            return

        self.log(f"Storing to Qdrant: {self.config.collection_name}")

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            self.log("FlagEmbedding not installed, skipping embeddings", "WARN")
            return

        embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
        client = QdrantClient(host=self.config.qdrant_host, port=self.config.qdrant_port)

        # Delete existing collection
        try:
            client.delete_collection(self.config.collection_name)
        except:
            pass

        # Create collection
        client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

        # Batch insert
        for i in range(0, len(self.chunks), self.config.batch_size):
            batch = self.chunks[i:i + self.config.batch_size]

            # Prepare texts for embedding
            texts = []
            for c in batch:
                if c.context:
                    texts.append(f"{c.context}\n\n{c.content}")
                else:
                    texts.append(c.content)

            embeddings = embedder.encode(texts)["dense_vecs"]

            points = []
            for j, chunk in enumerate(batch):
                points.append(PointStruct(
                    id=abs(hash(chunk.chunk_id)) % (2**63),
                    vector=embeddings[j].tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_name": chunk.doc_name,
                        "title": chunk.doc_title,
                        "content": chunk.content,
                        "context": chunk.context,
                        "summary": chunk.summary,
                        "entities": [e.to_dict() for e in chunk.entities],
                        "questions": chunk.questions,
                        "entity_count": len(chunk.entities),
                    }
                ))

            client.upsert(collection_name=self.config.collection_name, points=points)

            if (i + self.config.batch_size) % 50 == 0:
                self.log(f"  Stored {min(i + self.config.batch_size, len(self.chunks))}/{len(self.chunks)} chunks")

        self.log(f"Stored {len(self.chunks)} chunks to Qdrant collection: {self.config.collection_name}")

    def link_graph(self):
        """Link entities in Neo4j knowledge graph."""
        if self.config.dry_run or self.config.skip_graph:
            self.log("Skipping Neo4j linking (dry run or skip_graph)")
            return

        self.log("Linking entities in Neo4j...")

        try:
            from neo4j import GraphDatabase
        except ImportError:
            self.log("neo4j driver not installed, skipping graph linking", "WARN")
            return

        driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=self.config.neo4j_auth
        )

        # Count entities across chunks
        entity_counts = {}
        for chunk in self.chunks:
            for entity in chunk.entities:
                key = (entity.name, entity.type)
                entity_counts[key] = entity_counts.get(key, 0) + 1

        with driver.session() as session:
            # Create entities
            for (name, etype), count in entity_counts.items():
                session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type, e.count = $count
                """, name=name, type=etype, count=count)

            # Create documents
            doc_names = set(c.doc_name for c in self.chunks)
            for doc_name in doc_names:
                doc_title = doc_name.rsplit(".", 1)[0].replace("_", " ")
                session.run("""
                    MERGE (d:Document {name: $name})
                    SET d.title = $title
                """, name=doc_name, title=doc_title)

            # Create chunks and link to documents and entities
            for chunk in self.chunks:
                session.run("""
                    MERGE (c:Chunk {id: $id})
                    SET c.doc = $doc, c.title = $title, c.index = $index
                """, id=chunk.chunk_id, doc=chunk.doc_name,
                    title=chunk.doc_title, index=chunk.chunk_index)

                # Link chunk to document
                session.run("""
                    MATCH (d:Document {name: $doc_name})
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                """, doc_name=chunk.doc_name, chunk_id=chunk.chunk_id)

                # Link chunk to entities
                for entity in chunk.entities:
                    session.run("""
                        MATCH (c:Chunk {id: $chunk_id})
                        MATCH (e:Entity {name: $name})
                        MERGE (c)-[:MENTIONS]->(e)
                    """, chunk_id=chunk.chunk_id, name=entity.name)

        driver.close()
        self.log(f"Linked {len(entity_counts)} unique entities across {len(doc_names)} documents in Neo4j")

    def consolidate_entities(self):
        """
        Consolidate entities with same name but different types.

        When the same entity appears with multiple types (e.g., "Q-SYS" as
        PRODUCT, TECHNOLOGY, and CONCEPT), pick the most specific type
        based on a priority order.
        """
        # Type priority: more specific types have higher priority
        type_priority = {
            "FUNCTION": 10,     # Very specific
            "CLASS": 9,
            "EVENT": 8,
            "PARAMETER": 7,
            "PRODUCT": 6,
            "TECHNOLOGY": 5,
            "RELATIONSHIP": 4,
            "CONCEPT": 1,       # Most generic, lowest priority
        }

        # Build entity name -> types mapping
        entity_types: Dict[str, Dict[str, int]] = {}  # name -> {type -> count}

        for chunk in self.chunks:
            for entity in chunk.entities:
                name_lower = entity.name.lower()
                if name_lower not in entity_types:
                    entity_types[name_lower] = {}
                if entity.type not in entity_types[name_lower]:
                    entity_types[name_lower][entity.type] = 0
                entity_types[name_lower][entity.type] += 1

        # Find entities with multiple types
        multi_type = {k: v for k, v in entity_types.items() if len(v) > 1}

        if not multi_type:
            self.log("No entities with multiple types to consolidate")
            return

        self.log(f"Consolidating {len(multi_type)} entities with multiple types...")

        # Determine best type for each entity
        best_types: Dict[str, str] = {}
        for name, types_count in multi_type.items():
            # Pick type with highest priority (or highest count if tied)
            best_type = max(
                types_count.keys(),
                key=lambda t: (type_priority.get(t, 0), types_count[t])
            )
            best_types[name] = best_type

        # Update entities in chunks
        consolidated_count = 0
        for chunk in self.chunks:
            new_entities = []
            seen_names = set()

            for entity in chunk.entities:
                name_lower = entity.name.lower()

                # Skip duplicates within same chunk
                if name_lower in seen_names:
                    continue
                seen_names.add(name_lower)

                # Use best type if this entity had multiple types
                if name_lower in best_types:
                    if entity.type != best_types[name_lower]:
                        entity = Entity(name=entity.name, type=best_types[name_lower])
                        consolidated_count += 1

                new_entities.append(entity)

            chunk.entities = new_entities

        # Rebuild known_entities set with consolidated types
        self.known_entities = set()
        for chunk in self.chunks:
            for entity in chunk.entities:
                self.known_entities.add(entity.key)

        self.log(f"Consolidated {consolidated_count} entity type assignments")
        self.log(f"Final unique entities: {len(self.known_entities)}")

    def cleanup_entities(self):
        """
        Two-phase entity cleanup:
        1. Rule-based: Remove obvious noise using regex patterns
        2. LLM-based: Validate remaining entities using a thinking model
        """
        if not self.config.cleanup_enabled:
            self.log("Cleanup disabled, skipping")
            return

        self.log("\n" + "=" * 60)
        self.log("ENTITY CLEANUP PHASE")
        self.log("=" * 60)

        before_count = len(self.known_entities)

        # Phase 1: Rule-based cleanup
        self._cleanup_rule_based()

        # Phase 2: LLM-based validation
        self._cleanup_llm_validation()

        after_count = len(self.known_entities)
        removed = before_count - after_count
        self.log(f"Cleanup complete: {removed} entities removed ({removed/before_count*100:.1f}%)")
        self.log(f"Final entity count: {after_count}")

    def _cleanup_rule_based(self):
        """Remove entities matching noise patterns."""
        self.log("Phase 1: Rule-based cleanup...")

        # Compile patterns
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.config.noise_patterns]

        removed_count = 0
        removed_examples = []

        for chunk in self.chunks:
            clean_entities = []
            for entity in chunk.entities:
                is_noise = False
                for pattern in compiled_patterns:
                    if pattern.search(entity.name):
                        is_noise = True
                        if len(removed_examples) < 10:
                            removed_examples.append(f"{entity.name}:{entity.type}")
                        break

                if not is_noise:
                    clean_entities.append(entity)
                else:
                    removed_count += 1

            chunk.entities = clean_entities

        # Rebuild known_entities
        self.known_entities = set()
        for chunk in self.chunks:
            for entity in chunk.entities:
                self.known_entities.add(entity.key)

        self.log(f"  Rule-based: removed {removed_count} noisy entities")
        if removed_examples:
            self.log(f"  Examples: {removed_examples[:5]}")

    def _cleanup_llm_validation(self):
        """Use LLM to validate and score entities."""
        cleanup_model = self.config.cleanup_model or self.config.model
        self.log(f"Phase 2: LLM validation with {cleanup_model}...")

        # Get unique entities for validation
        unique_entities = {}
        for chunk in self.chunks:
            for entity in chunk.entities:
                key = entity.key
                if key not in unique_entities:
                    unique_entities[key] = entity

        entities_list = list(unique_entities.values())
        self.log(f"  Validating {len(entities_list)} unique entities...")

        # Process in batches
        discard_set = set()
        reclassify_map = {}  # old_key -> new_type

        for i in range(0, len(entities_list), self.config.cleanup_batch_size):
            batch = entities_list[i:i + self.config.cleanup_batch_size]

            # Format batch for LLM
            batch_text = "\n".join([f"- {e.name} [{e.type}]" for e in batch])

            prompt = f"""You are validating entities extracted from Q-SYS and Lua technical documentation for a knowledge base.

TASK: For each entity, think carefully about whether it would help someone searching the knowledge base, then decide:

DECISION CRITERIA:
- KEEP: Specific, searchable term that someone would look up (API names, functions, components, protocols, specific concepts)
- DISCARD: Generic words, descriptions, syntax fragments, or things no one would search for
- RETYPE:NEW_TYPE: Useful entity but wrong category

ENTITY TYPES (for RETYPE):
- FUNCTION: Callable functions/methods (e.g., string.format, Controls.Get)
- PARAMETER: Properties, settings, arguments (e.g., .Value, timeout, port)
- PRODUCT: Hardware/software products (e.g., Core 110f, Q-SYS Designer)
- TECHNOLOGY: Protocols, standards, formats (e.g., Dante, TCP/IP, JSON)
- CONCEPT: Technical concepts worth defining (e.g., metatables, coroutines)
- CLASS: Object types/classes (e.g., TcpSocket, Component)
- EVENT: Events/callbacks (e.g., EventHandler, OnChange)

THINK ABOUT EACH ENTITY:
1. Would a Q-SYS/Lua developer search for this term?
2. Is it specific enough to be useful, or too generic?
3. Is it an actual "thing" (noun) or just a description/behavior?
4. Is the type correct?

ENTITIES TO REVIEW:
{batch_text}

OUTPUT FORMAT (one per line, no explanations):
entity_name|DECISION

EXAMPLES:
Controls.Inputs|KEEP
string.format|KEEP
the value|DISCARD
break statement|DISCARD
error unwinds the stack|DISCARD
TcpSocket|KEEP
EventHandler|RETYPE:FUNCTION
Core 110f|KEEP
arithmetic operators|DISCARD

YOUR DECISIONS:"""

            # Call LLM (use cleanup model if specified)
            original_model = self.config.model
            if self.config.cleanup_model:
                self.config.model = self.config.cleanup_model
                # Update thinking model detection
                self.is_thinking_model = is_thinking_model(self.config.cleanup_model)

            resp = self.call_llm(prompt, max_tokens=800)

            # Restore original model
            if self.config.cleanup_model:
                self.config.model = original_model
                self.is_thinking_model = is_thinking_model(original_model)

            # Parse response
            if not resp.startswith("ERROR:"):
                for line in resp.strip().split("\n"):
                    line = line.strip()
                    if "|" not in line:
                        continue

                    parts = line.split("|", 1)
                    if len(parts) != 2:
                        continue

                    entity_name = parts[0].strip()
                    decision = parts[1].strip().upper()

                    # Find matching entity
                    matching_entity = None
                    for e in batch:
                        if e.name.lower() == entity_name.lower():
                            matching_entity = e
                            break

                    if not matching_entity:
                        continue

                    if decision == "DISCARD":
                        discard_set.add(matching_entity.key)
                    elif decision.startswith("RETYPE:"):
                        new_type = decision.replace("RETYPE:", "").strip()
                        if new_type in {"FUNCTION", "PARAMETER", "PRODUCT", "TECHNOLOGY", "CONCEPT", "CLASS", "EVENT", "RELATIONSHIP"}:
                            reclassify_map[matching_entity.key] = new_type

            # Progress
            processed = min(i + self.config.cleanup_batch_size, len(entities_list))
            if processed % 100 == 0 or processed == len(entities_list):
                self.log(f"  Validated {processed}/{len(entities_list)} entities...")

        # Apply changes to chunks
        self.log(f"  Applying: {len(discard_set)} discards, {len(reclassify_map)} reclassifications")

        for chunk in self.chunks:
            clean_entities = []
            for entity in chunk.entities:
                if entity.key in discard_set:
                    continue

                if entity.key in reclassify_map:
                    entity = Entity(name=entity.name, type=reclassify_map[entity.key])

                clean_entities.append(entity)

            chunk.entities = clean_entities

        # Rebuild known_entities
        self.known_entities = set()
        for chunk in self.chunks:
            for entity in chunk.entities:
                self.known_entities.add(entity.key)

        self.log(f"  LLM validation: discarded {len(discard_set)}, reclassified {len(reclassify_map)}")

    def run(self) -> Dict[str, Any]:
        """
        Main ingestion loop.

        Runs extraction passes until convergence is detected.
        Returns final statistics.
        """
        self.progress_file.write_text("")  # Clear progress log

        self.log("=" * 60)
        self.log("ROBUST KB INGESTION STARTING")
        self.log(f"Model: {self.config.model}")
        self.log(f"Thinking model: {self.is_thinking_model}")
        self.log(f"Source: {self.config.source_dir}")
        self.log(f"Collection: {self.config.collection_name}")
        self.log(f"Convergence threshold: {self.config.convergence_threshold:.0%}")
        self.log(f"Max runs: {self.config.max_runs}")
        self.log("=" * 60)

        # Load documents
        self.chunks = self.load_documents()

        if not self.chunks:
            self.log("No chunks to process!", "ERROR")
            return {"error": "No chunks found"}

        # Run until convergence
        run_num = 0
        total_start = time.time()

        while not self.has_converged() and not self.should_stop:
            run_num += 1
            stats = self.run_single_complete_pass(run_num)
            self.run_stats.append(stats)
            self.save_checkpoint(run_num)

        if self.should_stop:
            self.log("Ingestion stopped early due to shutdown request", "WARN")

        total_time = time.time() - total_start

        # Consolidate entities (merge same name with different types)
        self.consolidate_entities()

        # Cleanup entities (rule-based + LLM validation)
        if self.config.cleanup_enabled:
            self.cleanup_entities()

        # Final summary
        self.log("\n" + "=" * 60)
        self.log("INGESTION COMPLETE")
        self.log("=" * 60)
        self.log(f"Total runs: {len(self.run_stats)}")
        self.log(f"Total unique entities: {len(self.known_entities)}")
        self.log(f"Total time: {total_time/60:.1f} minutes")
        self.log("")

        # Show convergence curve
        self.log("CONVERGENCE CURVE:")
        for stats in self.run_stats:
            bar = "█" * int(stats.convergence_pct * 50)
            self.log(f"  Run {stats.run_number}: {stats.convergence_pct:5.1%} new {bar}")

        # Store results
        if not self.config.dry_run:
            self.store_to_qdrant()
            self.link_graph()

        # Save final results
        if self.config.output_dir:
            results = {
                "model": self.config.model,
                "source_dir": str(self.config.source_dir),
                "collection": self.config.collection_name,
                "total_chunks": len(self.chunks),
                "total_documents": len(set(c.doc_name for c in self.chunks)),
                "total_unique_entities": len(self.known_entities),
                "total_runs": len(self.run_stats),
                "convergence_threshold": self.config.convergence_threshold,
                "total_time_seconds": total_time,
                "run_stats": [asdict(s) for s in self.run_stats],
                "entity_type_distribution": self._get_entity_distribution(),
                "completed_at": datetime.now().isoformat(),
            }

            results_path = self.config.output_dir / "final_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            self.log(f"Results saved: {results_path}")

        return results

    def _get_entity_distribution(self) -> Dict[str, int]:
        """Get entity count by type."""
        dist = {}
        for chunk in self.chunks:
            for entity in chunk.entities:
                dist[entity.type] = dist.get(entity.type, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))


# =============================================================================
# CLI INTERFACE
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robust KB Ingestion - Run until convergence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python kb_ingest_robust.py --source ~/docs --collection my_kb

  # Use specific model
  python kb_ingest_robust.py --source ~/docs --collection my_kb --model llama3.1:8b

  # Custom convergence threshold
  python kb_ingest_robust.py --source ~/docs --collection my_kb --threshold 0.03

  # Dry run (no storage)
  python kb_ingest_robust.py --source ~/docs --dry-run --verbose
"""
    )

    # Required arguments
    parser.add_argument("--source", "-s", type=Path,
                        help="Source directory containing documents")
    parser.add_argument("--collection", "-c", type=str,
                        help="Qdrant collection name")

    # Model settings
    parser.add_argument("--model", "-m", type=str, default="llama3.1:8b",
                        help="Ollama model to use (default: llama3.1:8b)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="LLM temperature (default: 0.3)")

    # Convergence settings
    parser.add_argument("--threshold", "-t", type=float, default=0.05,
                        help="Convergence threshold - stop when new entities below this %% (default: 0.05)")
    parser.add_argument("--max-runs", type=int, default=10,
                        help="Maximum extraction runs (default: 10)")
    parser.add_argument("--min-runs", type=int, default=2,
                        help="Minimum extraction runs (default: 2)")

    # Chunking settings
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Chunk size in words (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=75,
                        help="Chunk overlap in words (default: 75)")

    # Output settings
    parser.add_argument("--output", "-o", type=Path,
                        help="Output directory for results (default: source_dir)")

    # Flags
    parser.add_argument("--dry-run", action="store_true",
                        help="Run extraction without storing results")
    parser.add_argument("--skip-graph", action="store_true",
                        help="Skip Neo4j graph linking")
    parser.add_argument("--skip-storage", action="store_true",
                        help="Skip Qdrant storage")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    # Cleanup settings
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Skip entity cleanup phase")
    parser.add_argument("--cleanup-model", type=str, default=None,
                        help="Model for cleanup validation (default: same as --model, recommend thinking model)")

    # Resume
    parser.add_argument("--resume", type=Path,
                        help="Resume from checkpoint file")

    return parser


def main():
    if not HAS_DEPS:
        sys.exit(1)

    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    if not args.source and not args.resume:
        parser.error("--source is required (or --resume to continue from checkpoint)")

    if args.source and not args.collection and not args.dry_run:
        # Auto-generate collection name
        args.collection = f"kb-{args.source.name}-{datetime.now().strftime('%Y%m%d-%H%M')}"
        console.print(f"[yellow]Auto-generated collection name: {args.collection}[/yellow]")

    # Create config
    config = IngestionConfig(
        source_dir=args.source,
        model=args.model,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        convergence_threshold=args.threshold,
        max_runs=args.max_runs,
        min_runs=args.min_runs,
        collection_name=args.collection,
        output_dir=args.output or args.source,
        dry_run=args.dry_run,
        skip_graph=args.skip_graph,
        skip_storage=args.skip_storage,
        verbose=args.verbose,
        cleanup_enabled=not args.no_cleanup,
        cleanup_model=args.cleanup_model,
    )

    # Run ingestion
    engine = IngestionEngine(config)

    try:
        results = engine.run()

        # Print summary table
        console.print("\n")
        table = Table(title="Ingestion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model", config.model)
        table.add_row("Total Runs", str(len(engine.run_stats)))
        table.add_row("Total Chunks", str(len(engine.chunks)))
        table.add_row("Unique Entities", str(len(engine.known_entities)))
        table.add_row("Collection", config.collection_name or "(dry run)")

        console.print(table)

        if not config.dry_run:
            console.print(f"\n[green]✓ Results saved to: {config.output_dir}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted - checkpoint saved[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
