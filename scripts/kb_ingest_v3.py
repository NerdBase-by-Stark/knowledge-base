#!/usr/bin/env python3
"""
KB Ingestion Pipeline v3 - Quality-Focused
==========================================
- Multi-model voting (LLAMA + QWEN + Mistral)
- Source text validation (no hallucinations)
- Rule-based type inference
- Structure-aware extraction
- Relationship extraction
- Hierarchical ontology
"""

import json
import re
import hashlib
import ollama
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import logging

# ============================================================================
# ONTOLOGY DEFINITION
# ============================================================================

ONTOLOGY = {
    "COMPONENT": {
        "description": "Q-SYS hardware or software module (Timer, Control, Mixer)",
        "patterns": [r"^[A-Z][a-z]+$", r"Component$"],
        "examples": ["Timer", "Control", "ChannelGroup", "Mixer"]
    },
    "FUNCTION": {
        "description": "Standalone callable function",
        "patterns": [r"\(\)$", r"^[a-z_]+$"],
        "examples": ["print()", "tonumber()", "pairs()"]
    },
    "METHOD": {
        "description": "Function belonging to an object (Object.Method)",
        "patterns": [r"^[A-Z][a-zA-Z]*\.[A-Z][a-zA-Z]+$", r"\.[A-Z]"],
        "examples": ["Timer.Start", "Control.GetValue", "string.format"]
    },
    "PROPERTY": {
        "description": "Readable/writable attribute of an object",
        "patterns": [r"^[A-Z][a-zA-Z]*\.[a-z][a-zA-Z]+$", r"\.Value$", r"\.String$"],
        "examples": ["Control.Value", "Control.String", "Timer.IsRunning"]
    },
    "EVENT": {
        "description": "Callback or event handler",
        "patterns": [r"^on[A-Z]", r"EventHandler$", r"Callback$"],
        "examples": ["onValueChange", "EventHandler", "Timer.EventHandler"]
    },
    "PARAMETER": {
        "description": "Function argument or configuration option",
        "patterns": [r"^[a-z_]+$", r"_arg$", r"_param$"],
        "examples": ["timeout", "callback", "interval", "count"]
    },
    "CONSTANT": {
        "description": "Fixed value or enumeration",
        "patterns": [r"^[A-Z_]+$", r"^[A-Z]{2,}"],
        "examples": ["NULL", "TRUE", "FALSE", "LUA_VERSION"]
    },
    "METAMETHOD": {
        "description": "Lua metamethod (double underscore prefix)",
        "patterns": [r"^__[a-z]+$"],
        "examples": ["__index", "__newindex", "__call", "__add"]
    },
    "CONCEPT": {
        "description": "Abstract technical concept or pattern",
        "patterns": [],
        "examples": ["event-driven programming", "garbage collection", "coroutine"]
    },
    "TECHNOLOGY": {
        "description": "External technology, protocol, or standard",
        "patterns": [],
        "examples": ["Lua", "Q-SYS", "JSON", "TCP/IP", "Dante"]
    }
}

# ============================================================================
# TYPE INFERENCE RULES
# ============================================================================

def infer_type(name: str) -> Tuple[str, float]:
    """
    Infer entity type from name patterns.
    Returns (type, confidence) where confidence is 0.0-1.0
    """
    name = name.strip()

    # High confidence rules (0.9+)
    if name.startswith("__") and name[2:].islower():
        return ("METAMETHOD", 0.95)

    if name.isupper() and "_" in name:
        return ("CONSTANT", 0.90)

    if re.match(r"^on[A-Z]", name):
        return ("EVENT", 0.90)

    if "EventHandler" in name or "Callback" in name:
        return ("EVENT", 0.85)

    # Method pattern: Component.Method
    if re.match(r"^[A-Z][a-zA-Z]*\.[A-Z][a-zA-Z]+$", name):
        return ("METHOD", 0.85)

    # Property pattern: Component.property (lowercase after dot)
    if re.match(r"^[A-Z][a-zA-Z]*\.[a-z][a-zA-Z]*$", name):
        return ("PROPERTY", 0.80)

    # Function with parens
    if name.endswith("()"):
        return ("FUNCTION", 0.85)

    # Lua stdlib functions
    lua_stdlib = ["print", "pairs", "ipairs", "tonumber", "tostring", "type",
                  "assert", "error", "pcall", "xpcall", "select", "next",
                  "rawget", "rawset", "setmetatable", "getmetatable"]
    if name.lower() in lua_stdlib:
        return ("FUNCTION", 0.95)

    # Q-SYS components (capitalized single word)
    qsys_components = ["Timer", "Control", "Controls", "Component", "Mixer",
                       "ChannelGroup", "Snapshot", "Design", "System"]
    if name in qsys_components:
        return ("COMPONENT", 0.90)

    # Technology names
    tech_names = ["Lua", "Q-SYS", "JSON", "XML", "TCP", "UDP", "HTTP",
                  "Dante", "AES67", "UCI", "GPIO"]
    if name in tech_names:
        return ("TECHNOLOGY", 0.90)

    # Medium confidence (0.5-0.7)
    if re.match(r"^[A-Z][a-z]+$", name):  # Capitalized word
        return ("COMPONENT", 0.60)

    if re.match(r"^[a-z_]+$", name) and len(name) < 20:  # lowercase identifier
        return ("PARAMETER", 0.50)

    # Default to CONCEPT with low confidence
    return ("CONCEPT", 0.30)


# ============================================================================
# SOURCE VALIDATION
# ============================================================================

def validate_against_source(entity_name: str, source_text: str) -> Tuple[bool, float]:
    """
    Validate that an entity actually appears in source text.
    Returns (is_valid, confidence)
    """
    # Exact match
    if entity_name in source_text:
        return (True, 1.0)

    # Case-insensitive match
    if entity_name.lower() in source_text.lower():
        return (True, 0.9)

    # Fuzzy match: check if all significant words appear
    words = re.findall(r'[a-zA-Z]+', entity_name)
    if len(words) >= 2:
        matches = sum(1 for w in words if w.lower() in source_text.lower())
        if matches == len(words):
            return (True, 0.7)
        if matches >= len(words) * 0.7:
            return (True, 0.5)

    # Check for code patterns (Timer.Start might appear as Timer:Start)
    variants = [
        entity_name.replace(".", ":"),
        entity_name.replace(".", "_"),
        entity_name.replace("_", "."),
    ]
    for variant in variants:
        if variant in source_text:
            return (True, 0.8)

    return (False, 0.0)


# ============================================================================
# STRUCTURE-AWARE EXTRACTION
# ============================================================================

@dataclass
class StructuredChunk:
    """A chunk with structural awareness."""
    chunk_id: str
    doc_name: str
    content: str
    headers: List[str] = field(default_factory=list)
    code_blocks: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    lists: List[str] = field(default_factory=list)
    bold_text: List[str] = field(default_factory=list)


def parse_structure(content: str) -> Dict[str, List[str]]:
    """Extract structural elements from markdown content."""
    structure = {
        "headers": [],
        "code_blocks": [],
        "tables": [],
        "lists": [],
        "bold_text": [],
    }

    # Headers (# ## ###)
    structure["headers"] = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)

    # Code blocks (``` or indented)
    code_pattern = r'```[\w]*\n(.*?)```'
    structure["code_blocks"] = re.findall(code_pattern, content, re.DOTALL)

    # Inline code
    inline_code = re.findall(r'`([^`]+)`', content)
    structure["code_blocks"].extend(inline_code)

    # Tables (| col | col |)
    table_rows = re.findall(r'^\|.+\|$', content, re.MULTILINE)
    structure["tables"] = table_rows

    # Lists (- item or * item or 1. item)
    structure["lists"] = re.findall(r'^[\s]*[-*\d.]+\s+(.+)$', content, re.MULTILINE)

    # Bold text (**text** or __text__)
    structure["bold_text"] = re.findall(r'\*\*([^*]+)\*\*|__([^_]+)__', content)
    structure["bold_text"] = [b[0] or b[1] for b in structure["bold_text"]]

    return structure


# ============================================================================
# ENTITY EXTRACTION
# ============================================================================

@dataclass
class ExtractedEntity:
    """An extracted entity with metadata."""
    name: str
    type: str
    source_chunk: str
    extraction_model: str
    confidence: float = 1.0
    source_validated: bool = False
    type_inferred: bool = False
    relationships: List[Dict] = field(default_factory=list)


def strip_thinking(text: str) -> str:
    """Strip <think> blocks from thinking model output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response."""
    # Try to find JSON block
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try to find array
    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        try:
            return {"entities": json.loads(array_match.group())}
        except json.JSONDecodeError:
            pass

    return None


class MultiModelExtractor:
    """Extract entities using multiple models with voting."""

    MODELS = [
        ("llama3.1:8b", False),      # (model_name, is_thinking_model)
        ("qwen2.5:7b", False),       # Fast, non-thinking
        ("mistral:latest", False),
    ]

    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        self.logger = logging.getLogger("extractor")

    def _call_model(self, model: str, prompt: str, is_thinking: bool) -> Optional[str]:
        """Call a single model with timeout."""
        def _do_call():
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_call)
                result = future.result(timeout=self.timeout)
                if is_thinking:
                    result = strip_thinking(result)
                return result
        except Exception as e:
            self.logger.warning(f"Model {model} failed: {e}")
            return None

    def extract_from_chunk(self, chunk: StructuredChunk) -> List[ExtractedEntity]:
        """Extract entities from a chunk using all models."""

        # Build structure-aware prompt
        prompt = self._build_prompt(chunk)

        # Collect extractions from each model
        all_extractions = []

        for model, is_thinking in self.MODELS:
            self.logger.info(f"  Extracting with {model}...")
            response = self._call_model(model, prompt, is_thinking)

            if response:
                parsed = extract_json(response)
                if parsed and "entities" in parsed:
                    for ent in parsed["entities"]:
                        if isinstance(ent, dict) and "name" in ent:
                            all_extractions.append({
                                "name": ent["name"],
                                "type": ent.get("type", "CONCEPT"),
                                "model": model
                            })
                        elif isinstance(ent, str):
                            # Handle "name:TYPE" format
                            parts = ent.rsplit(":", 1)
                            all_extractions.append({
                                "name": parts[0],
                                "type": parts[1] if len(parts) > 1 else "CONCEPT",
                                "model": model
                            })

        # Vote and validate
        validated = self._vote_and_validate(all_extractions, chunk)
        return validated

    def _build_prompt(self, chunk: StructuredChunk) -> str:
        """Build structure-aware extraction prompt."""

        prompt_parts = [
            "Extract technical entities from this Q-SYS/Lua documentation.",
            "",
            f"## Document: {chunk.doc_name}",
            ""
        ]

        if chunk.headers:
            prompt_parts.append("### Headers in this section:")
            for h in chunk.headers[:5]:
                prompt_parts.append(f"- {h}")
            prompt_parts.append("")

        if chunk.code_blocks:
            prompt_parts.append("### Code examples:")
            for code in chunk.code_blocks[:3]:
                prompt_parts.append(f"```\n{code[:200]}\n```")
            prompt_parts.append("")

        prompt_parts.extend([
            "### Content:",
            chunk.content[:2000],
            "",
            "### Entity Types (use ONLY these):",
        ])

        for etype, info in ONTOLOGY.items():
            prompt_parts.append(f"- {etype}: {info['description']}")
            if info['examples']:
                prompt_parts.append(f"  Examples: {', '.join(info['examples'][:3])}")

        prompt_parts.extend([
            "",
            "### Instructions:",
            "1. Extract ONLY entities that appear in the text",
            "2. Use the exact type from the ontology above",
            "3. Include functions, methods, components, parameters, events",
            "4. Do NOT extract generic words (the, this, value, etc.)",
            "",
            'Return JSON: {"entities": [{"name": "...", "type": "..."}, ...]}'
        ])

        return "\n".join(prompt_parts)

    def _vote_and_validate(
        self,
        extractions: List[Dict],
        chunk: StructuredChunk
    ) -> List[ExtractedEntity]:
        """Apply voting, source validation, and type inference."""

        # Group by normalized name
        def normalize(name):
            return name.lower().strip().replace("()", "")

        grouped = defaultdict(list)
        for ext in extractions:
            key = normalize(ext["name"])
            grouped[key].append(ext)

        validated = []

        for norm_name, occurrences in grouped.items():
            # Voting: need at least 2 models to agree (or 1 if only 1 model responded)
            model_count = len(set(e["model"] for e in occurrences))
            total_models = len([m for m, _ in self.MODELS])

            # Skip if only 1 model found it and we had multiple models respond
            if len(occurrences) == 1 and model_count < total_models:
                # Check if other models even responded
                responding_models = len(set(e["model"] for e in extractions))
                if responding_models > 1:
                    continue  # Skip - only 1 model found this

            # Use most common original name
            names = [e["name"] for e in occurrences]
            original_name = Counter(names).most_common(1)[0][0]

            # Source validation
            is_valid, source_conf = validate_against_source(original_name, chunk.content)
            if not is_valid:
                # Try with code blocks too
                all_text = chunk.content + " ".join(chunk.code_blocks)
                is_valid, source_conf = validate_against_source(original_name, all_text)

            if not is_valid:
                continue  # Hallucination - skip

            # Type inference
            llm_types = [e["type"] for e in occurrences]
            llm_type = Counter(llm_types).most_common(1)[0][0]

            inferred_type, type_conf = infer_type(original_name)

            # Use inferred type if high confidence, otherwise use LLM consensus
            if type_conf > 0.7:
                final_type = inferred_type
                type_inferred = True
            elif llm_type in ONTOLOGY:
                final_type = llm_type
                type_inferred = False
            else:
                final_type = inferred_type
                type_inferred = True

            # Calculate overall confidence
            vote_conf = len(occurrences) / total_models
            overall_conf = (vote_conf + source_conf + type_conf) / 3

            validated.append(ExtractedEntity(
                name=original_name,
                type=final_type,
                source_chunk=chunk.chunk_id,
                extraction_model=",".join(set(e["model"] for e in occurrences)),
                confidence=overall_conf,
                source_validated=is_valid,
                type_inferred=type_inferred
            ))

        return validated


# ============================================================================
# RELATIONSHIP EXTRACTION
# ============================================================================

class RelationshipExtractor:
    """Extract relationships between entities."""

    RELATIONSHIP_TYPES = [
        "BELONGS_TO",      # Timer.Start BELONGS_TO Timer
        "CALLS",           # function CALLS another_function
        "PARAMETER_OF",    # timeout PARAMETER_OF Timer.Start
        "RETURNS",         # function RETURNS type
        "EXTENDS",         # UCI EXTENDS Component
        "USES",            # Script USES Timer
        "TRIGGERS",        # Event TRIGGERS callback
        "CONTAINS",        # Table CONTAINS elements
    ]

    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    def extract_relationships(
        self,
        entities: List[ExtractedEntity],
        chunk_content: str
    ) -> List[Dict]:
        """Extract relationships between entities found in a chunk."""

        relationships = []

        # Rule-based relationship extraction
        relationships.extend(self._extract_rule_based(entities))

        # LLM-based relationship extraction
        if len(entities) >= 2:
            relationships.extend(self._extract_llm_based(entities, chunk_content))

        return relationships

    def _extract_rule_based(self, entities: List[ExtractedEntity]) -> List[Dict]:
        """Extract relationships using patterns."""
        relationships = []

        entity_names = {e.name: e for e in entities}

        for entity in entities:
            name = entity.name

            # Method BELONGS_TO Component (Timer.Start -> Timer)
            if "." in name:
                parts = name.split(".")
                parent = parts[0]
                if parent in entity_names:
                    relationships.append({
                        "source": name,
                        "target": parent,
                        "type": "BELONGS_TO",
                        "confidence": 0.95
                    })

            # EventHandler patterns
            if "EventHandler" in name or name.startswith("on"):
                # Find what component it belongs to
                for other in entities:
                    if other.type == "COMPONENT" and other.name in name:
                        relationships.append({
                            "source": name,
                            "target": other.name,
                            "type": "BELONGS_TO",
                            "confidence": 0.85
                        })

        return relationships

    def _extract_llm_based(
        self,
        entities: List[ExtractedEntity],
        chunk_content: str
    ) -> List[Dict]:
        """Extract relationships using LLM."""

        entity_list = "\n".join([f"- {e.name} ({e.type})" for e in entities[:30]])

        prompt = f"""Given these entities found in Q-SYS/Lua documentation:

{entity_list}

And this context:
{chunk_content[:1500]}

Identify relationships between these entities.

Relationship types:
- BELONGS_TO: method/property belongs to component
- CALLS: function calls another function
- PARAMETER_OF: parameter belongs to function
- RETURNS: function returns something
- USES: component uses another component
- TRIGGERS: event triggers callback

Return JSON: {{"relationships": [{{"source": "...", "target": "...", "type": "..."}}]}}

Only include high-confidence relationships visible in the text."""

        try:
            response = ollama.chat(
                model="Qwen3:latest",
                messages=[{"role": "user", "content": prompt}]
            )
            text = strip_thinking(response["message"]["content"])
            parsed = extract_json(text)

            if parsed and "relationships" in parsed:
                return [
                    {**r, "confidence": 0.7}
                    for r in parsed["relationships"]
                    if all(k in r for k in ["source", "target", "type"])
                ]
        except Exception as e:
            logging.warning(f"Relationship extraction failed: {e}")

        return []


# ============================================================================
# MAIN PIPELINE
# ============================================================================

@dataclass
class PipelineConfig:
    source_dir: str
    output_dir: str
    collection_name: str = "qsys-lua-v3"
    chunk_size: int = 3000
    chunk_overlap: int = 200
    min_votes: int = 2
    timeout: int = 120
    resume: bool = False


class QualityPipeline:
    """High-quality KB ingestion pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.extractor = MultiModelExtractor(timeout=config.timeout)
        self.rel_extractor = RelationshipExtractor(timeout=config.timeout)
        self.all_entities: List[ExtractedEntity] = []
        self.all_relationships: List[Dict] = []
        self.chunks: List[StructuredChunk] = []

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger("pipeline")

        # Setup output
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.log_file = self.output_path / "pipeline.log"
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        self.logger.addHandler(file_handler)

    def log(self, msg: str):
        self.logger.info(msg)

    def load_documents(self) -> List[Path]:
        """Load markdown documents from source directory."""
        source = Path(self.config.source_dir)
        docs = list(source.glob("*.md"))
        self.log(f"Found {len(docs)} documents in {source}")
        return docs

    def chunk_document(self, doc_path: Path) -> List[StructuredChunk]:
        """Chunk a document with structure awareness."""
        content = doc_path.read_text(encoding='utf-8', errors='ignore')
        structure = parse_structure(content)

        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # Simple chunking (could be improved with semantic boundaries)
        for i in range(0, len(content), chunk_size - overlap):
            chunk_content = content[i:i + chunk_size]
            chunk_id = hashlib.md5(f"{doc_path.name}:{i}".encode()).hexdigest()[:12]

            # Find structure elements in this chunk
            chunk_structure = parse_structure(chunk_content)

            chunks.append(StructuredChunk(
                chunk_id=chunk_id,
                doc_name=doc_path.stem,
                content=chunk_content,
                headers=chunk_structure["headers"],
                code_blocks=chunk_structure["code_blocks"],
                tables=chunk_structure["tables"],
                lists=chunk_structure["lists"],
                bold_text=chunk_structure["bold_text"]
            ))

        return chunks

    def run(self):
        """Run the full pipeline."""
        self.log("=" * 60)
        self.log("QUALITY KB INGESTION PIPELINE v3")
        self.log("=" * 60)
        self.log(f"Source: {self.config.source_dir}")
        self.log(f"Output: {self.config.output_dir}")
        self.log(f"Resume: {self.config.resume}")
        self.log(f"Models: {[m for m, _ in MultiModelExtractor.MODELS]}")
        self.log("=" * 60)

        # Try to load checkpoint if resuming
        start_chunk = self.load_checkpoint()

        # Load and chunk documents
        docs = self.load_documents()

        for doc in docs:
            doc_chunks = self.chunk_document(doc)
            self.chunks.extend(doc_chunks)

        total_chunks = len(self.chunks)
        self.log(f"Created {total_chunks} chunks from {len(docs)} documents")

        # Extract entities from each chunk
        self.log("\n" + "=" * 60)
        self.log("ENTITY EXTRACTION (multi-model voting)")
        self.log("=" * 60)

        # Start from the checkpoint position if resuming
        start_index = start_chunk if start_chunk is not None else 0

        for i in range(start_index, total_chunks):
            chunk = self.chunks[i]
            self.log(f"\nChunk {i+1}/{total_chunks}: {chunk.doc_name}")

            # Extract entities
            entities = self.extractor.extract_from_chunk(chunk)
            self.log(f"  Extracted {len(entities)} validated entities")

            # Extract relationships
            if entities:
                relationships = self.rel_extractor.extract_relationships(
                    entities, chunk.content
                )
                self.log(f"  Found {len(relationships)} relationships")

                # Store relationships in entities
                for ent in entities:
                    ent.relationships = [
                        r for r in relationships
                        if r["source"] == ent.name or r["target"] == ent.name
                    ]

                self.all_relationships.extend(relationships)

            self.all_entities.extend(entities)

            # Save checkpoint periodically
            if (i + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_{i+1}.json")

        # Final deduplication
        self.log("\n" + "=" * 60)
        self.log("FINAL DEDUPLICATION")
        self.log("=" * 60)

        before = len(self.all_entities)
        self.deduplicate_entities()
        after = len(self.all_entities)
        self.log(f"Deduplicated: {before} -> {after} entities")

        # Save final output
        self.save_checkpoint("final.json")
        self.save_for_storage()

        self.log("\n" + "=" * 60)
        self.log("PIPELINE COMPLETE")
        self.log(f"Total entities: {len(self.all_entities)}")
        self.log(f"Total relationships: {len(self.all_relationships)}")
        self.log("=" * 60)

    def deduplicate_entities(self):
        """Deduplicate entities, keeping highest confidence."""
        seen = {}

        for ent in self.all_entities:
            key = ent.name.lower().strip()

            if key not in seen or ent.confidence > seen[key].confidence:
                seen[key] = ent

        self.all_entities = list(seen.values())

    def save_checkpoint(self, filename: str):
        """Save current state to checkpoint."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "source_dir": self.config.source_dir,
                "output_dir": self.config.output_dir,
                "collection_name": self.config.collection_name,
            },
            "stats": {
                "chunks": len(self.chunks),
                "entities": len(self.all_entities),
                "relationships": len(self.all_relationships),
            },
            "entities": [
                {
                    "name": e.name,
                    "type": e.type,
                    "confidence": e.confidence,
                    "source_chunk": e.source_chunk,
                    "extraction_model": e.extraction_model,
                    "source_validated": e.source_validated,
                    "type_inferred": e.type_inferred,
                }
                for e in self.all_entities
            ],
            "relationships": self.all_relationships,
        }

        with open(self.output_path / filename, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        self.log(f"Checkpoint saved: {filename}")

    def load_checkpoint(self) -> Optional[int]:
        """Load the latest checkpoint and return the starting chunk index.
        Returns None if no valid checkpoint found, or the index to resume from.
        """
        if not self.config.resume:
            return None

        # Find the most recent checkpoint
        checkpoints = sorted(self.output_path.glob("checkpoint_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not checkpoints:
            self.log("No checkpoints found, starting fresh")
            return None

        latest = checkpoints[0]

        try:
            with open(latest, 'r') as f:
                checkpoint = json.load(f)

            # Verify checkpoint matches current config
            if checkpoint.get("config", {}).get("source_dir") != self.config.source_dir:
                self.log(f"Checkpoint source mismatch: {checkpoint['config']['source_dir']} != {self.config.source_dir}")
                self.log("Starting fresh due to source directory change")
                return None

            # Restore entities and relationships
            for ent_data in checkpoint.get("entities", []):
                self.all_entities.append(ExtractedEntity(
                    name=ent_data["name"],
                    type=ent_data["type"],
                    source_chunk=ent_data["source_chunk"],
                    extraction_model=ent_data.get("extraction_model", ""),
                    confidence=ent_data.get("confidence", 1.0),
                    source_validated=ent_data.get("source_validated", False),
                    type_inferred=ent_data.get("type_inferred", False),
                    relationships=ent_data.get("relationships", [])
                ))

            self.all_relationships = checkpoint.get("relationships", [])

            chunks_processed = checkpoint.get("stats", {}).get("chunks", 0)
            self.log(f"Resumed from checkpoint: {latest.name}")
            self.log(f"  Loaded {len(self.all_entities)} entities")
            self.log(f"  Loaded {len(self.all_relationships)} relationships")
            self.log(f"  Resuming from chunk {chunks_processed + 1}")

            return chunks_processed

        except Exception as e:
            self.log(f"Failed to load checkpoint: {e}")
            self.log("Starting fresh")
            return None

    def save_for_storage(self):
        """Save in format ready for ChromaDB and Neo4j."""

        # Chunks with entities for ChromaDB
        chunks_output = []
        for chunk in self.chunks:
            chunk_entities = [
                e for e in self.all_entities
                if e.source_chunk == chunk.chunk_id
            ]
            chunks_output.append({
                "chunk_id": chunk.chunk_id,
                "doc_name": chunk.doc_name,
                "content": chunk.content,
                "entities": [{"name": e.name, "type": e.type} for e in chunk_entities],
                "headers": chunk.headers,
            })

        with open(self.output_path / "chunks_for_chromadb.json", 'w') as f:
            json.dump(chunks_output, f, indent=2)

        # Entities and relationships for Neo4j
        neo4j_output = {
            "nodes": [
                {
                    "name": e.name,
                    "type": e.type,
                    "confidence": e.confidence,
                }
                for e in self.all_entities
            ],
            "edges": self.all_relationships,
        }

        with open(self.output_path / "graph_for_neo4j.json", 'w') as f:
            json.dump(neo4j_output, f, indent=2)

        self.log("Saved storage-ready files")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quality KB Ingestion Pipeline v3")
    parser.add_argument("--source", required=True, help="Source directory with .md files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--collection", default="qsys-lua-v3", help="Collection name")
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    args = parser.parse_args()

    config = PipelineConfig(
        source_dir=args.source,
        output_dir=args.output,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
        resume=args.resume,
    )

    pipeline = QualityPipeline(config)
    pipeline.run()
