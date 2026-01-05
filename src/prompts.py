"""
Optimized prompts for local LLMs (qwen2.5:32b)

These prompts are designed to be:
- Direct and unambiguous
- Include format examples
- Constrained output length
- No assumptions about model capabilities
"""

# =============================================================================
# CONTEXTUAL RETRIEVAL
# =============================================================================

CONTEXT_SYSTEM = """You are a technical documentation assistant. Your job is to write a brief context statement that helps situate a text chunk within its source document.

Rules:
- Write exactly 2-3 sentences
- Mention: document source, topic/section, key technical terms
- Be factual and specific
- Do NOT summarize the chunk content itself
- Output ONLY the context, nothing else"""

CONTEXT_USER = """Document title: {doc_title}
Document type: {doc_type}

Full document preview (first 2000 chars):
---
{doc_preview}
---

Chunk to contextualize:
---
{chunk}
---

Write a 2-3 sentence context for this chunk:"""

CONTEXT_EXAMPLE = """Example input:
Document title: Matrix_Mixer
Document type: Q-SYS Designer Help
Chunk: "Each crosspoint can be controlled independently with level adjustments from -100dB to +12dB."

Example output:
This chunk is from the Q-SYS Designer documentation for the Matrix Mixer audio component. It describes the crosspoint level control parameters used for audio signal routing in professional AV installations."""


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

ENTITY_SYSTEM = """You are a technical entity extractor. Extract important named entities from technical documentation.

Entity types to find:
- PRODUCT: Software, hardware, components (e.g., "Q-SYS Core", "Matrix Mixer")
- TECHNOLOGY: Protocols, standards, formats (e.g., "Dante", "AES67", "TCP/IP")
- CONCEPT: Technical concepts, features (e.g., "crosspoint", "gain staging")
- ORGANIZATION: Companies, groups (e.g., "QSC", "Audinate")
- PARAMETER: Settings, values, configurations (e.g., "sample rate", "bit depth")

Rules:
- Return valid JSON array only
- Maximum 8 entities per chunk
- No duplicates
- Be specific, not generic"""

ENTITY_USER = """Extract entities from this technical text:
---
{chunk}
---

Return JSON array like: [{{"name": "Entity Name", "type": "TYPE"}}]
JSON only, no explanation:"""

ENTITY_EXAMPLE = """Example input:
"The Q-SYS Core 110f processor supports Dante and AES67 protocols with 64 channels at 48kHz sample rate."

Example output:
[{"name": "Q-SYS Core 110f", "type": "PRODUCT"}, {"name": "Dante", "type": "TECHNOLOGY"}, {"name": "AES67", "type": "TECHNOLOGY"}, {"name": "sample rate", "type": "PARAMETER"}]"""


# =============================================================================
# SUMMARY EXTRACTION
# =============================================================================

SUMMARY_SYSTEM = """You are a technical writer. Write concise summaries of documentation chunks.

Rules:
- One sentence only
- Focus on the main technical point
- Include key terms/values
- Be specific, not vague"""

SUMMARY_USER = """Summarize this technical documentation in ONE sentence:
---
{chunk}
---

One sentence summary:"""


# =============================================================================
# QUESTIONS ANSWERED
# =============================================================================

QA_SYSTEM = """You are a technical documentation analyst. Identify what questions a documentation chunk can answer.

Rules:
- List 2-3 specific questions
- Questions should be answerable directly from the chunk
- Use technical terminology from the chunk
- Format: one question per line"""

QA_USER = """What specific questions does this documentation chunk answer?
---
{chunk}
---

List 2-3 questions (one per line):"""

QA_EXAMPLE = """Example input:
"The Matrix Mixer supports up to 64 inputs and 64 outputs. Each crosspoint has independent level control from -100dB to +12dB with 0.1dB resolution."

Example output:
How many inputs and outputs does the Matrix Mixer support?
What is the level control range for Matrix Mixer crosspoints?
What is the level adjustment resolution for crosspoints?"""


# =============================================================================
# KEYWORD EXTRACTION (LLM-enhanced)
# =============================================================================

KEYWORD_SYSTEM = """Extract technical keywords and phrases from documentation.

Rules:
- Return 5-10 keywords
- Include technical terms, product names, acronyms
- Comma-separated list only
- Most important terms first"""

KEYWORD_USER = """Extract technical keywords from this text:
---
{chunk}
---

Keywords (comma-separated):"""


# =============================================================================
# Helper to format prompts
# =============================================================================

def format_context_prompt(doc_title: str, doc_type: str, doc_preview: str, chunk: str) -> list:
    """Format the contextual retrieval prompt for Ollama."""
    return [
        {"role": "system", "content": CONTEXT_SYSTEM},
        {"role": "user", "content": CONTEXT_EXAMPLE},
        {"role": "user", "content": CONTEXT_USER.format(
            doc_title=doc_title,
            doc_type=doc_type,
            doc_preview=doc_preview[:2000],
            chunk=chunk[:1500]
        )}
    ]


def format_entity_prompt(chunk: str) -> list:
    """Format the entity extraction prompt for Ollama."""
    return [
        {"role": "system", "content": ENTITY_SYSTEM},
        {"role": "user", "content": ENTITY_EXAMPLE},
        {"role": "user", "content": ENTITY_USER.format(chunk=chunk[:2000])}
    ]


def format_summary_prompt(chunk: str) -> list:
    """Format the summary extraction prompt for Ollama."""
    return [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": SUMMARY_USER.format(chunk=chunk[:2000])}
    ]


def format_qa_prompt(chunk: str) -> list:
    """Format the QA extraction prompt for Ollama."""
    return [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content": QA_EXAMPLE},
        {"role": "user", "content": QA_USER.format(chunk=chunk[:2000])}
    ]


def format_keyword_prompt(chunk: str) -> list:
    """Format the keyword extraction prompt for Ollama."""
    return [
        {"role": "system", "content": KEYWORD_SYSTEM},
        {"role": "user", "content": KEYWORD_USER.format(chunk=chunk[:2000])}
    ]
