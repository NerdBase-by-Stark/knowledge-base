#!/bin/bash
cd ~/ai/knowledge-base

echo "$(date): Starting remaining benchmarks..."

# Run Qwen3
echo "$(date): Running Qwen3..."
python scripts/run_qwen3.py
echo "$(date): Qwen3 done"

# Run Mistral Nemo
echo "$(date): Running Mistral Nemo..."  
python scripts/run_mistral_nemo.py
echo "$(date): Mistral Nemo done"

# Run deepseek (create script first)
cat > /tmp/run_deepseek.py << 'PYEOF'
#!/usr/bin/env python3
import json, time, hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict
from pathlib import Path
import ollama
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

console = Console()
MODEL = "deepseek-coder:33b"
TEST_DOCS = ["3_-_The_Language.md", "HttpClient.md", "TcpSocket.md", "Uci.md", "State_Trigger.md"]
DOCS_DIR = Path.home() / "ai/knowledge-base/scrape-jobs/qsys-lua/markdown"
OUTPUT_DIR = Path.home() / "ai/knowledge-base/scrape-jobs/qsys-lua/shootout"
CHUNK_SIZE, CHUNK_OVERLAP = 512, 75

CONTEXT_PROMPT = """You are a technical documentation assistant. Write a brief context statement (2-3 sentences) that situates this chunk within its source document.
Document: {doc_title}
Chunk:
---
{chunk}
---
Rules:
- Mention document source and topic
- Include key technical terms
- Do NOT summarize the chunk itself
Context (2-3 sentences only):"""

ENTITY_PROMPT = """Extract technical entities from this text. Return JSON array only.
Entity types: PRODUCT, TECHNOLOGY, CONCEPT, PARAMETER, ORGANIZATION
Text:
---
{chunk}
---
Return JSON like: [{{"name": "Entity", "type": "TYPE"}}]
Maximum 8 entities. JSON only:"""

SUMMARY_PROMPT = """Summarize this technical documentation in ONE sentence:
---
{chunk}
---
One sentence summary:"""

QA_PROMPT = """What questions can this documentation answer? List 2-3 specific questions.
---
{chunk}
---
Questions (one per line):"""

@dataclass
class ChunkResult:
    chunk_id: str
    chunk_text: str
    context: str = ""
    entities: List[Dict] = field(default_factory=list)
    summary: str = ""
    questions: List[str] = field(default_factory=list)
    context_time_ms: float = 0
    entity_time_ms: float = 0
    summary_time_ms: float = 0
    qa_time_ms: float = 0
    errors: List[str] = field(default_factory=list)

def simple_chunk(text, chunk_size=512, overlap=75):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start+chunk_size]))
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]

def call_llm(prompt, max_tokens=300):
    start = time.time()
    try:
        r = ollama.chat(model=MODEL, messages=[{"role":"user","content":prompt}], options={"temperature":0.3,"num_predict":max_tokens})
        return r["message"]["content"], (time.time()-start)*1000
    except Exception as e:
        return f"ERROR: {e}", (time.time()-start)*1000

def parse_entities(text):
    try:
        s, e = text.find("["), text.rfind("]")+1
        if s >= 0 and e > s: return json.loads(text[s:e])
    except: pass
    return []

def parse_questions(text):
    return [l.strip().lstrip("0123456789.)").strip() for l in text.strip().split("\n") if l.strip() and not l.startswith("#")][:3]

def main():
    console.print(f"[bold cyan]Running {MODEL}[/]")
    docs = {}
    for d in TEST_DOCS:
        p = DOCS_DIR / d
        if p.exists():
            docs[d] = {"title": d.replace(".md","").replace("_"," "), "chunks": simple_chunk(p.read_text())}
            console.print(f"  {d}: {len(docs[d]['chunks'])} chunks")
    
    all_chunks = [{"id": hashlib.md5(f"{d}:{i}".encode()).hexdigest()[:8], "title": docs[d]["title"], "text": c} 
                  for d in docs for i, c in enumerate(docs[d]["chunks"])]
    
    results, total_entities, errors, start = [], 0, 0, time.time()
    with Progress(TextColumn(f"[cyan]{MODEL}[/]"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn()) as p:
        t = p.add_task("", total=len(all_chunks))
        for cd in all_chunks:
            cr = ChunkResult(cd["id"], cd["text"][:200])
            cr.context, cr.context_time_ms = call_llm(CONTEXT_PROMPT.format(doc_title=cd["title"], chunk=cd["text"]), 150)
            r, cr.entity_time_ms = call_llm(ENTITY_PROMPT.format(chunk=cd["text"]), 300)
            cr.entities = parse_entities(r)
            total_entities += len(cr.entities)
            cr.summary, cr.summary_time_ms = call_llm(SUMMARY_PROMPT.format(chunk=cd["text"]), 100)
            r, cr.qa_time_ms = call_llm(QA_PROMPT.format(chunk=cd["text"]), 200)
            cr.questions = parse_questions(r)
            results.append(cr)
            p.advance(t)
    
    total_time = time.time() - start
    console.print(f"[green]Done! {total_time:.1f}s, {total_entities} entities[/]")
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "deepseek-coder-33b.json", "w") as f:
        json.dump({"model": MODEL, "total_chunks": len(all_chunks), "total_time_s": total_time, 
                   "avg_chunk_time_ms": total_time*1000/len(all_chunks), "total_entities": total_entities,
                   "error_count": errors, "chunks": [asdict(c) for c in results]}, f, indent=2)

if __name__ == "__main__": main()
PYEOF

echo "$(date): Running DeepSeek Coder..."
python /tmp/run_deepseek.py
echo "$(date): DeepSeek done"

echo "$(date): ALL BENCHMARKS COMPLETE!"
ls -la ~/ai/knowledge-base/scrape-jobs/qsys-lua/shootout/*.json
