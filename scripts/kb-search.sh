#!/bin/bash
# Knowledge Base Search - supports multiple collections
# Usage: kb-search.sh "query" [collection] [limit]
#
# Collections:
#   qsys (default) - Q-SYS documentation
#   <future>       - Add more as needed

QUERY="$1"
COLLECTION="${2:-knowledge_base}"  # Default collection
LIMIT="${3:-5}"

if [ -z "$QUERY" ]; then
    echo "Usage: kb-search.sh \"query\" [collection] [limit]"
    echo ""
    echo "Collections available:"
    curl -s http://localhost:6333/collections 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for c in data.get('result', {}).get('collections', []):
        print(f'  - {c[\"name\"]}')
except: print('  (unable to list)')
"
    exit 1
fi

cd ~/ai/knowledge-base

python3 << EOF
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import sys

try:
    from qdrant_client import QdrantClient
    from FlagEmbedding import BGEM3FlagModel

    query = """$QUERY"""
    collection = "$COLLECTION"
    limit = $LIMIT

    # Embed query
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device='cpu')
    query_emb = model.encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs'][0].tolist()

    # Search
    client = QdrantClient(host='localhost', port=6333)

    # Check if collection exists
    collections = [c.name for c in client.get_collections().collections]
    if collection not in collections:
        print(f"Collection '{collection}' not found. Available: {', '.join(collections)}")
        sys.exit(1)

    results = client.query_points(
        collection_name=collection,
        query=query_emb,
        limit=limit
    )

    print('=' * 60)
    print(f'Knowledge Base Results [{collection}]: {query}')
    print('=' * 60)

    if not results.points:
        print("No results found.")
    else:
        for i, r in enumerate(results.points, 1):
            title = r.payload.get('title', 'Unknown')
            content = r.payload.get('content', '')[:400].replace('\n', ' ')
            score = r.score
            print(f'\n[{i}] {title} (score: {score:.3f})')
            print(f'    {content}...')
    print()

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF
