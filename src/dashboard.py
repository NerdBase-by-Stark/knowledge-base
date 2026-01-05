"""Streamlit Dashboard for Knowledge Base visualization."""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
import pandas as pd
import tempfile
from pathlib import Path
import json

# Page config
st.set_page_config(
    page_title="Knowledge Base Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize knowledge base
@st.cache_resource
def get_knowledge_base():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    kb.initialize(recreate=False)
    return kb


def render_sidebar():
    """Render the sidebar with navigation and stats."""
    st.sidebar.title("🧠 Knowledge Base")

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["🔍 Search", "📥 Ingest", "🕸️ Graph Explorer", "📊 Statistics"]
    )

    st.sidebar.divider()

    # Quick stats
    try:
        kb = get_knowledge_base()
        stats = kb.get_stats()

        st.sidebar.subheader("Quick Stats")
        col1, col2 = st.sidebar.columns(2)

        vs = stats.get("vector_store", {})
        gs = stats.get("graph_store", {})

        col1.metric("Vectors", vs.get("vectors_count", 0))
        col2.metric("Entities", gs.get("entities", 0))

        col1.metric("Documents", gs.get("documents", 0))
        col2.metric("Relations", gs.get("relationships", 0))

    except Exception as e:
        st.sidebar.error(f"Error loading stats: {e}")

    return page


def render_search_page():
    """Render the search interface."""
    st.title("🔍 Search Knowledge Base")

    kb = get_knowledge_base()

    # Search controls
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        query = st.text_input("Search query", placeholder="Enter your search query...")

    with col2:
        search_mode = st.selectbox("Mode", ["vector", "hybrid", "graph"])

    with col3:
        limit = st.number_input("Results", min_value=1, max_value=50, value=10)

    # Collection filter
    collections = kb.vector_store.list_collections()
    collection = st.selectbox(
        "Filter by collection",
        ["All"] + collections,
        index=0
    )

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            results = kb.search(
                query=query,
                limit=limit,
                collection=None if collection == "All" else collection,
                mode=search_mode
            )

        if not results:
            st.info("No results found.")
        else:
            st.subheader(f"Found {len(results)} results")

            for i, result in enumerate(results):
                with st.expander(
                    f"**{result.document_title}** (Score: {result.score:.3f})",
                    expanded=i < 3
                ):
                    st.markdown(result.content)

                    col1, col2, col3 = st.columns(3)
                    col1.caption(f"Source: {result.source}")
                    col2.caption(f"Doc ID: {result.document_id[:8]}...")
                    if result.chunk_index is not None:
                        col3.caption(f"Chunk: {result.chunk_index}")


def render_ingest_page():
    """Render the ingestion interface."""
    st.title("📥 Ingest Documents")

    kb = get_knowledge_base()

    tab1, tab2, tab3 = st.tabs(["📄 Text", "📁 File", "📂 Directory"])

    with tab1:
        st.subheader("Ingest Text")
        title = st.text_input("Title", placeholder="Document title")
        text = st.text_area("Content", height=200, placeholder="Paste your content here...")
        collection = st.text_input("Collection", value="default")
        source_url = st.text_input("Source URL (optional)")

        if st.button("Ingest Text", type="primary"):
            if text:
                with st.spinner("Processing..."):
                    doc_id = kb.ingest_text(
                        text=text,
                        title=title or "Untitled",
                        collection=collection,
                        source_url=source_url if source_url else None
                    )
                st.success(f"Ingested document: {doc_id}")
            else:
                st.warning("Please enter some text to ingest.")

    with tab2:
        st.subheader("Ingest File")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "pptx", "xlsx", "html", "md", "txt"]
        )
        file_collection = st.text_input("Collection", value="default", key="file_collection")

        if uploaded_file and st.button("Ingest File", type="primary"):
            # Save to temp file
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            with st.spinner("Processing document..."):
                try:
                    doc_id = kb.ingest_file(tmp_path, collection=file_collection)
                    st.success(f"Ingested: {uploaded_file.name} (ID: {doc_id})")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

    with tab3:
        st.subheader("Ingest Directory")
        directory = st.text_input("Directory path", placeholder="/path/to/documents")
        dir_collection = st.text_input("Collection", value="default", key="dir_collection")
        recursive = st.checkbox("Recursive", value=True)

        col1, col2 = st.columns(2)
        with col1:
            extensions = st.multiselect(
                "File types",
                ["pdf", "docx", "md", "txt", "html"],
                default=["md", "pdf"]
            )

        if st.button("Ingest Directory", type="primary"):
            if directory and Path(directory).is_dir():
                with st.spinner("Processing directory..."):
                    try:
                        doc_ids = kb.ingest_directory(
                            directory=directory,
                            collection=dir_collection,
                            recursive=recursive,
                            extensions=extensions if extensions else None
                        )
                        st.success(f"Ingested {len(doc_ids)} documents")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a valid directory path.")


def render_graph_page():
    """Render the knowledge graph explorer."""
    st.title("🕸️ Graph Explorer")

    kb = get_knowledge_base()

    # Entity search
    entity_query = st.text_input("Search for entity", placeholder="Enter entity name...")

    col1, col2 = st.columns(2)
    with col1:
        depth = st.slider("Exploration depth", min_value=1, max_value=3, value=2)
    with col2:
        layout = st.selectbox("Layout", ["barnes_hut", "force_atlas", "hierarchical"])

    if entity_query:
        # Search for entities
        entities = kb.graph_store.search_entities(entity_query, limit=10)

        if entities:
            st.subheader("Matching Entities")

            for entity in entities:
                if st.button(f"🔹 {entity['name']} ({entity['type']})", key=entity['name']):
                    # Get entity context
                    context = kb.get_entity_context(entity['name'], depth=depth)

                    # Display related entities
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Related Entities")
                        if context['related_entities']:
                            for rel in context['related_entities'][:10]:
                                st.write(f"- **{rel['name']}** ({rel['type']}) - distance: {rel['distance']}")
                        else:
                            st.info("No related entities found")

                    with col2:
                        st.subheader("Mentioned In")
                        if context['documents']:
                            for doc in context['documents'][:10]:
                                st.write(f"- {doc['title']} ({doc['mention_count']} mentions)")
                        else:
                            st.info("No documents found")

                    # Render graph visualization
                    graph_data = context['graph']
                    if graph_data['nodes']:
                        st.subheader("Knowledge Graph")
                        render_graph_visualization(graph_data, layout)
        else:
            st.info("No entities found matching your query.")

    # Show graph statistics
    st.divider()
    st.subheader("Graph Statistics")

    stats = kb.graph_store.get_graph_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", stats.get("documents", 0))
    col2.metric("Chunks", stats.get("chunks", 0))
    col3.metric("Entities", stats.get("entities", 0))
    col4.metric("Relationships", stats.get("relationships", 0))


def render_graph_visualization(graph_data: dict, layout: str = "barnes_hut"):
    """Render an interactive graph visualization using PyVis."""
    if not graph_data.get('nodes'):
        st.info("No graph data to visualize")
        return

    # Create network
    net = Network(
        height="500px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white"
    )

    # Configure physics
    if layout == "barnes_hut":
        net.barnes_hut()
    elif layout == "force_atlas":
        net.force_atlas_2based()
    else:
        net.set_options("""
        {
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "UD"
                }
            }
        }
        """)

    # Color mapping for node types
    colors = {
        "Entity": "#4CAF50",
        "Document": "#2196F3",
        "Chunk": "#9C27B0",
        "Person": "#FF9800",
        "Organization": "#E91E63",
        "Concept": "#00BCD4",
        "Unknown": "#607D8B"
    }

    # Add nodes
    for node in graph_data['nodes']:
        color = colors.get(node.get('type', 'Unknown'), colors['Unknown'])
        net.add_node(
            node['id'],
            label=node['label'],
            color=color,
            title=f"{node['label']} ({node.get('type', 'Unknown')})"
        )

    # Add edges
    for edge in graph_data['edges']:
        if edge['source'] and edge['target']:
            net.add_edge(
                edge['source'],
                edge['target'],
                title=edge.get('type', ''),
                arrows="to"
            )

    # Save to temp file and display
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            html = f.read()
        st.components.v1.html(html, height=550, scrolling=True)
        Path(tmp.name).unlink(missing_ok=True)


def render_stats_page():
    """Render the statistics page."""
    st.title("📊 Statistics")

    kb = get_knowledge_base()
    stats = kb.get_stats()

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    vs = stats.get("vector_store", {})
    gs = stats.get("graph_store", {})

    col1.metric("Total Vectors", vs.get("vectors_count", 0))
    col2.metric("Documents", gs.get("documents", 0))
    col3.metric("Entities", gs.get("entities", 0))
    col4.metric("Relationships", gs.get("relationships", 0))

    st.divider()

    # Vector store details
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vector Store (Qdrant)")
        st.json(vs)

    with col2:
        st.subheader("Graph Store (Neo4j)")
        st.json(gs)

    # Collections
    st.divider()
    st.subheader("Collections")

    collections = kb.vector_store.list_collections()
    if collections:
        for coll in collections:
            st.write(f"- {coll}")
    else:
        st.info("No collections found")

    # Links to external dashboards
    st.divider()
    st.subheader("External Dashboards")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Qdrant Dashboard**
        - URL: [localhost:6333/dashboard](http://localhost:6333/dashboard)
        - Vector space visualization
        - Collection management
        """)

    with col2:
        st.markdown("""
        **Neo4j Browser**
        - URL: [localhost:7474](http://localhost:7474)
        - Graph visualization
        - Cypher queries
        """)

    with col3:
        st.markdown("""
        **API Documentation**
        - URL: [localhost:8000/docs](http://localhost:8000/docs)
        - Swagger UI
        - API testing
        """)


def main():
    """Main application entry point."""
    page = render_sidebar()

    if "Search" in page:
        render_search_page()
    elif "Ingest" in page:
        render_ingest_page()
    elif "Graph" in page:
        render_graph_page()
    elif "Statistics" in page:
        render_stats_page()


if __name__ == "__main__":
    main()
