"""Embedding service using BGE-M3 for local embeddings."""
import torch
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from functools import lru_cache
from rich.console import Console

console = Console()


class EmbeddingService:
    """Embedding service with support for multiple embedding types."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self._model = None
        self._dimension = None

    @property
    def model(self) -> BGEM3FlagModel:
        """Lazy load the model."""
        if self._model is None:
            console.print(f"[bold blue]Loading embedding model: {self.model_name}[/bold blue]")
            console.print(f"[dim]Device: {self.device}, FP16: {self.use_fp16}[/dim]")

            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device
            )
            console.print("[bold green]Model loaded successfully[/bold green]")
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            # BGE-M3 has 1024 dimensions
            self._dimension = 1024
        return self._dimension

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert: bool = False
    ) -> dict:
        """
        Encode texts into embeddings.

        BGE-M3 supports three types of embeddings:
        - Dense: Standard dense vectors (1024 dim)
        - Sparse: Lexical/sparse vectors for hybrid search
        - ColBERT: Multi-vector representations for late interaction

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            return_dense: Return dense embeddings
            return_sparse: Return sparse embeddings
            return_colbert: Return ColBERT embeddings

        Returns:
            Dictionary with requested embedding types
        """
        if isinstance(texts, str):
            texts = [texts]

        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert
        )

        result = {}

        if return_dense:
            result["dense"] = output["dense_vecs"]
        if return_sparse:
            result["sparse"] = output["lexical_weights"]
        if return_colbert:
            result["colbert"] = output["colbert_vecs"]

        return result

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query for retrieval (returns dense vector)."""
        output = self.encode(query, return_dense=True)
        return output["dense"][0]

    def encode_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Encode documents for indexing (returns dense vectors)."""
        output = self.encode(
            documents,
            batch_size=batch_size,
            show_progress=show_progress,
            return_dense=True
        )
        return output["dense"]

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and documents."""
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(
            document_embeddings, axis=1, keepdims=True
        )
        # Compute cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        return similarities


# Global embedding service instance
@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service."""
    return EmbeddingService()
