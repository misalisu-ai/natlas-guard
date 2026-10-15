import json
from typing import List, Dict, Any
import numpy as np
from contracts import RetrievalResult, KnowledgeCard

class LightweightHausaRetriever:
    """
    Compute-efficient Hybrid Retriever for Colab CPU.
    Combines BM25 token matching with lightweight dense embeddings.
    """
    def __init__(self, kb_path: str):
        with open(kb_path, "r", encoding="utf-8") as f:
            self.cards: List[Dict[str, Any]] = json.load(f)
            
        self.corpus_hausa = [card["hausa_content"] for card in self.cards]
        self._init_bm25()
        self._init_dense_index()

    def _init_bm25(self):
        """Initializes tokenized word-frequency tables for BM25."""
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus_hausa]
        self.doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        self.avg_dl = sum(self.doc_lengths) / (len(self.doc_lengths) or 1)
        self.num_docs = len(self.tokenized_corpus)
        
        # Calculate Term Frequencies and Document Frequencies
        self.dfs = {}
        for doc in self.tokenized_corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.dfs[term] = self.dfs.get(term, 0) + 1

    def _compute_bm25_score(self, query_tokens: List[str], k1=1.5, b=0.75) -> np.ndarray:
        scores = np.zeros(self.num_docs)
        for term in query_tokens:
            if term not in self.dfs:
                continue
            df = self.dfs[term]
            idf = np.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            for i, doc in enumerate(self.tokenized_corpus):
                tf = doc.count(term)
                denom = tf + k1 * (1 - b + b * (self.doc_lengths[i] / self.avg_dl))
                if denom > 0:
                    scores[i] += idf * (tf * (k1 + 1)) / denom
        return scores

    def _init_dense_index(self):
        """Loads lightweight sentence embedder for semantic alignment."""
        try:
            from sentence_transformers import SentenceTransformer
            # Lightweight: ~117M parameters, runs on Colab CPU in <200ms
            self.encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            self.dense_embeddings = self.encoder.encode(
                self.corpus_hausa, 
                show_progress_bar=False, 
                normalize_embeddings=True
            )
            self.dense_available = True
        except ImportError:
            # Graceful fallback to pure BM25 if library is omitted
            self.dense_available = False

    # Calibrated empirical cutoff from 100-item benchmark analysis
    # Prevents out-of-domain queries from producing false-positive card bindings
    CALIBRATED_MIN_RETRIEVAL_CONFIDENCE = 0.65

    def retrieve_with_confidence(
        self, 
        canonical_query: str, 
        k: int = 1, 
        alpha: float = 0.5
    ) -> List[RetrievalResult]:
        from contracts import RetrievalResult, KnowledgeCard
        
        query_tokens = canonical_query.lower().split()
        bm25_raw = self._compute_bm25_score(query_tokens)
        bm25_norm = bm25_raw / (np.max(bm25_raw) + 1e-9) if np.max(bm25_raw) > 0 else bm25_raw

        if self.dense_available:
            query_emb = self.encoder.encode([canonical_query], normalize_embeddings=True)[0]
            dense_sim = np.dot(self.dense_embeddings, query_emb)
            final_scores = (alpha * dense_sim) + ((1.0 - alpha) * bm25_norm)
            method = "bm25_minilm_hybrid"
        else:
            final_scores = bm25_norm
            method = "bm25_exact"

        top_indices = np.argsort(final_scores)[::-1][:k]
        results = []

        for rank_idx, idx in enumerate(top_indices, start=1):
            score = float(final_scores[idx])
            raw_card = self.cards[idx]
            card_id = raw_card.get("card_id", f"UNKNOWN_{idx}")
            
            prov = raw_card.get("provenance")
            if not prov:
                raise ValueError(f"CRITICAL: Missing provenance block in KB card '{card_id}'")
            if not prov.get("document_sha256") or prov.get("document_sha256") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
                raise ValueError(f"CRITICAL: Missing authentic document_sha256 for KB card '{card_id}'")
            if not prov.get("source_locator"):
                raise ValueError(f"CRITICAL: Missing source_locator for KB card '{card_id}'")

            k_card = KnowledgeCard(
                card_id=card_id,
                topic=raw_card.get("topic", ""),
                hausa_content=raw_card.get("hausa_content", ""),
                english_reference=raw_card.get("english_reference", ""),
                source_entity=prov.get("source_entity", ""),
                source_title=prov.get("source_title", ""),
                document_version=prov.get("document_version", ""),
                publication_year=int(prov.get("publication_year", 2020)),
                source_locator=prov.get("source_locator"),
                document_sha256=prov.get("document_sha256"),
                evidence_type=prov.get("evidence_type", "guideline")
            )
            
            passed = score >= self.CALIBRATED_MIN_RETRIEVAL_CONFIDENCE
            results.append(
                RetrievalResult(
                    card=k_card if passed else None,
                    score=score,
                    rank=rank_idx,
                    retrieval_method=method,
                    num_candidates_evaluated=self.num_docs,
                    confidence_passed=passed
                )
            )

        return results