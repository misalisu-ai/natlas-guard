"""
Core immutable contracts across Phase 1 and Phase 2.
Enforces typed stage outputs, execution statuses, and service failure taxonomies.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class SafetyRoute(str, Enum):
    EMERGENCY_ESCALATION = "EMERGENCY_ESCALATION"      # Priority 1: Acute emergency referral
    PRESCRIPTION_INTERCEPT = "PRESCRIPTION_INTERCEPT"  # Priority 2: Safe drug/dosage refusal
    SAFE_ABSTENTION = "SAFE_ABSTENTION"                # Priority 3: Harmful remedy abstention
    UNCERTAIN_AMBIGUOUS = "UNCERTAIN_AMBIGUOUS"        # Priority 4: Under-specified elicitation
    ALLOW_RAG = "ALLOW_RAG"                            # Priority 5: Safe primary health inquiry


class PipelineStatus(str, Enum):
    SAFETY_INTERCEPT = "SAFETY_INTERCEPT"          # Successfully intercepted by safety gate
    GROUNDED_SUCCESS = "GROUNDED_SUCCESS"          # Completed RAG + N-ATLAS + Verification pass
    VERIFIER_REJECTION = "VERIFIER_REJECTION"      # Candidate failed clinical output check; fallback used
    NO_RETRIEVAL_MATCH = "NO_RETRIEVAL_MATCH"      # Retrieval score fell below confidence threshold
    ASR_FAILURE = "ASR_FAILURE"                    # Speech-to-text service error or timeout
    LLM_FAILURE = "LLM_FAILURE"                    # Language model API failure or timeout
    TTS_FAILURE = "TTS_FAILURE"                    # Speech synthesis failure


class ServiceErrorCode(str, Enum):
    NONE = "NONE"
    TIMEOUT = "TIMEOUT"
    AUTH_FAILURE = "AUTH_FAILURE"
    RATE_LIMITED = "RATE_LIMITED"
    MODEL_LOADING = "MODEL_LOADING"
    SERVER_ERROR = "SERVER_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    EMPTY_INPUT = "EMPTY_INPUT"
    INVALID_PAYLOAD = "INVALID_PAYLOAD"


@dataclass(frozen=True)
class NormalizedQuery:
    raw_text: str
    canonical_hausa: str         # Accents, hooked consonants intact for linguistic fidelity
    routing_representation: str  # ASCII-folded representation for deterministic rule matching


@dataclass(frozen=True)
class GateDecision:
    route: SafetyRoute
    deterministic_response: str
    confidence: float
    trigger_rule: Optional[str] = None
    negation_detected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeCard:
    card_id: str
    topic: str
    hausa_content: str
    english_reference: str
    source_entity: str
    source_title: str
    document_version: str
    publication_year: int
    source_locator: Dict[str, str]
    document_sha256: str
    evidence_type: str


@dataclass(frozen=True)
class RetrievalResult:
    card: Optional[KnowledgeCard]
    score: float
    rank: int
    retrieval_method: str
    num_candidates_evaluated: int
    confidence_passed: bool


@dataclass(frozen=True)
class ServiceResult:
    success: bool
    payload: str
    latency_ms: float
    error_code: ServiceErrorCode = ServiceErrorCode.NONE


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    final_output: str
    flag_reason: str
    query_relevance_score: float = 1.0
    evidence_overlap_ratio: float = 1.0
    hallucinated_dosages: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineResult:
    final_response: str
    safety_route: SafetyRoute
    pipeline_status: PipelineStatus
    audio_path: Optional[str] = None
    retrieval_result: Optional[RetrievalResult] = None
    verification_result: Optional[VerificationResult] = None
    latencies_ms: Dict[str, float] = field(default_factory=dict)
    service_error_code: ServiceErrorCode = ServiceErrorCode.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)