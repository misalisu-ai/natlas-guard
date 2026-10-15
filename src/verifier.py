"""
Multi-factor clinical safety verifier.
Enforces Query Relevance + Evidence Grounding + Safety Boundaries.
"""
import re
from typing import Set
from contracts import VerificationResult, KnowledgeCard


def hausa_stem(word: str) -> str:
    """Lightweight morphological stemmer for Hausa noun/verb inflections."""
    w = word.lower().strip()
    for suffix in ("na", "ta", "sa", "rsa", "nmu", "tmu", "nku", "nsu", "in", "an", "awa", "e"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w


class MultiFactorClinicalVerifier:
    def __init__(self):
        self.dosage_patterns = [
            r"\b\d+\s*(mg|ml|miligram|gram|kwaya)\b",
            r"\bsha\s+sau\s+\d+\b",
            r"\bsha\s+kwaya\s+\d+\b",
            r"\ballura[r]?\s+sau\s+\d+\b"
        ]

        self.diagnostic_patterns = [
            r"\b(alamomin\s+ka\s+na\s+nuna\s+)?(kana|kina)\s+da\s+(ciwon\s+)?\w+\b",
            r"\bwannan\s+yana\s+nufin\s+(kana|kina)\s+da\s+\w+\b",
            r"\b(ka|kin)\s+kamu\s+da\s+cutar\b",
            r"\bwannan\s+cuta\s+ce\s+ta\b"
        ]

        self.verifier_fallback_hausa = (
            "SANARWA: Ba za mu iya tabbatar da cikakken amincin wannan shawarar ba. "
            "Don kare lafiyarka, ziyarci cibiyar lafiya mafi kusa don samun "
            "sahihin bincike da magani daga kwararren jami'in lafiya."
        )

    def verify_candidate_output(
        self, 
        candidate_text: str, 
        canonical_query: str, 
        grounding_card: KnowledgeCard
    ) -> VerificationResult:
        norm_resp = candidate_text.lower()

        # 1. Dosage Leakage Audit
        leaked_dosages = []
        for pat in self.dosage_patterns:
            matches = re.findall(pat, norm_resp)
            if matches:
                leaked_dosages.extend([m if isinstance(m, str) else m[0] for m in matches])

        if leaked_dosages:
            return VerificationResult(
                passed=False,
                final_output=self.verifier_fallback_hausa,
                flag_reason="DOSAGE_LEAKAGE_DETECTED",
                hallucinated_dosages=leaked_dosages
            )

        # 2. Diagnostic Assertion Audit
        for pat in self.diagnostic_patterns:
            if re.search(pat, norm_resp):
                return VerificationResult(
                    passed=False,
                    final_output=self.verifier_fallback_hausa,
                    flag_reason="UNAUTHORIZED_DIAGNOSTIC_CLAIM"
                )

        # 3. Query Lexical Relevance Audit
        query_stems: Set[str] = {hausa_stem(w) for w in canonical_query.lower().split() if len(w) > 3}
        resp_stems: Set[str] = {hausa_stem(w) for w in norm_resp.split() if len(w) > 3}
        card_stems: Set[str] = {hausa_stem(w) for w in grounding_card.hausa_content.lower().split() if len(w) > 3}

        query_overlap = len(query_stems.intersection(resp_stems.union(card_stems)))
        if query_stems and query_overlap == 0:
            return VerificationResult(
                passed=False,
                final_output=self.verifier_fallback_hausa,
                flag_reason="QUERY_RESPONSE_DIVERGENCE",
                query_relevance_score=0.0,
                evidence_overlap_ratio=0.0
            )

        # 4. Evidence Grounding Audit
        grounding_ratio = 0.0
        if resp_stems:
            grounded_stem_count = sum(1 for s in resp_stems if s in card_stems)
            grounding_ratio = grounded_stem_count / len(resp_stems)
            if grounding_ratio < 0.20:
                return VerificationResult(
                    passed=False,
                    final_output=self.verifier_fallback_hausa,
                    flag_reason="INSUFFICIENT_EVIDENCE_GROUNDING",
                    query_relevance_score=float(query_overlap),
                    evidence_overlap_ratio=grounding_ratio
                )

        return VerificationResult(
            passed=True,
            final_output=candidate_text,
            flag_reason="CLEAN",
            query_relevance_score=float(query_overlap),
            evidence_overlap_ratio=round(grounding_ratio, 4)
        )