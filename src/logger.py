import csv
import os
import hashlib
from datetime import datetime, timezone
from contracts import PipelineResult

class ZeroPIIEventLogger:
    def __init__(self, log_path: str = "logs/event_audit.csv"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self.fields = [
            "query_hash",
            "timestamp_utc",
            "audio_duration_sec",
            "word_count_whitespace",
            "safety_route",
            "pipeline_status",
            "retrieved_card_id",
            "retrieval_score",
            "verifier_flag",
            "total_latency_ms",
            "asr_latency_ms",
            "safety_gate_latency_ms",
            "rag_latency_ms",
            "llm_latency_ms",
            "verifier_latency_ms",
            "tts_latency_ms",
            "service_error_code"
        ]
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def record_turn(self, result: PipelineResult, query_text_for_hash: str, audio_duration: float = 0.0):
        # Daily salted query hash for deduplication without storing text
        salt = datetime.now(timezone.utc).strftime("%Y%m%d")
        query_hash = hashlib.sha256(f"{salt}_{query_text_for_hash}".encode("utf-8")).hexdigest()[:12]

        card_id = "NONE"
        score = 0.0
        if result.retrieval_result and result.retrieval_result.card:
            card_id = result.retrieval_result.card.card_id
            score = result.retrieval_result.score

        ver_flag = result.verification_result.flag_reason if result.verification_result else "BYPASSED"

        row = {
            "query_hash": query_hash,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "audio_duration_sec": round(audio_duration, 2),
            "word_count_whitespace": len(query_text_for_hash.split()),
            "safety_route": result.safety_route.value,
            "pipeline_status": result.pipeline_status.value,
            "retrieved_card_id": card_id,
            "retrieval_score": round(score, 4),
            "verifier_flag": ver_flag,
            "total_latency_ms": round(result.latencies_ms.get("total", 0.0), 1),
            "asr_latency_ms": round(result.latencies_ms.get("asr", 0.0), 1),
            "safety_gate_latency_ms": round(result.latencies_ms.get("gate", 0.0), 2),
            "rag_latency_ms": round(result.latencies_ms.get("rag", 0.0), 1),
            "llm_latency_ms": round(result.latencies_ms.get("llm", 0.0), 1),
            "verifier_latency_ms": round(result.latencies_ms.get("verifier", 0.0), 2),
            "tts_latency_ms": round(result.latencies_ms.get("tts", 0.0), 1),
            "service_error_code": result.service_error_code.value
        }

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(row)