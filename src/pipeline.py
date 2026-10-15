import time
import numpy as np
from typing import Optional
from contracts import (
    SafetyRoute, 
    PipelineStatus, 
    PipelineResult, 
    ServiceErrorCode
)
from normalizer import normalize_hausa_utterance
from safety_gate import evaluate_safety_gate
from retriever import LightweightHausaRetriever
from verifier import MultiFactorClinicalVerifier
from services import ASRService, LLMService, TTSService
from logger import ZeroPIIEventLogger


class HausaCareSafePipeline:
    def __init__(self, kb_path: str = "data/knowledge_base.json"):
        self.retriever = LightweightHausaRetriever(kb_path)
        self.verifier = MultiFactorClinicalVerifier()
        self.asr_service = ASRService()
        self.llm_service = LLMService()
        self.tts_service = TTSService()
        self.logger = ZeroPIIEventLogger()

    def process_transcript(self, raw_transcript: str, log_turn: bool = True) -> PipelineResult:
        """Text entrypoint for benchmarking and text-only evaluation."""
        t_total_0 = time.perf_counter()
        latencies = {}

        # 1. Normalization
        query = normalize_hausa_utterance(raw_transcript)

        # 2. Safety Gate
        t_g0 = time.perf_counter()
        gate_dec = evaluate_safety_gate(query)
        latencies["gate"] = (time.perf_counter() - t_g0) * 1000

        # Safety Intercepts
        if gate_dec.route != SafetyRoute.ALLOW_RAG:
            latencies["total"] = (time.perf_counter() - t_total_0) * 1000
            res = PipelineResult(
                final_response=gate_dec.deterministic_response,
                safety_route=gate_dec.route,
                pipeline_status=PipelineStatus.SAFETY_INTERCEPT,
                latencies_ms=latencies
            )
            if log_turn:
                self.logger.record_turn(res, query.canonical_hausa)
            return res

        # 3. Grounded Retrieval
        t_r0 = time.perf_counter()
        retrieval_list = self.retriever.retrieve_with_confidence(query.canonical_hausa, k=1)
        latencies["rag"] = (time.perf_counter() - t_r0) * 1000
        ret_result = retrieval_list[0]

        if not ret_result.confidence_passed or not ret_result.card:
            latencies["total"] = (time.perf_counter() - t_total_0) * 1000
            res = PipelineResult(
                final_response="Babu ingantaccen bayanin lafiya da aka samu kan wannan tambayar a halin yanzu.",
                safety_route=SafetyRoute.ALLOW_RAG,
                pipeline_status=PipelineStatus.NO_RETRIEVAL_MATCH,
                retrieval_result=ret_result,
                latencies_ms=latencies
            )
            if log_turn:
                self.logger.record_turn(res, query.canonical_hausa)
            return res

        # 4. Grounded LLM Generation
        llm_res = self.llm_service.generate_grounded(query.canonical_hausa, ret_result.card.hausa_content)
        latencies["llm"] = llm_res.latency_ms

        if not llm_res.success:
            latencies["total"] = (time.perf_counter() - t_total_0) * 1000
            res = PipelineResult(
                final_response="An sami matsala wajen binciko amsar. Don Allah a sake gwadawa.",
                safety_route=SafetyRoute.ALLOW_RAG,
                pipeline_status=PipelineStatus.LLM_FAILURE,
                retrieval_result=ret_result,
                service_error_code=llm_res.error_code,
                latencies_ms=latencies
            )
            if log_turn:
                self.logger.record_turn(res, query.canonical_hausa)
            return res

        # 5. Tri-Partite Clinical Verification
        t_v0 = time.perf_counter()
        ver_res = self.verifier.verify_candidate_output(
            candidate_text=llm_res.payload,
            canonical_query=query.canonical_hausa,
            grounding_card=ret_result.card
        )
        latencies["verifier"] = (time.perf_counter() - t_v0) * 1000
        status = PipelineStatus.GROUNDED_SUCCESS if ver_res.passed else PipelineStatus.VERIFIER_REJECTION

        latencies["total"] = (time.perf_counter() - t_total_0) * 1000
        res = PipelineResult(
            final_response=ver_res.final_output,
            safety_route=SafetyRoute.ALLOW_RAG,
            pipeline_status=status,
            retrieval_result=ret_result,
            verification_result=ver_res,
            latencies_ms=latencies
        )
        if log_turn:
            self.logger.record_turn(res, query.canonical_hausa)
        return res

    def process_audio(self, audio_array_16k: np.ndarray, audio_duration: float = 0.0) -> PipelineResult:
        """End-to-end voice entrypoint: 16k Audio Array -> ASR -> Core Pipeline -> TTS."""
        t_total_0 = time.perf_counter()
        
        # 1. Speech Recognition (using 16kHz float32 numpy array)
        asr_res = self.asr_service.transcribe_array_16k(audio_array_16k)
        if not asr_res.success:
            res = PipelineResult(
                final_response="Ba a iya fahimtar sautin muryar ba. Don Allah a sake gwadawa.",
                safety_route=SafetyRoute.ALLOW_RAG,
                pipeline_status=PipelineStatus.ASR_FAILURE,
                latencies_ms={"asr": asr_res.latency_ms, "total": asr_res.latency_ms},
                service_error_code=asr_res.error_code
            )
            self.logger.record_turn(res, "[ASR_FAILED]", audio_duration)
            return res

        # 2. Execute Core Pipeline via Transcript (skip internal log)
        pipeline_res = self.process_transcript(asr_res.payload, log_turn=False)
        updated_latencies = dict(pipeline_res.latencies_ms)
        updated_latencies["asr"] = asr_res.latency_ms

        # 3. Speech Synthesis (TTS)
        tts_res = self.tts_service.synthesize(pipeline_res.final_response)
        updated_latencies["tts"] = tts_res.latency_ms
        updated_latencies["total"] = (time.perf_counter() - t_total_0) * 1000

        audio_path = tts_res.payload if tts_res.success else None
        final_status = pipeline_res.pipeline_status  # Retain clinical pipeline status
        
        final_res = PipelineResult(
            final_response=pipeline_res.final_response,
            safety_route=pipeline_res.safety_route,
            pipeline_status=final_status,
            audio_path=audio_path,
            retrieval_result=pipeline_res.retrieval_result,
            verification_result=pipeline_res.verification_result,
            latencies_ms=updated_latencies,
            service_error_code=tts_res.error_code if not tts_res.success else pipeline_res.service_error_code,
            metadata={"raw_transcript": asr_res.payload}
        )

        # 4. Single authoritative log
        norm_q = normalize_hausa_utterance(asr_res.payload)
        self.logger.record_turn(final_res, norm_q.canonical_hausa, audio_duration)

        return final_res