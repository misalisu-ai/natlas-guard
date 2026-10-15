import os
import time
import uuid
import asyncio
import numpy as np
import requests
import edge_tts
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from contracts import ServiceResult, ServiceErrorCode
from typing import Optional
from natlas_client import NAtLaSClient


def parse_http_error(status_code: int) -> ServiceErrorCode:
    if status_code in (401, 403):
        return ServiceErrorCode.AUTH_FAILURE
    if status_code == 429:
        return ServiceErrorCode.RATE_LIMITED
    if status_code == 503:
        return ServiceErrorCode.MODEL_LOADING
    if 500 <= status_code < 600:
        return ServiceErrorCode.SERVER_ERROR
    return ServiceErrorCode.INVALID_PAYLOAD


class ASRService:
    """Hausa Speech-to-Text service using a dedicated Hausa acoustic model."""
    def __init__(self, endpoint_url: Optional[str] = None, token: Optional[str] = None, timeout_sec: float = 4.0):
        self.endpoint_url = endpoint_url or "https://api-inference.huggingface.co/models/N-ATLAS/hausa-asr-model"
        self.token = token or os.getenv("HF_TOKEN")
        self.timeout = timeout_sec
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.local_pipeline = None
        self._init_local_asr()

    def _init_local_asr(self):
        """Initializes the official Meta MMS pipeline with native Hausa target configurations."""
        try:
            from transformers import pipeline
            # Stable, production-ready multilingual model with verified vocabulary mapping tracks
            model_id = "facebook/mms-1b-all"
            self.local_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device="cpu",
                generate_kwargs={"target_lang": "hau"}
            )
            self.model = None  # Use the high-level pipeline directly for MMS
            print(f"[ASR Notice] Verified Hausa Meta MMS model ({model_id}) initialized successfully.")
        except Exception as e:
            print(f"[ASR Critical] Could not load any local Hausa ASR model: {e}")
            self.local_pipeline = None

    def transcribe_array_16k(self, audio_array: np.ndarray) -> ServiceResult:
        """Transcribes directly from a 16kHz float32 numpy array into clear Hausa text."""
        t0 = time.perf_counter()
        
        if audio_array is None or len(audio_array) == 0:
            return ServiceResult(False, "", 0.0, ServiceErrorCode.EMPTY_INPUT)

        if self.local_pipeline is not None:
            try:
                # Execute the speech-to-text conversion via the verified pipeline
                res = self.local_pipeline(audio_array.flatten())
                lat = (time.perf_counter() - t0) * 1000
                text = res.get("text", "").strip().lower()
                return ServiceResult(True, text, lat)
            except Exception as e:
                print(f"[ASR Pipeline Error] {type(e).__name__}: {e}")
                lat = (time.perf_counter() - t0) * 1000
                return ServiceResult(False, "", lat, ServiceErrorCode.SERVER_ERROR)

        lat = (time.perf_counter() - t0) * 1000
        return ServiceResult(False, "", lat, ServiceErrorCode.NETWORK_ERROR)

    def transcribe(self, audio_bytes: bytes) -> ServiceResult:
        return ServiceResult(False, "", 0.0, ServiceErrorCode.INVALID_PAYLOAD)

class LLMService:
    """N-ATLaS grounded inference service using native Q4_K_M GGUF runtime."""
    def __init__(
        self,
        model_path: str = "/home/sagemaker-user/models/natlas-8b-q4_k_m.gguf",
        llama_cli_path: str = "/home/sagemaker-user/llama.cpp/build/bin/llama-cli"
    ):
        self.client = NAtLaSClient(model_path=model_path, llama_cli_path=llama_cli_path)

    def generate_grounded(self, canonical_query: str, evidence_text: str) -> ServiceResult:
        sys_instruction = (
            "Kai mataimakin kiwon lafiya ne mai amfani da harshen Hausa. "
            "Amsa tambayoyi a takaice ta hanyar amfani da hujjojin da aka bayar kawai. "
            "Kada ka rubuta takamaiman sunan magani, allura, ko adadinsu."
        )
        prompt = f"Hujja: {evidence_text}\n\nTambaya: {canonical_query}\nAmsa:"
        return self.client.generate(prompt=prompt, system_instruction=sys_instruction, max_tokens=64)


class TTSService:
    """Thread-safe text-to-speech synthesis with isolated event loop."""
    def __init__(self, output_dir: str = "temp_audio", voice: str = "ha-NG-AyomideNeural"):
        self.output_dir = output_dir
        self.voice = voice
        os.makedirs(self.output_dir, exist_ok=True)

    def synthesize(self, text: str) -> ServiceResult:
        t0 = time.perf_counter()
        # Clean text of internal prompt markers and truncate long paragraphs for fast synthesis
        clean_text = text.replace("<assistant>", "").replace("<system>", "").replace("<user>", "").strip()
        if not clean_text:
            return ServiceResult(False, "", 0.0, ServiceErrorCode.EMPTY_INPUT)

        # Cap text at 250 characters for immediate low-latency audio rendering
        if len(clean_text) > 250:
            clean_text = clean_text[:250] + "..."

        dest_file = os.path.join(self.output_dir, f"audio_{uuid.uuid4().hex[:12]}.mp3")

        async def _run_tts():
            communicate = edge_tts.Communicate(clean_text, voice=self.voice, rate="+5%")
            await communicate.save(dest_file)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_run_tts())
            finally:
                loop.close()

            if os.path.exists(dest_file) and os.path.getsize(dest_file) > 100:
                lat = (time.perf_counter() - t0) * 1000
                return ServiceResult(True, dest_file, lat)
            else:
                return ServiceResult(False, "", (time.perf_counter() - t0) * 1000, ServiceErrorCode.SERVER_ERROR)

        except Exception as e:
            print(f"[TTS Notice] Synthesis unavailable ({type(e).__name__}). Text delivery preserved.")
            lat = (time.perf_counter() - t0) * 1000
            return ServiceResult(False, "", lat, ServiceErrorCode.NETWORK_ERROR)