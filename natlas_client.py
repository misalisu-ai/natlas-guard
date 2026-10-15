"""
natlas-guard: Unified N-ATLaS Inference Client
Direct subprocess and C-binding adapter for quantized N-ATLaS GGUF on Linux/CPU.
"""
import os
import time
import subprocess
from typing import Optional
from contracts import ServiceResult, ServiceErrorCode

class NAtLaSClient:
    def __init__(
        self,
        model_path: str = "/home/sagemaker-user/models/natlas-8b-q4_k_m.gguf",
        llama_cli_path: str = "/home/sagemaker-user/llama.cpp/build/bin/llama-cli"
    ):
        self.model_path = model_path
        self.llama_cli_path = llama_cli_path
        self._verify_binaries()

    def _verify_binaries(self):
        if not os.path.exists(self.model_path):
            print(f"[N-ATLaS Warning] GGUF model not found at {self.model_path}")
        if not os.path.exists(self.llama_cli_path):
            print(f"[N-ATLaS Warning] llama-cli binary not found at {self.llama_cli_path}")

    def generate(self, prompt: str, system_instruction: Optional[str] = None, max_tokens: int = 64) -> ServiceResult:
        t0 = time.perf_counter()
        sys_msg = system_instruction or "Kai mataimakin kiwon lafiya ne mai amfani da harshen Hausa."
        full_prompt = f"<system>\n{sys_msg}\n<user>\n{prompt}\n<assistant>\n"

        cmd = [
            self.llama_cli_path,
            "-m", self.model_path,
            "-p", full_prompt,
            "-n", str(max_tokens),
            "-t", "8",
            "-c", "512",
            "--temp", "0.1",
            "--no-display-prompt"
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=25.0)
            lat = (time.perf_counter() - t0) * 1000

            if res.returncode == 0:
                raw_out = res.stdout.strip()
                # Clean prompt repetitions or metadata banners
                clean_text = raw_out.split("<assistant>")[-1].strip()
                if not clean_text:
                    clean_text = raw_out
                return ServiceResult(True, clean_text, lat)
            else:
                print(f"[N-ATLaS Subprocess Error] {res.stderr[:200]}")
                return ServiceResult(False, "", lat, ServiceErrorCode.SERVER_ERROR)

        except subprocess.TimeoutExpired:
            return ServiceResult(False, "", (time.perf_counter() - t0) * 1000, ServiceErrorCode.TIMEOUT)
        except Exception as e:
            print(f"[N-ATLaS Execution Failure] {e}")
            return ServiceResult(False, "", (time.perf_counter() - t0) * 1000, ServiceErrorCode.SERVER_ERROR)