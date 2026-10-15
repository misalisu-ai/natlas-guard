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
        self.has_local_gguf = os.path.exists(self.model_path) and os.path.exists(self.llama_cli_path)
        if not self.has_local_gguf:
            print("[N-ATLaS Cloud Notice] Local GGUF runtime not mounted in this container; using deterministic evidence-backed fallback mode.")

    def generate(self, prompt: str, system_instruction: Optional[str] = None, max_tokens: int = 64) -> ServiceResult:
        t0 = time.perf_counter()
        if not self.has_local_gguf:
            # Extract evidence card text passed in prompt for safe cloud demo display
            if "Hujja:" in prompt and "Tambaya:" in prompt:
                ev = prompt.split("Hujja:")[-1].split("Tambaya:")[0].strip()
                lat = (time.perf_counter() - t0) * 1000
                return ServiceResult(True, ev, lat)
            return ServiceResult(False, "", (time.perf_counter() - t0) * 1000, ServiceErrorCode.SERVER_ERROR)

        sys_msg = system_instruction or "Kai mataimakin kiwon lafiya ne mai amfani da harshen Hausa."
        full_prompt = f"<system>\n{sys_msg}\n<user>\n{prompt}\n<assistant>\n"
        cmd = [
            self.llama_cli_path, "-m", self.model_path,
            "-p", full_prompt, "-n", str(max_tokens),
            "-t", "8", "-c", "512", "--temp", "0.1"
        ]
        try:
            res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=25.0)
            lat = (time.perf_counter() - t0) * 1000
            if res.returncode == 0:
                raw_out = res.stdout.strip()
                clean_text = raw_out.split("<assistant>")[-1].split("[ Prompt:")[0].strip()
                return ServiceResult(True, clean_text or raw_out, lat)
            return ServiceResult(False, "", lat, ServiceErrorCode.SERVER_ERROR)
        except Exception:
            return ServiceResult(False, "", (time.perf_counter() - t0) * 1000, ServiceErrorCode.SERVER_ERROR)
