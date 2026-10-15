import json
import time
import subprocess
import os

print("--- Running natlas-guard Integration Sanity Verification ---")
model_path = "/home/sagemaker-user/models/natlas-8b-q4_k_m.gguf"
cli_path = "/home/sagemaker-user/llama.cpp/build/bin/llama-cli"

test_query = "Wadanne hanyoyi ne mafi kyau na wanke hannu domin riga-kafin cututtuka?"
evidence = "Wanke hannu da sabulu da ruwa mai gudu na kashe kwayoyin cuta kuma yana riga-kafin gudawa da mura."
sys_msg = "Kai mataimakin kiwon lafiya ne. Yi amfani da hujjojin da aka bayar kawai."
full_prompt = f"<system>\n{sys_msg}\n<user>\nHujja: {evidence}\n\nTambaya: {test_query}\nAmsa:\n<assistant>\n"

cmd = [
    cli_path,
    "-m", model_path,
    "-p", full_prompt,
    "-n", "48",
    "-t", "8",
    "-c", "512",
    "--temp", "0.1"
]

t0 = time.perf_counter()
res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30.0)
lat_ms = (time.perf_counter() - t0) * 1000

# Extract assistant output
raw_text = res.stdout
assistant_text = ""
if "<assistant>" in raw_text:
    assistant_text = raw_text.split("<assistant>")[-1].split("[ Prompt:")[0].strip()
elif "\n\n" in raw_text:
    assistant_text = raw_text.strip()

trace = {
    "test_name": "genuine_natlas_invocation",
    "status": "SUCCESS" if res.returncode == 0 and len(assistant_text) > 0 else "FAILED",
    "backend": "llama.cpp (Q4_K_M GGUF)",
    "input_query": test_query,
    "model_output": assistant_text,
    "latency_ms": round(lat_ms, 2),
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
}

os.makedirs("docs/integration_evidence", exist_ok=True)
with open("docs/integration_evidence/sample_api_trace.json", "w") as f:
    json.dump(trace, f, indent=2)

print("\n--- Verified Result Written to sample_api_trace.json ---")
print(json.dumps(trace, indent=2))
