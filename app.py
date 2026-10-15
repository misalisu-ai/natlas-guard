import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), "src"))
"""
HausaCare Voice - Interactive Gradio Interface
Connects microphone/audio input to the audited HausaCare-Safe pipeline.
"""
import os
import gradio as gr
from pipeline import HausaCareSafePipeline
from audio_utils import read_and_resample_16k, process_gradio_input_16k
from contracts import SafetyRoute, PipelineStatus

# Initialize audited pipeline
pipeline = HausaCareSafePipeline(kb_path="data/knowledge_base.json")

ROUTE_METADATA = {
    SafetyRoute.EMERGENCY_ESCALATION.value: ("🚨 GAGGAWA: Bukatar Asibiti (Emergency Escalation)", "#b91c1c"),
    SafetyRoute.PRESCRIPTION_INTERCEPT.value: ("🛡️ HANI: Ba a Ba da Magani Ba (Prescription Intercept)", "#c2410c"),
    SafetyRoute.SAFE_ABSTENTION.value: ("⚠️ GARGADI: Guji Maganin Gargajiya (Safe Abstention)", "#b45309"),
    SafetyRoute.UNCERTAIN_AMBIGUOUS.value: ("❓ NEMAN BAYANI: Karin Bayani (Ambiguity Elicitation)", "#4338ca"),
    SafetyRoute.ALLOW_RAG.value: ("✅ ILIMI: Bayanin Lafiya na Farko (Grounded Primary Health)", "#15803d")
}


from audio_utils import read_and_resample_16k

def process_voice_turn(audio_data, user_rating):
    if audio_data is None:
        return (
            "Don Allah yi magana ta amfani da makirufo ko saka fayil din murya.",
            "",
            None,
            "Awaiting Input"
        )

    # 1. Direct In-Memory Tuple Resampling Adapter (Bypasses local file system locks)
    is_valid, duration, audio_array_16k, err_msg = process_gradio_input_16k(audio_data)
    if not is_valid:
        return f"Kuskure: {err_msg}", "", None, "Audio Validation Failed"

    # 2. Pipeline Execution
    result = pipeline.process_audio(audio_array_16k=audio_array_16k, audio_duration=duration)

    # 3. Status Display
    if result.pipeline_status == PipelineStatus.ASR_FAILURE:
        badge_label = "⚠️ KUSKUREN SAUTI (ASR Failure)"
    elif result.pipeline_status == PipelineStatus.LLM_FAILURE:
        badge_label = "⚠️ KUSKUREN AI (LLM Service Error)"
    elif result.pipeline_status == PipelineStatus.NO_RETRIEVAL_MATCH:
        badge_label = "❓ BABU KATIN BAYANI (Low Confidence)"
    else:
        badge_label, _ = ROUTE_METADATA.get(result.safety_route.value, ("Sakamako", "#374151"))

    total_lat = result.latencies_ms.get("total", 0.0)
    card_id = result.retrieval_result.card.card_id if (result.retrieval_result and result.retrieval_result.card) else "NONE"
    
    status_summary = (
        f"**Tafarkin Tsaro:** {badge_label} | "
        f"**Lokaci:** {total_lat:.1f}ms | "
        f"**Katin Bayani:** {card_id} | "
        f"**Hali:** {result.pipeline_status.value}"
    )

    raw_transcript = result.metadata.get("raw_transcript", "")

    return raw_transcript, result.final_response, result.audio_path, status_summary


custom_css = """
.gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
#status-box { padding: 12px; border-radius: 8px; margin-top: 8px; font-weight: 500; }
"""

with gr.Blocks(css=custom_css, title="HausaCare Voice - AMGScale AI") as demo:
    gr.Markdown(
        """
        # 🎙️ HausaCare Voice
        ### *Safety-Aware Voice-First Primary Healthcare Assistant for Hausa-Speaking Communities*
        **AMGScale AI Research Prototype** | Powered by N-ATLAS & HausaCare-Safe Framework
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="Danna domin yin magana (Record Voice Turn)"
            )
            rating_in = gr.Radio(
                choices=["1 - Ba kyau", "2 - Matsakaici", "3 - Da kyau", "4 - Mai kyau sosai", "5 - Kyakkyawa"],
                label="Maki kan fahimta da ingancin amsa (Clarity Rating)",
                value="4 - Mai kyau sosai"
            )
            submit_btn = gr.Button("Aika Tambaya (Submit Query)", variant="primary")

        with gr.Column(scale=1):
            status_box = gr.Markdown("### **Tafarkin Tsaro:** Awaiting Input", elem_id="status-box")
            transcript_box = gr.Textbox(label="Abin da aka fahimta (Recognized Speech)", interactive=False)
            response_box = gr.Textbox(label="Amsar HausaCare (Verified Response)", interactive=False, lines=4)
            audio_out = gr.Audio(label="Saurari Amsar (Voice Playback)", autoplay=True)

    submit_btn.click(
        fn=process_voice_turn,
        inputs=[audio_in, rating_in],
        outputs=[transcript_box, response_box, audio_out, status_box]
    )

if __name__ == "__main__":
    demo.queue().launch()