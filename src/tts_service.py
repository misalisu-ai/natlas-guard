import os
import uuid
import asyncio
import edge_tts
from torch import Optional

class HausaTTSService:
    def __init__(self, output_dir: str = "temp_audio"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Default high-fidelity neural voice for Nigerian/West African speech
        self.voice = "ha-NG-AyomideNeural"

    async def _synthesize_file(self, text: str, out_path: str):
        communicate = edge_tts.Communicate(text, voice=self.voice)
        await communicate.save(out_path)

    def generate_speech(self, text: str) -> Optional[str]:
        """
        Synthesizes Hausa text to an isolated MP3 file.
        Returns path to generated file or None if failed.
        """
        if not text:
            return None
            
        file_id = f"resp_{uuid.uuid4().hex[:10]}.mp3"
        out_path = os.path.join(self.output_dir, file_id)
        
        try:
            asyncio.run(self._synthesize_file(text, out_path))
            return out_path
        except Exception:
            # Non-blocking graceful failure for TTS
            return None