import os
import numpy as np
import soundfile as sf
from scipy import signal
from typing import Tuple, Optional

def read_and_resample_16k(audio_path: Optional[str]) -> Tuple[bool, float, Optional[np.ndarray], str]:
    """
    Reads audio via soundfile, converts to mono, and resamples to 16kHz
    using scipy without requiring FFmpeg or torchaudio.
    Returns: (is_valid, duration_sec, float32_16k_mono_array, error_msg)
    """
    if not audio_path or not os.path.exists(audio_path):
        return False, 0.0, None, "FAYIL_BABU: Ba a sami muryar ba."

    try:
        # Read raw data directly
        data, sample_rate = sf.read(audio_path, dtype="float32")
        
        # Convert stereo/multichannel to mono
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        duration = len(data) / float(sample_rate)

        if duration < 0.5:
            return False, duration, None, "MURYA_TA_YI_KANKANTA: Muryar ba ta kai rabin dakika ba."
        if duration > 45.0:
            return False, duration, None, "MURYA_TA_YI_TSAWO: Don Allah kada sautin ya wuce dakika 45."

        # Resample to 16000 Hz if needed
        target_sr = 16000
        if sample_rate != target_sr:
            num_target_samples = int(len(data) * target_sr / float(sample_rate))
            data = signal.resample(data, num_target_samples).astype(np.float32)

        return True, duration, data, "CLEAN"

    except Exception as e:
        return False, 0.0, None, f"AUDIO_DECODE_ERROR: {type(e).__name__} - {e}"

def process_gradio_input_16k(audio_input) -> Tuple[bool, float, Optional[np.ndarray], str]:
    """
    Production-grade adapter that robustly processes both in-memory 
    NumPy data structures and path strings without file-locking crashes.
    """
    if audio_input is None:
        return False, 0.0, None, "FAYIL_BABU: Ba a sami muryar ba."

    try:
        # 1. Handle file paths safely using a direct binary stream read bypass
        if isinstance(audio_input, str):
            import scipy.io.wavfile as wav
            try:
                sample_rate, data = wav.read(audio_input)
            except Exception:
                import soundfile as sf
                data, sample_rate = sf.read(audio_input, dtype="float32")
        else:
            # Handle standard Gradio tuple layout unpack natively
            sample_rate, data = audio_input

        # 2. Prevent empty array or structural corruption leaks early
        if data is None or len(data) == 0:
            return False, 0.0, None, "Fayil din sautin muryar babu komai."

        # 3. Cast to Float32 amplitude ranges safely based on input type
        if np.issubdtype(data.dtype, np.integer):
            max_val = float(np.iinfo(data.dtype).max or 32768.0)
            data = data.astype(np.float32) / max_val
        else:
            data = data.astype(np.float32)

        # 4. Flatten multi-channel stereo down to single-channel mono natively
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        duration = len(data) / float(sample_rate)
        if duration < 0.4:
            return False, duration, None, "Sautin ya yi kankanta."

        # 5. High-fidelity resampling using standard linear 1D stride slicing 
        # to ensure zero file-locking overhead or array dimension collisions
        if sample_rate != 16000:
            import scipy.signal as signal
            num_samples = int(len(data) * 16000 / sample_rate)
            data = signal.resample(data, num_samples).astype(np.float32)

        return True, duration, data, "CLEAN"

    except Exception as e:
        print(f"[ASR Audio Processing Crash] {type(e).__name__}: {e}")
        return False, 0.0, None, f"AUDIO_ERROR: {str(e)}"
