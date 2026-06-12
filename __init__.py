import os
import subprocess
import numpy as np
import whisper

WHISPER_MODEL_CACHE = {}


class LTX23AudioCaptioner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"forceInput": True, "default": ""}),
                "trigger_name": ("STRING", {"default": "character"}),
                "whisper_model_type": (["base", "tiny", "small", "medium", "character"], {"default": "character", "forceInput": False}),
                "overwrite_existing": ("BOOLEAN", {"default": True}),
                "detect_singing": ("BOOLEAN", {"default": False}),
                # Thresholds for singing detection heuristic
                "avg_segment_length_threshold": ("FLOAT", {"default": 1.8, "min": 0.1, "max": 10.0, "step": 0.1}),
                "energy_variance_threshold": ("FLOAT", {"default": 0.005, "min": 0.0001, "max": 0.1, "step": 0.0001}),
                # Audio quality settings (passed to ffmpeg for ambient extraction)
                "audio_channels": (["1", "2"], {"default": "1"}),
                "audio_sample_rate": (["8000", "16000", "22050", "44100", "48000"], {"default": "16000"}),
                # Ambient audio RMS thresholds
                "silence_rms_threshold": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.01, "step": 0.0001}),
                "low_rms_threshold": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
                # Vocal detection threshold in windows
                "vocal_window_rms_threshold": ("FLOAT", {"default": 0.015, "min": 0.001, "max": 0.5, "step": 0.001}),
                # Audio window size in milliseconds for RMS analysis
                "audio_window_ms": ("INT", {"default": 30, "min": 5, "max": 200, "step": 1}),
            },
            "optional": {
                "visual_caption": ("STRING", {"forceInput": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_caption",)
    FUNCTION = "process_audio_caption"
    OUTPUT_NODE = True
    CATEGORY = "LTX-2.3 Dataset Tools"

    def analyze_singing_vs_speaking(self, video_path, segments, detect_singing, segment_threshold, variance_threshold, vocal_rms_threshold, window_ms):
        """
        Analyzes whether the vocal performance leans towards singing vs speaking.
        Sustained vowels and continuous high-energy tracking denote singing.
        Only runs audio analysis when detect_singing is True.
        """
        if not detect_singing:
            return "says"

        if not segments:
            return "says"

        try:
            # Calculate average duration of transcribed spoken phrases
            durations = [seg.get("end", 0) - seg.get("start", 0) for seg in segments]
            avg_segment_length = np.mean(durations) if durations else 0

            # Frame the audio into windows to track continuous sound blocks
            window_size = int(16000 * (window_ms / 1000.0))

            # Extract raw audio to analyze sustained acoustic velocity/gaps
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1'
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            audio_data, _ = process.communicate()

            if not audio_data:
                return "says"

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Frame the audio into windows to track continuous sound blocks
            windows = [audio_np[i:i + window_size] for i in range(0, len(audio_np), window_size) if len(audio_np[i:i + window_size]) == window_size]
            rms_per_window = [np.sqrt(np.mean(w ** 2)) for w in windows]

            # Count the ratio of continuous sound vs short gaps typical of speech
            vocal_windows = [r for r in rms_per_window if r > vocal_rms_threshold]
            if not vocal_windows:
                return "says"

            # Singing has much lower variance in window volume because tones are held steady
            energy_variance = np.var(vocal_windows)

            # Threshold Rule: Held notes + long vocal segments = Singing
            print(f"[LTX-AudioCaptioner] Detect speaking or singing: avg_segment_length={avg_segment_length} | energy_variance={energy_variance}")
            if avg_segment_length > segment_threshold and energy_variance < variance_threshold:
                return "sings"
            else:
                return "says"
        except Exception:
            return "says"

    def get_audio_ambient_type(self, video_path, silence_threshold, low_threshold):
        try:
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1'
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            audio_data, _ = process.communicate()

            if not audio_data:
                return "accompanied by absolute silence."

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_np ** 2)) if len(audio_np) > 0 else 0

            if rms < silence_threshold:
                return "accompanied by absolute dead silence and no audio."
            elif rms < low_threshold:
                return "accompanied by a subtle low background static hiss and room tone."
            else:
                return "accompanied by general ambient background noise and a faint hum."
        except Exception:
            return "with low ambient background sound."

    def process_audio_caption(self, video_path, trigger_name, whisper_model_type, overwrite_existing, detect_singing,
                              avg_segment_length_threshold, energy_variance_threshold, audio_channels, audio_sample_rate,
                              silence_rms_threshold, low_rms_threshold, vocal_window_rms_threshold, audio_window_ms,
                              visual_caption=""):
        video_path = video_path.strip().strip('"').strip("'").strip('[').strip(']')

        if not video_path or not os.path.exists(video_path):
            print(f"[LTX-AudioCaptioner] ERROR: Target path not found -> {video_path}")
            return (f"Error: Video path '{video_path}' not found.",)

        global WHISPER_MODEL_CACHE
        # Map legacy "character" label to the actual Whisper model name
        whisper_name = whisper_model_type if whisper_model_type != "character" else "base"

        if whisper_name not in WHISPER_MODEL_CACHE:
            print(f"[LTX-AudioCaptioner] Loading local Whisper '{whisper_name}' model...")
            WHISPER_MODEL_CACHE[whisper_name] = whisper.load_model(whisper_name)

        model = WHISPER_MODEL_CACHE[whisper_name]
        txt_path = os.path.splitext(video_path)[0] + ".txt"

        print(f"[LTX-AudioCaptioner] Parsing audio from file: {video_path}")
        result = model.transcribe(video_path, word_timestamps=True)
        speech_text = result["text"].strip()
        segments = result.get("segments", [])

        audio_description = ""

        if speech_text:
            # Dynamically determine the vocal action string
            vocal_action = self.analyze_singing_vs_speaking(
                video_path, segments, detect_singing,
                avg_segment_length_threshold, energy_variance_threshold,
                vocal_window_rms_threshold, audio_window_ms
            )

            audio_description = f"{trigger_name} {vocal_action}, '{speech_text}'."
            for segment in segments:
                text_low = segment.get("text", "").lower()
                if "[" in text_low or "(" in text_low:
                    clean_fx = text_low.replace("[", "").replace("]", "").replace("(", "").replace(")", "").strip()
                    audio_description += f" A noticeable sound effect of {clean_fx} is heard."
        else:
            audio_description = self.get_audio_ambient_type(video_path, silence_rms_threshold, low_rms_threshold)

        base_caption = visual_caption.strip()
        if not base_caption and os.path.exists(txt_path) and not overwrite_existing:
            with open(txt_path, "r", encoding="utf-8") as f:
                base_caption = f.read().strip()

        if base_caption:
            if base_caption.endswith("."):
                base_caption = base_caption[:-1]
            final_caption = f"{base_caption}. {audio_description}"
        else:
            final_caption = f"A video of {trigger_name}. The audio is {audio_description}"

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(final_caption)
            print(f"[LTX-AudioCaptioner] SUCCESS: Saved caption file -> {txt_path}")
        except Exception as e:
            print(f"[LTX-AudioCaptioner] FILE WRITING ERROR: {str(e)}")

        return (final_caption,)


NODE_CLASS_MAPPINGS = {"LTX23AudioCaptioner": LTX23AudioCaptioner}
NODE_DISPLAY_NAME_MAPPINGS = {"LTX23AudioCaptioner": "LTX-2.3 Audio-Video Captioner (Local)"}
