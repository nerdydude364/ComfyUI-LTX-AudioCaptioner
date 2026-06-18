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
                # Audio quality settings (passed to ffmpeg for extraction)
                "audio_channels": (["1", "2"], {"default": "1"}),
                "audio_sample_rate": (["8000", "16000", "22050", "44100", "48000"], {"default": "16000"}),
                # Ambient audio RMS thresholds
                "silence_rms_threshold": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.01, "step": 0.0001}),
                "low_rms_threshold": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
                # Vocal detection threshold in windows
                "vocal_window_rms_threshold": ("FLOAT", {"default": 0.015, "min": 0.001, "max": 0.5, "step": 0.001}),
                # Audio window size in milliseconds for RMS analysis
                "audio_window_ms": ("INT", {"default": 30, "min": 5, "max": 200, "step": 1}),
                # Music detection thresholds (RMS must exceed this, and spectral flatness must be below tonal_threshold)
                "music_rms_threshold": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 0.5, "step": 0.001}),
                "music_tonal_threshold": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 1.0, "step": 0.01}),
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

    def analyze_singing_vs_speaking(self, video_path, segments, detect_singing, segment_threshold, variance_threshold, vocal_rms_threshold, window_ms, audio_channels, audio_sample_rate):
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
            window_size = int(int(audio_sample_rate) * (window_ms / 1000.0))

            # Extract raw audio to analyze sustained acoustic velocity/gaps
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-ac', audio_channels, '-ar', audio_sample_rate, '-f', 's16le', 'pipe:1'
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

    def get_audio_ambient_type(self, video_path, silence_threshold, low_threshold, audio_channels,
                               audio_sample_rate):
        try:
            # Extract raw audio for ambient analysis
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-ac', audio_channels, '-ar', audio_sample_rate, '-f', 's16le', 'pipe:1'
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

    def _detect_music(self, audio_np, sample_rate, music_rms_threshold, music_tonal_threshold):
        """
        Detect whether audio contains music. Returns True/False.

        Strategy: spectral flatness is the primary discriminator.
        Music → tonal peaks (low flatness). Noise → flat spectrum (high flatness).
        We then confirm with two lightweight secondary checks so isolated tonal
        artifacts (e.g. a single sustained note) don't false-positive.
        """
        try:
            rms = np.sqrt(np.mean(audio_np ** 2))
            print(f"[LTX-AudioCaptioner] DETECT MUSIC: rms={rms} | music_rms_threshold={music_rms_threshold} | music_tonal_threshold={music_tonal_threshold}")
            if rms < music_rms_threshold:
                print(f"[LTX-AudioCaptioner] DETECT MUSIC: False (rms < music_rms_threshold)")
                return False

            fft_vals = np.fft.rfft(audio_np)
            power = np.maximum(np.abs(fft_vals) ** 2, 1e-12)
            flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
            freqs = np.fft.rfftfreq(len(audio_np), d=1.0 / sample_rate)
            centroid = float(np.sum(freqs * power) / np.sum(power))
            print(f"[LTX-AudioCaptioner] DETECT MUSIC: fft_vals={fft_vals} | power={power} | flatness={flatness} | freqs={freqs} | centroid={centroid}")

            # --- Primary: spectral flatness (harmonic richness) ---
            if flatness > music_tonal_threshold:
                print(f"[LTX-AudioCaptioner] DETECT MUSIC: False (flatness > music_tonal_threshold)")
                return False

            # --- Secondary check 1: harmonic density ---
            # Music has multiple tonal peaks (partials/harmonics), not just one spike.
            peaks = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])  # local maxima in right-half FFT
            peak_count = int(np.sum(peaks))
            print(f"[LTX-AudioCaptioner] DETECT MUSIC: peaks={peaks} | peak_count={peak_count}")
            # A few isolated peaks can happen in speech (vowels), but music typically has more
            # We lower the bar — even 10+ peaks is reasonable for music with partials
            if peak_count < 10:
                print(f"[LTX-AudioCaptioner] DETECT MUSIC: False (peak_count < 10)")
                return False

            # --- Secondary check 2: harmonic consistency ---
            # Check if peaks occur roughly at integer multiples of a fundamental
            # (i.e. they form a harmonic series).
            if peak_count >= 4:
                peak_freqs = freqs[1:-1][peaks]
                print(f"[LTX-AudioCaptioner] DETECT MUSIC: Continue (peak_count >= 4) | peak_freqs={peak_freqs}")
                # Look for a fundamental that explains many of these peaks
                if len(peak_freqs) >= 4:
                    # Use the lowest peak as a candidate fundamental
                    f0_cand = peak_freqs[0]
                    # Check how many peaks are near integer multiples
                    harmonics_match = sum(
                        1 for k in range(1, 10)
                        if np.any(np.abs(peak_freqs - k * f0_cand) < f0_cand * 0.15)
                    )
                    print(f"[LTX-AudioCaptioner] DETECT MUSIC: Continue (len(peak_freqs) >= 4) | f0_cand={f0_cand} | harmonics_match={harmonics_match}")
                    if harmonics_match >= 3:
                        print(f"[LTX-AudioCaptioner] DETECT MUSIC: True (harmonics_match >= 3) | --->> !MUSIC DETECTED! <<---")
                        return True
                # If peak detection alone isn't enough, accept flatness + centroid signals
                # Music typically has broader energy spread than a pure tone
                mid_mask = (freqs >= 250) & (freqs < 2000)
                mid_energy = float(np.sum(power[mid_mask])) if np.any(mid_mask) else 0
                total = float(np.sum(power)) + 1e-12
                print(f"[LTX-AudioCaptioner] DETECT MUSIC: Continue (len(peak_freqs) >= 4) | mid_mask={mid_mask} | mid_energy={mid_energy} | total={total}")
                if mid_energy / total > 0.01:
                    print(f"[LTX-AudioCaptioner] DETECT MUSIC: True (mid_energy / total > 0.01) | --->> !MUSIC DETECTED! <<---")
                    return True

            print(f"[LTX-AudioCaptioner] DETECT MUSIC: False (else) (peak_count < 4)")
            return False

        except Exception as ex:
            print(f"[LTX-AudioCaptioner] DETECT MUSIC: False (Exception raised: {ex})")
            return False

    def _get_music_description(self, audio_np, sample_rate, music_rms_threshold, music_tonal_threshold):
        """Run music detection and return a description string, or None if no music."""
        if self._detect_music(audio_np, sample_rate, music_rms_threshold, music_tonal_threshold):
            return "with music playing."
        return None

    def _get_music_description_from_video(self, video_path, audio_channels, audio_sample_rate,
                                           music_rms_threshold, music_tonal_threshold):
        """Extract audio from a video file and run music detection. Returns a description string or None."""
        try:
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-ac', audio_channels, '-ar', audio_sample_rate, '-f', 's16le', 'pipe:1'
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            audio_data, _ = process.communicate()
            if not audio_data:
                return None
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            return self._get_music_description(audio_np, int(audio_sample_rate), music_rms_threshold, music_tonal_threshold)
        except Exception:
            return None

    def process_audio_caption(self, video_path, trigger_name, whisper_model_type, overwrite_existing, detect_singing,
                              avg_segment_length_threshold, energy_variance_threshold, audio_channels, audio_sample_rate,
                              silence_rms_threshold, low_rms_threshold, vocal_window_rms_threshold, audio_window_ms,
                              music_rms_threshold, music_tonal_threshold,
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
                vocal_window_rms_threshold, audio_window_ms,
                audio_channels, audio_sample_rate
            )

            audio_description = f"{trigger_name} {vocal_action}, '{speech_text}'."
            for segment in segments:
                text_low = segment.get("text", "").lower()
                if "[" in text_low or "(" in text_low:
                    clean_fx = text_low.replace("[", "").replace("]", "").replace("(", "").replace(")", "").strip()
                    audio_description += f" A noticeable sound effect of {clean_fx} is heard."
            # --- Append music detection (independent of speech) ---
            music_desc = self._get_music_description_from_video(video_path, audio_channels, audio_sample_rate,
                                                                  music_rms_threshold, music_tonal_threshold)
            if music_desc:
                audio_description += f" The audio is {music_desc}"
        else:
            ambient_desc = self.get_audio_ambient_type(
                video_path, silence_rms_threshold, low_rms_threshold,
                audio_channels, audio_sample_rate
            )
            audio_description = ambient_desc
            # --- Append music detection (independent of ambient) ---
            music_desc = self._get_music_description_from_video(video_path, audio_channels, audio_sample_rate,
                                                                  music_rms_threshold, music_tonal_threshold)
            if music_desc:
                # Don't double up "with music playing." if ambient already covers it
                if "with music playing" not in audio_description:
                    audio_description += f" There is {music_desc}"

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
