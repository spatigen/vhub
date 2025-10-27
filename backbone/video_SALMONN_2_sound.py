"""Sound-aware Video SALMONN 2 adapter: pass reconstructed video + 16k mono audio as media to the processor, not as inline base64 text."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import torch
# MoviePy import fix: use editor submodule
from moviepy import VideoFileClip
from transformers import AutoProcessor, AutoModelForCausalLM


try:  # Allow execution both as a script and as a package module.
    from backbone_utils import (
        extract_frames_base64,
        get_max_frame_and_interval,
        reconstruct_video,
        encode_video,  # kept for compatibility (not used in the main path)
    )
except ModuleNotFoundError:  # pragma: no cover - only hits when run as package
    from .backbone_utils import (  # type: ignore
        extract_frames_base64,
        get_max_frame_and_interval,
        reconstruct_video,
        encode_video,
    )


class VideoSALMONN2_Sound:
    """Video SALMONN 2 backbone that reconstructs audio-aware clips."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        device: str | torch.device | None = None,
    ):
        """Initializes Video SALMONN 2 with audio reconstruction support."""
        env_model_path = os.getenv("VIDEO_SALMONN2_PATH")
        default_local_path = "/data/henry/pretrain_model/video-SALMONN-2"

        if model_name_or_path:
            self.model_path = model_name_or_path
        elif env_model_path and os.path.exists(env_model_path):
            self.model_path = env_model_path
        elif os.path.exists(default_local_path):
            self.model_path = default_local_path
        else:
            self.model_path = "tsinghua-ee/video-SALMONN-2"

        local_only = os.path.isdir(self.model_path)

        env_device = os.getenv("VIDEO_SALMONN2_DEVICE")
        requested_device = device if device is not None else env_device
        self.device = self._resolve_device(requested_device)
        print(f"Using device: {self.device}")

        # Processor/model
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=local_only
        )
        # Use bf16 on CUDA when available; otherwise fp32
        model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=local_only,
        ).to(self.device)

        # Token limits (kept as in your code)
        self.max_tokens = 16384
        self.patch_h = 14
        self.patch_w = 14
        self.model_max_tokens_per_image = 1338
        self.model_min_tokens_per_image = 4

        # Aliases for utilities expecting the Qwen naming.
        self.model_h = self.patch_h
        self.model_w = self.patch_w
        self.model_max_tokens = self.model_max_tokens_per_image
        self.model_min_tokens = self.model_min_tokens_per_image

    def get_completion(self, system_prompt: str, user_prompt: str, video_path: str) -> str:
        """
        Build a reconstructed clip with subsampled frames and (if present) 16k mono audio,
        then pass media directly to the processor for generation.
        """
        # --- Probe the video and compute sampling schedule ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise IOError("Could not read first frame of video.")
        nframes, interval = get_max_frame_and_interval(
            self.max_tokens,
            cap,
            frame,
            self.model_h,
            self.model_w,
            self.model_max_tokens,
            self.model_min_tokens,
        )
        if not original_fps or original_fps <= 0:
            original_fps = 30.0
        time_between_frames_sec = interval / original_fps
        target_fps = 1.0 / time_between_frames_sec if time_between_frames_sec > 0 else 1.0
        cap.release()

        # --- Extract frames (base64 list used only by your reconstructor) ---
        base64_frames = extract_frames_base64(video_path, nframes=nframes, interval=interval)

        # --- Ensure audio is 16 kHz mono (SALMONN expects mono 16k wav) ---
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_wav = tmpdir_path / "audio_16k_mono.wav"
            audio_in = self._extract_audio(video_path, audio_wav)  # returns None if no audio

            # Fallback extraction via MoviePy ONLY if ffmpeg path is missing or failed gracefully.
            # (MoviePy may not enforce 16k mono; prefer ffmpeg path above.)
            if audio_in is None:
                try:
                    clip = VideoFileClip(video_path)
                    if clip.audio is not None:
                        clip.audio.write_audiofile(audio_wav.as_posix(), fps=16000, nbytes=2, codec="pcm_s16le")
                        audio_in = audio_wav if audio_wav.exists() else None
                except Exception:
                    audio_in = None

            # --- Reconstruct a short, synchronized clip at target_fps and (optional) audio ---
            recon_mp4 = tmpdir_path / "reconstruct_video.mp4"
            reconstruct_video(base64_frames, audio_in.as_posix() if audio_in else None, target_fps, recon_mp4.as_posix())

            # --- Compose chat messages ---
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            # --- Build processor inputs (primary path: pass media as args) ---
            # Different processors use slightly different argument names; try robust variants.
            inputs = None
            exceptions: list[str] = []

            def _to_device(tensors: dict) -> dict:
                moved = {}
                for k, v in tensors.items():
                    moved[k] = v.to(self.device) if hasattr(v, "to") else v
                return moved

            # Try common multimodal signatures in order of likelihood.
            for media_kwargs in (
                {"videos": [recon_mp4.as_posix()], "audios": [audio_in.as_posix()] if audio_in else None},
                {"video": recon_mp4.as_posix(), "audio": audio_in.as_posix() if audio_in else None},
                # Some implementations take file paths inside messages; this is the least preferred path.
            ):
                # Strip None entries
                media_kwargs = {k: v for k, v in media_kwargs.items() if v is not None}
                try:
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        **media_kwargs,
                    )
                    inputs = _to_device(inputs)
                    break
                except Exception as e:
                    exceptions.append(f"{type(e).__name__}: {e}")

            # Absolute last resort: inline tags (still NOT base64 stuffing).
            if inputs is None:
                try:
                    tag_user_prompt = (
                        f"<video>\n{user_prompt}\n</video>" if audio_in is None
                        else f"<video><audio>\n{user_prompt}\n</audio></video>"
                    )
                    fallback_msgs = []
                    if system_prompt:
                        fallback_msgs.append({"role": "system", "content": system_prompt})
                    fallback_msgs.append({"role": "user", "content": tag_user_prompt})
                    inputs = self.processor.apply_chat_template(
                        fallback_msgs,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        videos=[recon_mp4.as_posix()],
                        **({"audios": [audio_in.as_posix()]} if audio_in else {}),
                    )
                    inputs = _to_device(inputs)
                except Exception as e:
                    joined = "\n".join(exceptions + [f"{type(e).__name__}: {e}"])
                    raise RuntimeError(
                        "Could not build processor inputs for video+audio. Tried multiple signatures.\n" + joined
                    )

            # --- Generate ---
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
            input_length = inputs["input_ids"].shape[1]
            decoded = self.processor.batch_decode(generated[:, input_length:], skip_special_tokens=True)
            return decoded[0].strip()

    def _resolve_device(self, requested_device: str | torch.device | None) -> torch.device:
        """Resolve a torch.device value with graceful fallbacks."""
        if requested_device is not None:
            try:
                device = torch.device(requested_device)
            except (TypeError, ValueError, RuntimeError) as exc:
                print(f"Invalid device '{requested_device}'; falling back to CPU. Reason: {exc}")
                return torch.device("cpu")

            if device.type == "cuda":
                if not torch.cuda.is_available():
                    print("CUDA requested but not available; falling back to CPU.")
                    return torch.device("cpu")
                device_index = device.index or 0
                if device_index >= torch.cuda.device_count():
                    print(
                        f"CUDA device index {device_index} unavailable (only {torch.cuda.device_count()} visible); falling back to CPU."
                    )
                    return torch.device("cpu")
                return torch.device(f"cuda:{device_index}")
            return device

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _extract_audio(self, video_path: str, destination: Path) -> Path | None:
        """Extract mono 16 kHz audio via ffmpeg when the source contains sound."""
        if not self._video_has_audio(video_path):
            return None

        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-vn",
            "-ac", "1", "-ar", "16000", "-f", "wav", destination.as_posix(),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError as exc:  # pragma: no cover - CLI environment specific
            # If ffmpeg not on PATH, signal None so MoviePy fallback may run.
            return None
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg failed to extract audio from '{video_path}'.") from exc

        if not destination.exists():
            raise RuntimeError("ffmpeg reported success, yet the expected audio file was not created.")

        return destination

    def _video_has_audio(self, video_path: str) -> bool:
        """Determine whether the source video carries an audio stream."""
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "json", video_path,
        ]
        try:
            result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:  # pragma: no cover
            # If ffprobe is missing, be permissive: we’ll try MoviePy extraction later.
            return True
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffprobe failed while inspecting '{video_path}'.") from exc

        try:
            data = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            data = {}
        streams = data.get("streams", [])
        return len(streams) > 0

