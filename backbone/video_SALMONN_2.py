import os
import sys
import cv2
import torch
from typing import Optional, Union
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, TextStreamer

from backbone_utils import get_max_frame_and_interval

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def extract_frames_pil(video_path, nframes, interval):
    """
    Extracts frames from a video and returns them as a list of PIL Image objects.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps * interval))
    pil_frames = []
    count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_frames.append(pil_image)
        count += 1
        if len(pil_frames) >= nframes:
            break

    cap.release()
    return pil_frames

class VideoSALMONN2:
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initializes the open-source Video SALMONN 2 model, processor, and tokenizer.
        """
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=local_only,
        ).to(self.device)

        self.max_tokens = 16384

        self.patch_h = 14
        self.patch_w = 14
        self.model_max_tokens_per_image = 1338
        self.model_min_tokens_per_image = 4

    def get_completion(self, system_prompt, user_prompt, video_path):
        """
        Generates a text completion from the local Video SALMONN 2 model.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise IOError("Could not read first frame of video.")

        nframes, interval = get_max_frame_and_interval(
            self.max_tokens, cap, frame, self.patch_h, self.patch_w, self.model_max_tokens_per_image, self.model_min_tokens_per_image
        )
        cap.release()

        pil_frames = extract_frames_pil(video_path, nframes=nframes, interval=interval)

        image_token = "<image>"
        prompt_text = f"{image_token * len(pil_frames)}\n{user_prompt}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]

        final_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # The processor is just a tokenizer, so we only tokenize the text
        inputs = self.processor(text=final_prompt, return_tensors="pt").to(self.device)

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        try:
            response = self.model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=1500,
                do_sample=True,
                temperature=0.7,
            )
            # grab generated new tokens
            response = self.tokenizer.decode(response[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return response
        except Exception as e:
            return f"Error: {str(e)}"

    def _resolve_device(self, requested_device: Optional[Union[str, torch.device]]) -> torch.device:
        """Resolves the torch device, honoring explicit requests when possible."""

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

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")
