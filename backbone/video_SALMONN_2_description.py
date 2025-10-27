"""Text-only variant of Video SALMONN 2.

The description pipeline mirrors ``qwen25_omni_description.py`` by driving the
local Video SALMONN 2 weights with pure text prompts (no video attachment).
"""

from __future__ import annotations

from transformers import TextStreamer

from .video_SALMONN_2 import VideoSALMONN2


class VideoSALMONN2_Description(VideoSALMONN2):
    """Run Video SALMONN 2 on description-only tasks."""

    def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a completion given textual instructions only."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        final_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(text=final_prompt, return_tensors="pt").to(self.device)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        try:
            response = self.model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
            )
            generated = response[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)
        except Exception as exc:  # pragma: no cover - defensive guard
            return f"Error: {exc}"


__all__ = ["VideoSALMONN2_Description"]
