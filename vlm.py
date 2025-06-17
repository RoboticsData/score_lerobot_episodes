import pathlib, base64, cv2, numpy as np, google.generativeai as genai
from pydantic import BaseModel, Field
import json

class ScoreOutput(BaseModel):
    score: float

class VLMInterface:
    # pick whichever model tier your quota allows
    #_MODEL = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    _MODEL = genai.GenerativeModel("gemini-2.0-flash")

    @staticmethod
    def _load_mp4_bytes(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def task_success(self, video_path: str, prompt: str) -> float:
        """
        Ask Gemini to grade whether the *desired* behaviour occurred.
        The model responds with a float 0-1 in JSON.
        """
        video_bytes = self._load_mp4_bytes(video_path)
        system_instruction = (
            "You are an automated evaluator. "
            "Return ONLY valid JSON: {\"score\": <0-1 float>} where 1.0 = full success."
        )
        user_instruction = (
            f"Here is the task description: {prompt}\n"
            "Watch the video and judge whether the task was accomplished."
        )

        response = self._MODEL.generate_content(
            [
                {"mime_type": "video/mp4", "data": video_bytes},
                system_instruction,
                user_instruction,
            ],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": ScoreOutput,
                "temperature": 0.0,
            },
        )
        j = json.loads(response.text)#candidates[0].content.text)
        return j["score"]
    
    def negative_visual_quality(self, frame: np.ndarray) -> float:
        """
        Returns a penalty (0-1) where 0 = pristine and 1 = unusable.
        """
        # encode OpenCV BGR frame → JPEG bytes
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Could not encode frame")

        prompt = (
            "Rate the VISUAL QUALITY of this frame on a continuous scale "
            "where 0 = excellent, 1 = terrible. "
            "Only respond with JSON: {\"score\": <float>}."
        )

        response = self._MODEL.generate_content(
            [
                {"mime_type": "image/jpeg", "data": jpg.tobytes()},
                prompt,
            ],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": ScoreOutput,
                "temperature": 0.0
            },
        )
        j = json.loads(response.text)
        return float(j["score"])


if __name__ == "__main__":
    """
    Quick check that Gemini can be called end-to-end.

    Usage:
        python vlm_interface.py               # → uses auto-generated black video
        python vlm_interface.py path/to.mp4   # → uses your video file

    Requires `GOOGLE_API_KEY` (or equivalent) to be set in the environment.
    """

    import sys, tempfile, cv2, numpy as np, pathlib

    vlm = VLMInterface()
    video_path = 'input_video.mp4'

    prompt = (
        "Open the book"
    )
    ts_score = vlm.task_success(str(video_path), prompt)
    assert 0.0 <= ts_score <= 1.0, f"task_success score out of range: {ts_score}"
    print(f"task_success → {ts_score:.3f}")

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read first frame for quality check")

    nq_score = vlm.negative_visual_quality(frame)
    assert 0.0 <= nq_score <= 1.0, f"negative_visual_quality out of range: {nq_score}"
    print(f"negative_visual_quality → {nq_score:.3f}")
