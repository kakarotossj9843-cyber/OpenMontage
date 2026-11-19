"""eSpeak NG local text-to-speech provider tool."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class EspeakTTS(BaseTool):
    name = "espeak_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "espeak"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = ["cmd:espeak-ng"]
    install_instructions = (
        "Install eSpeak NG (free, offline, robotic voice):\n"
        "  apt-get install -y espeak-ng"
    )
    agent_skills = ["text-to-speech"]

    capabilities = ["text_to_speech", "offline_generation"]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": True,
        "native_audio": True,
    }
    best_for = [
        "zero-cost offline narration when no neural voice provider is configured",
    ]
    not_good_for = [
        "natural or expressive narration",
        "voice clone matching",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "voice": {"type": "string", "default": "en-us"},
            "speed_wpm": {"type": "integer", "default": 150},
            "pitch": {"type": "integer", "default": 50},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=50, network_required=False
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=[])
    idempotency_key_fields = ["text", "voice", "speed_wpm", "pitch"]
    side_effects = ["writes audio file to output_path"]
    user_visible_verification = ["Listen to generated audio for intelligibility"]

    def get_status(self) -> ToolStatus:
        if shutil.which("espeak-ng"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="eSpeak NG not available. " + self.install_instructions)

        start = time.time()
        try:
            result = self._generate(inputs)
        except Exception as exc:
            return ToolResult(success=False, error=f"Local TTS generation failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        return result

    def _generate(self, inputs: dict[str, Any]) -> ToolResult:
        output_path = Path(inputs.get("output_path", "tts_output.wav"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        voice = inputs.get("voice", "en-us")

        proc = subprocess.run(
            [
                "espeak-ng",
                "-v", voice,
                "-s", str(inputs.get("speed_wpm", 150)),
                "-p", str(inputs.get("pitch", 50)),
                "-w", str(output_path),
                inputs["text"],
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
        )

        if proc.returncode != 0:
            return ToolResult(success=False, error=f"eSpeak NG failed (exit {proc.returncode}): {proc.stderr}")
        if not output_path.exists():
            return ToolResult(success=False, error=f"eSpeak NG output file missing: {output_path}")

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "voice": voice,
                "text_length": len(inputs["text"]),
                "output": str(output_path),
                "format": "wav",
            },
            artifacts=[str(output_path)],
            model=f"espeak-ng:{voice}",
        )
