"""One-shot check that gpt-4o-2024-08-06 is available via Modal openai secret."""

from __future__ import annotations

import modal

app = modal.App("pre-output-physiology-phase21-grader-check")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "openai==1.59.6",
    "httpx==0.27.2",
)


@app.function(image=image, secrets=[modal.Secret.from_name("openai")])
def check() -> str:
    import os

    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("STOP: OPENAI_API_KEY missing")
    client = OpenAI()
    model = "gpt-4o-2024-08-06"
    try:
        m = client.models.retrieve(model)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"STOP: exact grader model {model} unavailable: {e}") from e
    return m.id


@app.local_entrypoint()
def main() -> None:
    print(check.remote())
