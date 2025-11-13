import os
from pathlib import Path

import numpy as np
import requests
from pydantic import BaseModel, ConfigDict

# LLM API configuration
LLM_API_URI = os.getenv("LLM_API_URI", "http://localhost:11434")
LLM_EMBED_PATH = os.getenv("LLM_EMBED_PATH", "api/embed")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")

# Training data directory path
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "macos_correlation_rules")

# Prompt template for LLM to normalize SIEM events
PROMPT_CORRELATION_TEMPLATE = """
You are an Information Security engineer. You are working on correlation rules.

The file contains structured but not normalized events:
{directory}/events_{file_idx}.json 

Generate a file at the following path with normalized SIEM fields for the events above:
{directory}/norm_fields_{file_idx}.json

Output ONLY a valid JSON object. No explanations.
"""


def render_prompt_correlation(**kwargs) -> str:
    """
    Renders a prompt template by replacing placeholders with provided values.
    """
    return PROMPT_CORRELATION_TEMPLATE.format(**kwargs)


def get_embedding(text: str) -> list[float]:
    """
    Generates vector embedding for the input text using embedding API.
    """
    resp = requests.post(
        f"{LLM_API_URI}/{LLM_EMBED_PATH}",
        json={
            "model": LLM_MODEL,
            "input": text,
        },
    )

    return resp.json()["embeddings"][0]


class ReferenceItem(BaseModel):
    """
    Data model for storing reference examples with original events,
    normalized events, and embeddings.
    """

    event_text: str  # Original unnormalized event text
    norm_text: str  # Normalized SIEM event text
    embed: np.ndarray  # Vector embedding of the event text

    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_reference_examples(base_path: str) -> list[ReferenceItem]:
    """
    Loads reference examples from directory structure by finding test directories and processing event files.

    Searches for pattern: base_path/*/tests/events_*.json and corresponding norm_fields_*.json files.
    Generates embeddings for each event and creates ReferenceItem objects.
    """
    base = Path(base_path)
    references = []

    # Iterate through all test directories matching pattern: base_path/*/tests
    for test_dir in base.glob("*/*/tests"):
        if not test_dir.is_dir():
            continue

        # Scan for event files in current test directory
        for event_file in test_dir.glob("events_*.json"):
            norm_file = event_file.with_name(
                event_file.name.replace("events_", "norm_fields_")
            )
            # Skip if normalized file doesn't exist
            if not norm_file.exists():
                continue

            # Read event and normalized text files
            event_text = event_file.read_text()
            norm_text = norm_file.read_text()

            # Generate embedding vector for the event text
            embed = get_embedding(event_text)

            references.append(
                ReferenceItem(
                    event_text=event_text,
                    norm_text=norm_text,
                    embed=np.array(embed, dtype=np.float32),
                )
            )
            break  # TODO
        break  # TODO

    return references


def main():
    # Load all reference examples from the training data directory
    # and generate embedings for each
    ref_exs = load_reference_examples(TRAIN_DATA_PATH)
    print(ref_exs)


if __name__ == "__main__":
    main()
