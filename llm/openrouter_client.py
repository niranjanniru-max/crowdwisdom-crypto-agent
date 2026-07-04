# ============================================================
#  llm/openrouter_client.py
#  Single LLM wrapper used by ALL agents.
#
#  DESIGN: A single ordered fallback chain of free OpenRouter models.
#  Swapping or adding models is a one-line change to the list below.
#  The wrapper tries each model in order, handles errors gracefully,
#  and logs which model ultimately responded.
#
#  NOTE: The free model roster on OpenRouter rotates over time.
#  If every model in OPENROUTER_MODEL_FALLBACK_CHAIN fails with
#  a "model not found" or "no such model" style error, check:
#  https://openrouter.ai/models?q=&order=top-weekly&supported_parameters=free
#  and update the list below with currently available free models.
# ============================================================

import time
import logging

import openai

from utils.config import OPENROUTER_API_KEY, _mask_key
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------
# Ordered list of free OpenRouter models to try.
# The wrapper tries index 0, then 1, then 2, etc.
# ---------------------------------------------------------------
OPENROUTER_MODEL_FALLBACK_CHAIN = [
    "openrouter/free",
    "deepseek/deepseek-r1:free",
    "qwen/qwen3-coder:free",
]

# Create one shared OpenAI client pointed at OpenRouter
_client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# Exponential back-off base delay (seconds) for rate-limit retries
_BACKOFF_BASE = 2  # waits 2s, 4s, 8s


def call_llm(
    prompt: str,
    system: str | None = None,
    max_tokens: int = 300,
    response_format: dict | None = None,
    temperature: float = 0.7,
) -> str:
    """
    Send a prompt to the LLM using the fallback chain.

    Tries each model in OPENROUTER_MODEL_FALLBACK_CHAIN in order:
    - 401/403  → authentication error; tells user to check their key
    - 429      → rate limit; exponential back-off up to 3 retries per model
    - timeout  → retries once, then moves to next model
    - bad shape → logs raw response, moves to next model

    Returns the assistant's reply string.
    Raises RuntimeError only if every model in the chain fails.

    Args:
        prompt:     User-facing text to send.
        system:     Optional system message (agent role definition).
        max_tokens: Max tokens in the reply.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error: Exception | None = None

    for model in OPENROUTER_MODEL_FALLBACK_CHAIN:
        log.debug(f"Trying LLM model: {model}")
        attempt = 0
        max_attempts = 3

        while attempt < max_attempts:
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "timeout": 30,
                }
                if response_format:
                    kwargs["response_format"] = response_format
                    
                response = _client.chat.completions.create(**kwargs)
                
                resp_dict = response.model_dump()
                if not resp_dict.get("choices"):
                    log.warning(f"Model {model} returned unexpected response: {resp_dict}")
                    break
                    
                choice = resp_dict['choices'][0]
                content = choice.get('message', {}).get('content')
                reasoning = choice.get('message', {}).get('reasoning')

                # Catch case where model dumped its thoughts into reasoning block
                if not content or content.strip() == "":
                    if reasoning:
                        content = reasoning
                    else:
                        raise ValueError("Model returned an entirely empty response payload.")

                content = content.strip()
                log.info(f"LLM responded via model: [cyan]{model}[/cyan] (key: {_mask_key(OPENROUTER_API_KEY)})")
                return content

            except openai.AuthenticationError as e:
                # 401/403 — key is wrong; no point retrying other attempts
                log.error(
                    f"[red]Authentication error with OpenRouter[/red] "
                    f"(model: {model}). Check your OPENROUTER_API_KEY. "
                    f"Key ending: {_mask_key(OPENROUTER_API_KEY)}"
                )
                last_error = e
                break  # skip retries; try next model

            except openai.RateLimitError as e:
                # 429 — back off and retry same model up to max_attempts
                wait = _BACKOFF_BASE ** (attempt + 1)
                log.warning(
                    f"[yellow]Rate limited by OpenRouter[/yellow] "
                    f"(model: {model}, attempt {attempt+1}/{max_attempts}). "
                    f"Backing off {wait}s…"
                )
                time.sleep(wait)
                attempt += 1
                last_error = e

            except openai.APITimeoutError as e:
                if attempt == 0:
                    log.warning(f"Timeout calling {model}. Retrying once…")
                    attempt += 1
                    last_error = e
                else:
                    log.warning(f"Timeout on retry for {model}. Moving to next model.")
                    last_error = e
                    break

            except (openai.NotFoundError, openai.APIStatusError) as e:
                # Catch 404 (NotFoundError) and other API status errors (e.g. 502, unhandled 429)
                log.warning(f"Endpoint or status error from {model}: {e}. Moving to next model.")
                last_error = e
                break

            except openai.APIError as e:
                # Catch-all for other API errors
                log.warning(f"API error from {model}: {e}. Moving to next model.")
                last_error = e
                break

            except Exception as e:
                log.warning(f"Unexpected error calling {model}: {e}. Moving to next model.")
                last_error = e
                break

    # Every model in the chain failed
    raise RuntimeError(
        f"All models in OPENROUTER_MODEL_FALLBACK_CHAIN failed. "
        f"Last error: {last_error}. "
        f"Check https://openrouter.ai/models for currently available free models."
    )
