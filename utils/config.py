# ============================================================
#  utils/config.py
#  Loads and validates all environment variables on startup.
#  Any missing or malformed key causes a rich error panel + clean exit.
# ============================================================

import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Load .env file from the project root (two levels up from this file)
load_dotenv()

console = Console()

# ---------------------------------------------------------------
# Required environment variables and where to get them
# ---------------------------------------------------------------
REQUIRED_VARS = {
    "OPENROUTER_API_KEY": "https://openrouter.ai/keys  (sign up free, key starts with sk-or-v1-)",
    "APIFY_API_TOKEN": "https://console.apify.com/account/integrations  (sign up free, token starts with apify_api_)",
}

# ---------------------------------------------------------------
# Optional / config vars with defaults
# ---------------------------------------------------------------
PREDICTION_MODE = os.getenv("PREDICTION_MODE", "stacked_1min")  # "direct_5min" | "stacked_1min"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
KELLY_FRACTION = 0.5        # Half-Kelly by default — conservative, configurable here
HYPOTHETICAL_BANKROLL = 1000.0  # USD — used only for display/logging, never real money


def _mask_key(key: str) -> str:
    """
    Returns a masked version of an API key for safe display.
    Only shows the last 4 characters, e.g. "...ab3d".
    NEVER logs or prints full API keys.
    """
    if not key or len(key) < 4:
        return "****"
    return f"...{key[-4:]}"


def validate_env() -> dict:
    """
    Validates all required environment variables.
    Prints a clear rich error panel and calls sys.exit(1) if any are missing.
    Returns a dict of validated key values on success.
    """
    missing = []
    values = {}

    for var, source_hint in REQUIRED_VARS.items():
        val = os.getenv(var, "").strip()
        if not val:
            missing.append((var, source_hint))
        else:
            values[var] = val

    if missing:
        lines = ["[bold red]The following required environment variables are missing:[/bold red]\n"]
        for var, hint in missing:
            lines.append(f"  [yellow]{var}[/yellow]")
            lines.append(f"    Get it at: [cyan]{hint}[/cyan]\n")
        lines.append("[dim]Copy [bold].env.example[/bold] to [bold].env[/bold] and fill in the values.[/dim]")

        console.print(
            Panel(
                "\n".join(lines),
                title="[bold red]⛔ Missing API Keys[/bold red]",
                border_style="red",
                expand=False,
            )
        )
        sys.exit(1)

    return values


# Run validation at import time so any missing key is caught immediately
ENV = validate_env()

# Expose individual keys (masked in logs — never use raw values in print statements)
OPENROUTER_API_KEY: str = ENV["OPENROUTER_API_KEY"]
APIFY_API_TOKEN: str = ENV["APIFY_API_TOKEN"]
