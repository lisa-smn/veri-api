import os
from dotenv import load_dotenv

load_dotenv()
from typing import Any

from openai import OpenAI

from app.llm.llm_client import LLMClient


def mask_api_key(key: str | None) -> str:
    """Maskiert API-Key für sichere Logging (erste 6 + letzte 4 Zeichen)."""
    if not key:
        return "<MISSING>"
    key = key.strip()
    if len(key) <= 12:
        return "<TOO_SHORT>"
    return f"{key[:6]}…{key[-4:]}"


def validate_api_key(key: str | None) -> str:
    """
    Validiert API-Key und wirft Exception bei fehlendem oder Platzhalter-Key.
    
    Raises:
        ValueError: Wenn Key fehlt, zu kurz ist oder Platzhalter enthält.
    """
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Set it in .env file or as environment variable. "
            "Required for LLM-based verification."
        )
    
    key = key.strip()
    
    # Prüfe auf Platzhalter-Patterns (nur offensichtliche Platzhalter, keine echten Key-Patterns)
    placeholder_patterns = [
        "your-ope",
        "your-openai",
        "your-openai-api-key-here",
        "your-key",
        "your-key-here",
        "********",
        "changeme",
        "replace-me",
        "your_api_key",
    ]
    
    key_lower = key.lower()
    for pattern in placeholder_patterns:
        if pattern in key_lower:
            raise ValueError(
                f"OPENAI_API_KEY appears to be a placeholder: '{mask_api_key(key)}'. "
                f"Please set a valid API key in .env file or environment variable."
            )
    
    # Prüfe minimale Länge (OpenAI Keys sind typisch > 20 Zeichen)
    if len(key) < 20:
        raise ValueError(
            f"OPENAI_API_KEY is too short (length: {len(key)}). "
            f"Valid OpenAI API keys are typically 40+ characters. "
            f"Current value (masked): {mask_api_key(key)}"
        )
    
    # Prüfe auf gültiges Prefix (OpenAI Keys starten mit "sk-")
    if not key.startswith("sk-"):
        raise ValueError(
            f"OPENAI_API_KEY does not start with 'sk-'. "
            f"Valid OpenAI API keys start with 'sk-'. "
            f"Current value (masked): {mask_api_key(key)}"
        )
    
    return key


class OpenAIClient(LLMClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        # Lade und validiere API-Key
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Debug-Logging (maskiert)
        key_source = "ENV"
        if not api_key:
            # Prüfe ob .env geladen wurde
            if os.path.exists(".env"):
                key_source = ".env (but key not found)"
            else:
                key_source = "ENV (no .env file)"
        
        print(f"[OpenAIClient] OPENAI_API_KEY(masked)={mask_api_key(api_key)} | source={key_source} | process=script")
        
        # Fail-fast: Validiere Key bevor Client erstellt wird
        validated_key = validate_api_key(api_key)
        
        # Erstelle Client mit explizitem API-Key (für bessere Fehlerbehandlung)
        self.client = OpenAI(api_key=validated_key)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800,
        )

        return response.choices[0].message.content or ""
