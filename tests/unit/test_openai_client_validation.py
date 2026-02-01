"""Unit tests for OpenAI client API key validation."""

import pytest

from app.llm.openai_client import validate_api_key, mask_api_key


class TestValidateAPIKey:
    """Tests for validate_api_key function."""

    def test_missing_key_raises_error(self):
        """Test that missing key raises ValueError."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            validate_api_key(None)
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            validate_api_key("")

    def test_placeholder_patterns_rejected(self):
        """Test that obvious placeholder patterns are rejected."""
        placeholders = [
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
        
        for placeholder in placeholders:
            with pytest.raises(ValueError, match="appears to be a placeholder"):
                validate_api_key(placeholder)

    def test_sk_proj_key_accepted(self):
        """Test that real OpenAI project keys (sk-proj-*) are accepted."""
        # Real OpenAI project keys start with "sk-proj-"
        project_key = "sk-proj-1234567890ABCDEFGHIJKLMN"
        result = validate_api_key(project_key)
        assert result == project_key

    def test_sk_regular_key_accepted(self):
        """Test that regular OpenAI keys (sk-*) are accepted."""
        regular_key = "sk-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = validate_api_key(regular_key)
        assert result == regular_key

    def test_too_short_key_rejected(self):
        """Test that keys shorter than 20 characters are rejected."""
        short_key = "sk-12345"  # Only 8 characters
        with pytest.raises(ValueError, match="too short"):
            validate_api_key(short_key)

    def test_invalid_prefix_rejected(self):
        """Test that keys not starting with 'sk-' are rejected."""
        invalid_key = "invalid-key-1234567890ABCDEFGHIJKLMN"
        with pytest.raises(ValueError, match="does not start with 'sk-'"):
            validate_api_key(invalid_key)

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped from keys."""
        key_with_whitespace = "  sk-proj-1234567890ABCDEFGHIJKLMN  "
        result = validate_api_key(key_with_whitespace)
        assert result == "sk-proj-1234567890ABCDEFGHIJKLMN"

    def test_long_valid_key_accepted(self):
        """Test that long valid keys are accepted."""
        long_key = "sk-proj-" + "A" * 50  # 58 characters total
        result = validate_api_key(long_key)
        assert result == long_key


class TestMaskAPIKey:
    """Tests for mask_api_key function."""

    def test_missing_key(self):
        """Test masking of missing key."""
        assert mask_api_key(None) == "<MISSING>"
        assert mask_api_key("") == "<MISSING>"

    def test_too_short_key(self):
        """Test masking of too short key."""
        assert mask_api_key("sk-12345") == "<TOO_SHORT>"

    def test_valid_key_masked(self):
        """Test masking of valid key."""
        key = "sk-proj-1234567890ABCDEFGHIJKLMN"
        masked = mask_api_key(key)
        assert masked.startswith("sk-pro")
        assert masked.endswith("KLMN")
        assert "…" in masked

    def test_whitespace_stripped_before_masking(self):
        """Test that whitespace is stripped before masking."""
        key = "  sk-proj-1234567890ABCDEFGHIJKLMN  "
        masked = mask_api_key(key)
        assert masked.startswith("sk-pro")
        assert masked.endswith("KLMN")
