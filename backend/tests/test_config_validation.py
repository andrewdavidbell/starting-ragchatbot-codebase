"""
Configuration validation tests
Ensures that config.py has valid values and catches misconfigurations
"""

import pytest
from config import Config


@pytest.mark.unit
class TestConfigValidation:
    """Test that configuration values are valid"""

    def test_max_results_not_zero(self):
        """
        CRITICAL: MAX_RESULTS must be greater than 0
        If this fails, all searches will return 0 results!
        """
        assert Config.MAX_RESULTS > 0, (
            f"MAX_RESULTS is {Config.MAX_RESULTS}. "
            "This will cause all vector searches to return 0 results! "
            "Fix: Change MAX_RESULTS to at least 5 in backend/config.py:21"
        )

    def test_max_results_reasonable_range(self):
        """Test that MAX_RESULTS is in a reasonable range"""
        assert Config.MAX_RESULTS >= 3, (
            f"MAX_RESULTS={Config.MAX_RESULTS} is too low. "
            "Recommended: 5-10 for good search quality"
        )
        assert Config.MAX_RESULTS <= 50, (
            f"MAX_RESULTS={Config.MAX_RESULTS} is extremely high. "
            "This may cause token limit issues. Recommended: 5-10"
        )

    def test_chunk_size_positive(self):
        """Test that CHUNK_SIZE is positive"""
        assert Config.CHUNK_SIZE > 0, (
            f"CHUNK_SIZE must be positive, got {Config.CHUNK_SIZE}"
        )

    def test_chunk_size_reasonable(self):
        """Test that CHUNK_SIZE is in a reasonable range"""
        assert Config.CHUNK_SIZE >= 100, (
            f"CHUNK_SIZE={Config.CHUNK_SIZE} is too small. "
            "Chunks may not contain enough context. Recommended: 500-1000"
        )
        assert Config.CHUNK_SIZE <= 2000, (
            f"CHUNK_SIZE={Config.CHUNK_SIZE} is very large. "
            "This may cause inefficient chunking. Recommended: 500-1000"
        )

    def test_chunk_overlap_valid(self):
        """Test that CHUNK_OVERLAP is valid"""
        assert Config.CHUNK_OVERLAP >= 0, (
            f"CHUNK_OVERLAP must be non-negative, got {Config.CHUNK_OVERLAP}"
        )
        assert Config.CHUNK_OVERLAP < Config.CHUNK_SIZE, (
            f"CHUNK_OVERLAP ({Config.CHUNK_OVERLAP}) must be less than "
            f"CHUNK_SIZE ({Config.CHUNK_SIZE})"
        )

    def test_chunk_overlap_reasonable(self):
        """Test that CHUNK_OVERLAP is a reasonable percentage of CHUNK_SIZE"""
        overlap_ratio = Config.CHUNK_OVERLAP / Config.CHUNK_SIZE
        assert overlap_ratio <= 0.3, (
            f"CHUNK_OVERLAP ({Config.CHUNK_OVERLAP}) is {overlap_ratio*100:.0f}% of "
            f"CHUNK_SIZE ({Config.CHUNK_SIZE}). Recommended: 10-20% overlap"
        )

    def test_max_history_positive(self):
        """Test that MAX_HISTORY is positive"""
        assert Config.MAX_HISTORY > 0, (
            f"MAX_HISTORY must be positive, got {Config.MAX_HISTORY}"
        )

    def test_max_history_reasonable(self):
        """Test that MAX_HISTORY is in a reasonable range"""
        assert Config.MAX_HISTORY >= 3, (
            f"MAX_HISTORY={Config.MAX_HISTORY} is too small. "
            "Users need some conversation context. Recommended: 5-10"
        )
        assert Config.MAX_HISTORY <= 50, (
            f"MAX_HISTORY={Config.MAX_HISTORY} is very large. "
            "This may cause token limit issues. Recommended: 5-10"
        )

    def test_embedding_model_set(self):
        """Test that EMBEDDING_MODEL is configured"""
        assert Config.EMBEDDING_MODEL, "EMBEDDING_MODEL is not set"
        assert isinstance(Config.EMBEDDING_MODEL, str), (
            f"EMBEDDING_MODEL must be a string, got {type(Config.EMBEDDING_MODEL)}"
        )

    def test_anthropic_model_set(self):
        """Test that ANTHROPIC_MODEL is configured"""
        assert Config.ANTHROPIC_MODEL, "ANTHROPIC_MODEL is not set"
        assert isinstance(Config.ANTHROPIC_MODEL, str), (
            f"ANTHROPIC_MODEL must be a string, got {type(Config.ANTHROPIC_MODEL)}"
        )

    def test_anthropic_model_valid(self):
        """Test that ANTHROPIC_MODEL is a known Claude model"""
        valid_model_prefixes = [
            "claude-3",
            "claude-sonnet",
            "claude-opus",
            "claude-haiku"
        ]
        assert any(prefix in Config.ANTHROPIC_MODEL for prefix in valid_model_prefixes), (
            f"ANTHROPIC_MODEL='{Config.ANTHROPIC_MODEL}' doesn't appear to be a valid Claude model. "
            f"Expected one of: {valid_model_prefixes}"
        )

    def test_docs_directory_set(self):
        """Test that DOCS_DIRECTORY is configured"""
        assert Config.DOCS_DIRECTORY, "DOCS_DIRECTORY is not set"
        assert isinstance(Config.DOCS_DIRECTORY, str), (
            f"DOCS_DIRECTORY must be a string, got {type(Config.DOCS_DIRECTORY)}"
        )

    def test_chroma_directory_set(self):
        """Test that CHROMA_DIRECTORY is configured"""
        assert Config.CHROMA_DIRECTORY, "CHROMA_DIRECTORY is not set"
        assert isinstance(Config.CHROMA_DIRECTORY, str), (
            f"CHROMA_DIRECTORY must be a string, got {type(Config.CHROMA_DIRECTORY)}"
        )


@pytest.mark.integration
class TestConfigEnvironmentVariables:
    """Test configuration from environment variables"""

    def test_config_has_all_required_fields(self):
        """Test that Config class has all expected fields"""
        required_fields = [
            'CHUNK_SIZE',
            'CHUNK_OVERLAP',
            'MAX_RESULTS',
            'MAX_HISTORY',
            'EMBEDDING_MODEL',
            'ANTHROPIC_MODEL',
            'DOCS_DIRECTORY',
            'CHROMA_DIRECTORY'
        ]

        for field in required_fields:
            assert hasattr(Config, field), (
                f"Config is missing required field: {field}"
            )

    def test_config_dataclass_structure(self):
        """Test that Config is properly structured as a dataclass"""
        import dataclasses
        assert dataclasses.is_dataclass(Config), (
            "Config should be a dataclass"
        )


@pytest.mark.unit
class TestConfigDefaultValues:
    """Test default configuration values are sensible"""

    def test_default_chunk_size(self):
        """Test default CHUNK_SIZE value"""
        # Current default should be 800
        assert Config.CHUNK_SIZE == 800, (
            f"Default CHUNK_SIZE changed from 800 to {Config.CHUNK_SIZE}. "
            "Verify this is intentional."
        )

    def test_default_chunk_overlap(self):
        """Test default CHUNK_OVERLAP value"""
        # Current default should be 100
        assert Config.CHUNK_OVERLAP == 100, (
            f"Default CHUNK_OVERLAP changed from 100 to {Config.CHUNK_OVERLAP}. "
            "Verify this is intentional."
        )

    def test_default_max_history(self):
        """Test default MAX_HISTORY value"""
        # Current default should be 10
        assert Config.MAX_HISTORY == 10, (
            f"Default MAX_HISTORY changed from 10 to {Config.MAX_HISTORY}. "
            "Verify this is intentional."
        )

    def test_default_embedding_model(self):
        """Test default EMBEDDING_MODEL value"""
        # Current default should be all-MiniLM-L6-v2
        assert Config.EMBEDDING_MODEL == "all-MiniLM-L6-v2", (
            f"Default EMBEDDING_MODEL changed from 'all-MiniLM-L6-v2' to '{Config.EMBEDDING_MODEL}'. "
            "Verify this is intentional and model is available."
        )

    def test_default_anthropic_model(self):
        """Test default ANTHROPIC_MODEL value"""
        # Current default should be claude-sonnet-4-20250514
        assert Config.ANTHROPIC_MODEL == "claude-sonnet-4-20250514", (
            f"Default ANTHROPIC_MODEL changed from 'claude-sonnet-4-20250514' to '{Config.ANTHROPIC_MODEL}'. "
            "Verify this is intentional and model is available."
        )


@pytest.mark.integration
class TestConfigCombinations:
    """Test valid combinations of configuration values"""

    def test_chunk_overlap_percentage_of_chunk_size(self):
        """Test that overlap is a reasonable percentage of chunk size"""
        overlap_ratio = Config.CHUNK_OVERLAP / Config.CHUNK_SIZE

        # Overlap should be between 5% and 30% of chunk size
        assert 0.05 <= overlap_ratio <= 0.30, (
            f"CHUNK_OVERLAP ({Config.CHUNK_OVERLAP}) is {overlap_ratio*100:.1f}% of "
            f"CHUNK_SIZE ({Config.CHUNK_SIZE}). "
            f"Recommended range: 5-30% (40-240 chars for chunk size 800)"
        )

    def test_max_results_and_context_window(self):
        """
        Test that MAX_RESULTS won't cause token limit issues
        Rough estimate: chunk_size * max_results should be reasonable
        """
        estimated_context = Config.CHUNK_SIZE * Config.MAX_RESULTS

        # With 800 char chunks and 5 results = 4000 chars ≈ 1000 tokens (safe)
        # With 800 char chunks and 10 results = 8000 chars ≈ 2000 tokens (safe)
        # With 800 char chunks and 20 results = 16000 chars ≈ 4000 tokens (borderline)

        assert estimated_context <= 20000, (
            f"MAX_RESULTS ({Config.MAX_RESULTS}) × CHUNK_SIZE ({Config.CHUNK_SIZE}) "
            f"= {estimated_context} chars may exceed token limits. "
            f"Consider reducing MAX_RESULTS or CHUNK_SIZE."
        )


@pytest.mark.unit
class TestConfigDocumentation:
    """Test that configuration is properly documented"""

    def test_config_file_exists(self):
        """Test that config.py file exists and is importable"""
        try:
            import config
            assert hasattr(config, 'Config'), "config.py must export Config class"
        except ImportError as e:
            pytest.fail(f"Could not import config: {e}")

    def test_config_class_has_docstring(self):
        """Test that Config class has documentation"""
        from config import Config

        # Check if class has a docstring
        assert Config.__doc__ is not None, (
            "Config class should have a docstring explaining configuration"
        )


@pytest.mark.integration
class TestConfigSystemImpact:
    """Test how configuration affects system behaviour"""

    def test_max_results_zero_impact(self):
        """
        CRITICAL: Document the impact of MAX_RESULTS=0
        This is the main bug we're diagnosing
        """
        if Config.MAX_RESULTS == 0:
            pytest.fail(
                "MAX_RESULTS=0 detected! This causes the following issues:\n"
                "1. All vector searches return 0 results\n"
                "2. AI has no course content to answer questions\n"
                "3. Users see 'query failed' or unhelpful responses\n"
                "4. Search appears to work but returns nothing\n\n"
                "FIX: Change MAX_RESULTS to 5 in backend/config.py:21\n"
                "Recommended value: 5 for balance of quality and performance"
            )

    def test_configuration_summary(self):
        """Print current configuration for debugging"""
        config_summary = f"""
Current RAG System Configuration:
==================================
CHUNK_SIZE:       {Config.CHUNK_SIZE}
CHUNK_OVERLAP:    {Config.CHUNK_OVERLAP}
MAX_RESULTS:      {Config.MAX_RESULTS} {'⚠️  CRITICAL: ZERO!' if Config.MAX_RESULTS == 0 else '✓'}
MAX_HISTORY:      {Config.MAX_HISTORY}
EMBEDDING_MODEL:  {Config.EMBEDDING_MODEL}
ANTHROPIC_MODEL:  {Config.ANTHROPIC_MODEL}
DOCS_DIRECTORY:   {Config.DOCS_DIRECTORY}
CHROMA_DIRECTORY: {Config.CHROMA_DIRECTORY}

Calculated Values:
==================
Overlap Ratio:    {Config.CHUNK_OVERLAP / Config.CHUNK_SIZE * 100:.1f}%
Est. Context Size: ~{Config.CHUNK_SIZE * Config.MAX_RESULTS} chars
        """
        print(config_summary)

        # Always pass - this is just informational
        assert True
