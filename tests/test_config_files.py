"""Tests for configuration files added/modified in this PR.

Covers:
- .claude/launch.json  (new file)
- .coderabbit.yaml     (updated tone_instructions and path_filters)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

# Root of the repository, two levels above this test file.
REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_launch_json() -> dict:
    path = REPO_ROOT / ".claude" / "launch.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_coderabbit_yaml() -> dict:
    path = REPO_ROOT / ".coderabbit.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# .claude/launch.json
# ---------------------------------------------------------------------------

class TestLaunchJsonStructure:
    """Validate the top-level schema of .claude/launch.json."""

    def test_file_is_valid_json(self):
        path = REPO_ROOT / ".claude" / "launch.json"
        content = path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_has_version_field(self):
        data = _load_launch_json()
        assert "version" in data

    def test_version_is_string(self):
        data = _load_launch_json()
        assert isinstance(data["version"], str)

    def test_version_value(self):
        data = _load_launch_json()
        assert data["version"] == "0.0.1"

    def test_has_configurations_field(self):
        data = _load_launch_json()
        assert "configurations" in data

    def test_configurations_is_list(self):
        data = _load_launch_json()
        assert isinstance(data["configurations"], list)

    def test_configurations_is_non_empty(self):
        data = _load_launch_json()
        assert len(data["configurations"]) >= 1


class TestLaunchJsonConfiguration:
    """Validate each configuration entry in .claude/launch.json."""

    @pytest.fixture()
    def first_config(self) -> dict:
        return _load_launch_json()["configurations"][0]

    def test_config_has_name(self, first_config):
        assert "name" in first_config

    def test_config_name_is_string(self, first_config):
        assert isinstance(first_config["name"], str)

    def test_config_name_value(self, first_config):
        assert first_config["name"] == "AKS Web UI"

    def test_config_has_runtime_executable(self, first_config):
        assert "runtimeExecutable" in first_config

    def test_runtime_executable_is_string(self, first_config):
        assert isinstance(first_config["runtimeExecutable"], str)

    def test_runtime_executable_is_non_empty(self, first_config):
        assert first_config["runtimeExecutable"].strip() != ""

    def test_config_has_runtime_args(self, first_config):
        assert "runtimeArgs" in first_config

    def test_runtime_args_is_list(self, first_config):
        assert isinstance(first_config["runtimeArgs"], list)

    def test_runtime_args_non_empty(self, first_config):
        assert len(first_config["runtimeArgs"]) > 0

    def test_runtime_args_all_strings(self, first_config):
        assert all(isinstance(a, str) for a in first_config["runtimeArgs"])

    def test_runtime_args_include_serve(self, first_config):
        """The configuration must launch the 'serve' sub-command."""
        assert "serve" in first_config["runtimeArgs"]

    def test_runtime_args_include_aks(self, first_config):
        """The runtime args must reference the 'aks' CLI entry-point."""
        assert "aks" in first_config["runtimeArgs"]

    def test_config_has_port(self, first_config):
        assert "port" in first_config

    def test_port_is_integer(self, first_config):
        assert isinstance(first_config["port"], int)

    def test_port_value(self, first_config):
        assert first_config["port"] == 7337

    def test_port_is_valid_range(self, first_config):
        assert 1 <= first_config["port"] <= 65535

    def test_reload_flag_in_args(self, first_config):
        """Hot-reload flag must be present for development convenience."""
        assert "--reload" in first_config["runtimeArgs"]

    def test_no_extra_unknown_top_level_fields(self):
        """Only the two expected top-level keys should be present."""
        data = _load_launch_json()
        assert set(data.keys()) == {"version", "configurations"}


# ---------------------------------------------------------------------------
# .coderabbit.yaml
# ---------------------------------------------------------------------------

class TestCodeRabbitYamlStructure:
    """Validate the top-level schema of .coderabbit.yaml."""

    def test_file_is_valid_yaml(self):
        path = REPO_ROOT / ".coderabbit.yaml"
        content = path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)

    def test_has_language_field(self):
        data = _load_coderabbit_yaml()
        assert "language" in data

    def test_language_value(self):
        data = _load_coderabbit_yaml()
        assert data["language"] == "en-US"

    def test_has_tone_instructions(self):
        data = _load_coderabbit_yaml()
        assert "tone_instructions" in data

    def test_tone_instructions_is_string(self):
        data = _load_coderabbit_yaml()
        assert isinstance(data["tone_instructions"], str)

    def test_has_enable_free_tier(self):
        data = _load_coderabbit_yaml()
        assert "enable_free_tier" in data

    def test_enable_free_tier_is_bool(self):
        data = _load_coderabbit_yaml()
        assert isinstance(data["enable_free_tier"], bool)

    def test_enable_free_tier_is_true(self):
        data = _load_coderabbit_yaml()
        assert data["enable_free_tier"] is True

    def test_has_reviews_section(self):
        data = _load_coderabbit_yaml()
        assert "reviews" in data

    def test_has_chat_section(self):
        data = _load_coderabbit_yaml()
        assert "chat" in data


class TestCodeRabbitToneInstructions:
    """Validate the updated tone_instructions content."""

    @pytest.fixture()
    def tone(self) -> str:
        return _load_coderabbit_yaml()["tone_instructions"]

    def test_mentions_python(self, tone):
        assert "Python" in tone

    def test_mentions_aks_project(self, tone):
        """Tone should identify this as the AKS project."""
        assert "AKS" in tone

    def test_mentions_pep8(self, tone):
        assert "PEP 8" in tone

    def test_mentions_type_hints(self, tone):
        assert "type hints" in tone

    def test_mentions_async(self, tone):
        assert "async" in tone

    def test_mentions_llm_integration(self, tone):
        assert "LLM" in tone

    def test_mentions_security(self, tone):
        assert "Security" in tone or "security" in tone

    def test_mentions_api_key_handling(self, tone):
        assert "API key" in tone

    def test_is_non_empty(self, tone):
        assert len(tone.strip()) > 0


class TestCodeRabbitPathFilters:
    """Validate path_filters reflect the Python project layout."""

    @pytest.fixture()
    def filters(self) -> list[str]:
        return _load_coderabbit_yaml()["reviews"]["path_filters"]

    def test_filters_is_list(self, filters):
        assert isinstance(filters, list)

    def test_filters_non_empty(self, filters):
        assert len(filters) > 0

    def test_all_filters_are_strings(self, filters):
        assert all(isinstance(f, str) for f in filters)

    # Python-specific filters that must be present (added in this PR)
    def test_excludes_egg_info(self, filters):
        assert "!*.egg-info/**" in filters

    def test_excludes_pycache(self, filters):
        assert "!__pycache__/**" in filters

    def test_excludes_pyc_files(self, filters):
        assert "!*.pyc" in filters

    def test_excludes_venv(self, filters):
        assert "!.venv/**" in filters

    def test_excludes_knowledge_index(self, filters):
        assert "!knowledge/.index/**" in filters

    def test_excludes_knowledge_conversations(self, filters):
        assert "!knowledge/conversations/**" in filters

    def test_excludes_knowledge_documents(self, filters):
        assert "!knowledge/documents/**" in filters

    def test_excludes_claude_config(self, filters):
        assert "!.claude/**" in filters

    def test_excludes_github(self, filters):
        assert "!.github/**" in filters

    def test_excludes_log_files(self, filters):
        assert "!*.log" in filters

    def test_excludes_dist(self, filters):
        assert "!dist/**" in filters

    def test_excludes_build(self, filters):
        assert "!build/**" in filters

    # Node.js / frontend patterns that must NOT be present (removed in this PR)
    def test_does_not_exclude_node_modules(self, filters):
        assert "!node_modules/**" not in filters

    def test_does_not_exclude_min_js(self, filters):
        assert "!*.min.js" not in filters

    def test_does_not_exclude_ai_dir(self, filters):
        assert "!.ai/**" not in filters

    def test_does_not_exclude_angular_dir(self, filters):
        assert "!.angular/**" not in filters

    def test_does_not_exclude_vscode_dir(self, filters):
        assert "!.vscode/**" not in filters

    def test_does_not_exclude_docs_dir(self, filters):
        assert "!.docs/**" not in filters

    def test_all_filters_are_exclusions(self, filters):
        """Every filter in this config is a negation pattern (starts with '!')."""
        assert all(f.startswith("!") for f in filters)

    # Boundary / regression cases
    def test_no_duplicate_filters(self, filters):
        assert len(filters) == len(set(filters))

    def test_no_blank_filter_entries(self, filters):
        assert all(f.strip() for f in filters)


class TestCodeRabbitReviewsSection:
    """Validate reviews section settings."""

    @pytest.fixture()
    def reviews(self) -> dict:
        return _load_coderabbit_yaml()["reviews"]

    def test_has_profile(self, reviews):
        assert "profile" in reviews

    def test_profile_value(self, reviews):
        assert reviews["profile"] == "chill"

    def test_high_level_summary_enabled(self, reviews):
        assert reviews.get("high_level_summary") is True

    def test_auto_approve_disabled(self, reviews):
        assert reviews.get("auto_approve_workflow") is False

    def test_auto_review_enabled(self, reviews):
        assert reviews["auto_review"]["enabled"] is True

    def test_auto_review_drafts_disabled(self, reviews):
        assert reviews["auto_review"]["drafts"] is False

    def test_has_path_filters(self, reviews):
        assert "path_filters" in reviews