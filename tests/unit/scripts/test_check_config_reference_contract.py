"""Self-tests for ``scripts/check-config-reference-contract.py``.

The repository-level assertion matters only if its scanner and documentation
markers fail in both directions: an unwired field must be rejected, and a
newly wired field must stop being described as inert.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "check-config-reference-contract.py"


@pytest.fixture(scope="module")
def contract():
    spec = importlib.util.spec_from_file_location("check_config_reference_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_current_repository_passes_standalone_contract() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Config reference contract OK" in result.stdout


def test_runtime_scan_finds_attribute_alias_and_literal_getattr_reads(
    contract, tmp_path: Path
) -> None:
    source = tmp_path / "runtime.py"
    source.write_text(
        """
section = settings.evaluation
direct = settings.consensus.models
alias = section.stage1_enabled
dynamic = getattr(section, "stage2_enabled")
settings.evaluation.stage3_enabled = False
text = "settings.evaluation.satisfaction_threshold"

def local_alias(config):
    evaluation = config.evaluation
    return evaluation.uncertainty_threshold

def shadow(section):
    return section.satisfaction_threshold

section = object()
ignored_after_rebind = section.semantic_model
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "satisfaction_threshold"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("evaluation", "semantic_model"),
            contract.ConfigField("consensus", "models"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "models"),
        }
    )


def test_runtime_scan_handles_natural_roots_without_unrelated_object_false_positives(
    contract, tmp_path: Path
) -> None:
    source = tmp_path / "natural_roots.py"
    source.write_text(
        """
factory_read = get_config().evaluation.stage1_enabled
indexed_read = configs[key].evaluation.stage2_enabled
method_read = self.config.consensus.models
loaded = get_config()
aliased_root_read = loaded.evaluation.stage3_enabled

unrelated_attribute = report.evaluation.satisfaction_threshold
unrelated_call = build_report().evaluation.satisfaction_threshold
unrelated_subscript = reports[key].evaluation.uncertainty_threshold
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "satisfaction_threshold"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "models"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("consensus", "models"),
        }
    )


def test_runtime_scan_joins_conditional_alias_fallbacks_but_honors_reassignment(
    contract, tmp_path: Path
) -> None:
    source = tmp_path / "conditional_aliases.py"
    source.write_text(
        """
def branch_fallback(config, override, enabled):
    section = config.evaluation
    if enabled:
        section = override
    return section.stage1_enabled

def expression_fallback(config, override, enabled):
    section = override if enabled else config.evaluation
    alternate = override or config.evaluation
    return section.stage2_enabled, alternate.stage3_enabled

def unconditional_reassignment(config):
    section = config.evaluation
    section = object()
    return section.satisfaction_threshold
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "satisfaction_threshold"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
        }
    )


def test_every_schema_field_needs_exactly_one_disposition(contract) -> None:
    active = contract.ConfigField("evaluation", "active")
    inert = contract.ConfigField("evaluation", "inert")
    schema_only = contract.ConfigField("evaluation", "schema_only")
    fields = frozenset({active, inert, schema_only})
    rows = {
        active: contract.ReferenceRow("true", "Active runtime control."),
        inert: contract.ReferenceRow(
            "true", "Currently inert. Effective control: RuntimeConfig.inert."
        ),
        schema_only: contract.ReferenceRow("true", "Compatibility-only schema value."),
    }
    markers = {
        inert: contract.InertMarker(inert, "RuntimeConfig.inert"),
    }

    report = contract.audit_contract(
        fields=fields,
        reads=frozenset({active}),
        rows=rows,
        markers=markers,
        allowlist={schema_only: "Retained to read configuration written by release 0.1.0."},
        documented_defaults={},
    )

    assert report.violations == ()


def test_new_unwired_field_fails_with_actionable_name(contract) -> None:
    field = contract.ConfigField("evaluation", "new_knob")

    report = contract.audit_contract(
        fields=frozenset({field}),
        reads=frozenset(),
        rows={field: contract.ReferenceRow("true", "A new control.")},
        markers={},
        allowlist={},
        documented_defaults={},
    )

    assert report.violations == (
        "evaluation.new_knob: no production read, inert documentation, or schema-only rationale",
    )


def test_wired_field_cannot_remain_documented_inert(contract) -> None:
    field = contract.ConfigField("consensus", "threshold")

    report = contract.audit_contract(
        fields=frozenset({field}),
        reads=frozenset({field}),
        rows={
            field: contract.ReferenceRow(
                "0.67", "Currently inert. Effective control: majority_threshold."
            )
        },
        markers={field: contract.InertMarker(field, "majority_threshold")},
        allowlist={},
        documented_defaults={},
    )

    assert "consensus.threshold: conflicting config-field dispositions" in report.violations
    assert (
        "consensus.threshold: production-wired field is still documented inert" in report.violations
    )


def test_visible_inert_claim_requires_structured_effective_control(contract) -> None:
    field = contract.ConfigField("consensus", "min_models")

    report = contract.audit_contract(
        fields=frozenset({field}),
        reads=frozenset(),
        rows={field: contract.ReferenceRow("3", "Currently inert.")},
        markers={field: contract.InertMarker(field, "two successful votes")},
        allowlist={},
        documented_defaults={},
    )

    assert "consensus.min_models: inert docs do not label the effective control" in (
        report.violations
    )
    assert any("visible docs omit marker effective control" in item for item in report.violations)


def test_reference_parser_uses_sections_and_structured_json_markers(contract) -> None:
    text = """
## `evaluation`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `stage1_enabled` | `bool` | `true` | **Currently inert. Effective control:** `PipelineConfig.stage1_enabled`. |

<!-- config-field-contract: {"section":"evaluation","field":"stage1_enabled","status":"inert","effective_control":"PipelineConfig.stage1_enabled"} -->
"""
    field = contract.ConfigField("evaluation", "stage1_enabled")

    assert contract.parse_reference_rows(text)[field].default == "`true`"
    assert contract.parse_inert_markers(text)[field].effective_control == (
        "PipelineConfig.stage1_enabled"
    )


def test_opus_defaults_preserve_direct_and_openrouter_formats(contract) -> None:
    assert (
        contract.opus_default_violations("claude-opus-4-8", "openrouter/anthropic/claude-opus-4.8")
        == ()
    )

    violations = contract.opus_default_violations("claude-opus-4-8", "claude-opus-4-8")

    assert violations == (
        "Opus defaults must use distinct direct and OpenRouter identifiers",
        "consensus.advocate_model: invalid OpenRouter Opus default 'claude-opus-4-8'; "
        "expected 'openrouter/anthropic/claude-opus-<major>.<minor>'",
    )


@pytest.mark.parametrize(
    ("direct_model", "consensus_model", "expected_fragments"),
    [
        (
            "claude-opus-4.8",
            "claude-opus-4.8",
            ("distinct", "invalid direct", "invalid OpenRouter"),
        ),
        (
            "claude-opus-",
            "claude-opus-",
            ("distinct", "invalid direct", "invalid OpenRouter"),
        ),
        (
            "claude-opus-4-8",
            "claude-opus-4-8",
            ("distinct", "invalid OpenRouter"),
        ),
        (
            "claude-opus-4-8",
            "openrouter/anthropic/claude-opus-4-8",
            ("invalid OpenRouter",),
        ),
        (
            "claude-opus-4-8-extra",
            "openrouter/anthropic/claude-opus-4.8",
            ("invalid direct",),
        ),
    ],
)
def test_opus_defaults_reject_collapsed_or_malformed_formats(
    contract,
    direct_model: str,
    consensus_model: str,
    expected_fragments: tuple[str, ...],
) -> None:
    violations = contract.opus_default_violations(direct_model, consensus_model)

    assert all(
        any(fragment in violation for violation in violations) for fragment in expected_fragments
    )


def test_opus_defaults_reject_mismatched_valid_versions(contract) -> None:
    violations = contract.opus_default_violations(
        "claude-opus-4-8", "openrouter/anthropic/claude-opus-4.9"
    )

    assert violations == (
        "consensus.advocate_model: OpenRouter Opus default "
        "'openrouter/anthropic/claude-opus-4.9' does not correspond to direct default "
        "'claude-opus-4-8'; expected 'openrouter/anthropic/claude-opus-4.8'",
    )


def test_documented_default_must_match_its_ssot_value(contract) -> None:
    field = contract.ConfigField("evaluation", "semantic_model")

    report = contract.audit_contract(
        fields=frozenset({field}),
        reads=frozenset({field}),
        rows={field: contract.ReferenceRow('"claude-opus-4-6"', "Wired model.")},
        markers={},
        allowlist={},
        documented_defaults={field: "claude-opus-4-8"},
    )

    assert report.violations == (
        "evaluation.semantic_model: documented default 'claude-opus-4-6' "
        "does not match 'claude-opus-4-8'",
    )
