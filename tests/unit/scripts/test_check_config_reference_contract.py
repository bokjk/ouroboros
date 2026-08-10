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

class Runtime:
    def read(self):
        return self.get_config().evaluation.semantic_model

unrelated_attribute = report.evaluation.satisfaction_threshold
unrelated_call = build_report().evaluation.satisfaction_threshold
unrelated_subscript = reports[key].evaluation.uncertainty_threshold
unrelated_factory_method = report.get_config().consensus.threshold
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "satisfaction_threshold"),
            contract.ConfigField("evaluation", "semantic_model"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "models"),
            contract.ConfigField("consensus", "threshold"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "semantic_model"),
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


def test_runtime_scan_preserves_zero_iteration_and_loop_body_provenance(
    contract, tmp_path: Path
) -> None:
    (tmp_path / "loops.py").write_text(
        """
def loop_paths(config, report, enabled):
    section = config.evaluation
    for section in []:
        pass
    zero_iteration_read = section.stage1_enabled

    section = report.evaluation
    while enabled:
        section = config.evaluation
    possible_iteration_read = section.stage2_enabled

    for candidate, ignored in (
        (report.evaluation, 0),
        (config.evaluation, 1),
    ):
        loop_destructured_read = candidate.stage3_enabled

    unrelated = report.config.evaluation.satisfaction_threshold
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


def test_runtime_scan_joins_successful_try_handler_and_finally_paths(
    contract, tmp_path: Path
) -> None:
    (tmp_path / "try_paths.py").write_text(
        """
def try_paths(config, report, risky):
    successful = config.evaluation
    try:
        risky()
    except ValueError:
        successful = report.evaluation
    successful_path_read = successful.stage1_enabled

    handler_source = report.evaluation
    try:
        handler_source = config.evaluation
        risky()
        handler_source = report.evaluation
    except RuntimeError:
        handler_read = handler_source.stage2_enabled

    final_source = report.consensus
    try:
        final_source = config.consensus
        risky()
    except LookupError:
        final_source = report.consensus
    finally:
        final_read = final_source.models

    unrelated = report.settings.consensus.threshold
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("consensus", "models"),
            contract.ConfigField("consensus", "threshold"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("consensus", "models"),
        }
    )


def test_runtime_scan_joins_match_branches_and_destructured_aliases(
    contract, tmp_path: Path
) -> None:
    (tmp_path / "match_and_destructure.py").write_text(
        """
def match_paths(config, report, selector):
    evaluation_alias, other = config.evaluation, report.evaluation
    assigned_evaluation_read = evaluation_alias.uncertainty_threshold

    [list_alias] = [config.evaluation]
    assigned_list_read = list_alias.satisfaction_threshold

    consensus_alias, ignored_alias = config.consensus, report.evaluation
    assigned_consensus_read = consensus_alias.devil_model

    section = report.evaluation
    match selector:
        case 0:
            section = config.evaluation
        case _:
            section = report.evaluation
    branch_read = section.stage1_enabled

    match (config.evaluation, (config.consensus, report.evaluation)):
        case (evaluation, (consensus, ignored)):
            destructured_evaluation_read = evaluation.stage2_enabled
            destructured_consensus_read = consensus.models

    unrelated = report.config.consensus.threshold
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "satisfaction_threshold"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "devil_model"),
            contract.ConfigField("consensus", "models"),
            contract.ConfigField("consensus", "threshold"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "satisfaction_threshold"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "devil_model"),
            contract.ConfigField("consensus", "models"),
        }
    )


def test_runtime_scan_preserves_mapping_class_and_rest_pattern_provenance(
    contract, tmp_path: Path
) -> None:
    (tmp_path / "structural_patterns.py").write_text(
        """
def mapping_pattern(config, report):
    match {"section": config.evaluation}:
        case {"section": direct_section}:
            direct_mapping_read = direct_section.stage1_enabled

    mapping_source = {"section": config.evaluation}
    match {**mapping_source}:
        case {"section": expanded_section}:
            expanded_mapping_read = expanded_section.assertion_extraction_model

    payload = {
        "section": config.evaluation,
        "remaining": config.consensus,
        "unrelated": report.evaluation,
    }
    match payload:
        case {"section": section, "unrelated": unrelated, **rest}:
            mapping_read = section.stage2_enabled
            rest_read = rest["remaining"].models
            unrelated_read = unrelated.satisfaction_threshold

class Envelope:
    __match_args__ = ("positional",)

def class_pattern(config, report):
    match Envelope(section=config.evaluation):
        case Envelope(section=direct_section):
            direct_keyword_read = direct_section.stage3_enabled

    keyword_source = {"section": config.consensus}
    match Envelope(**keyword_source):
        case Envelope(section=expanded_section):
            expanded_keyword_read = expanded_section.judge_model

    payload = Envelope(
        config.consensus,
        section=config.evaluation,
        unrelated=report.consensus,
    )
    match payload:
        case Envelope(positional, section=section, unrelated=unrelated):
            positional_read = positional.devil_model
            keyword_read = section.uncertainty_threshold
            unrelated_read = unrelated.threshold
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "assertion_extraction_model"),
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "satisfaction_threshold"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "devil_model"),
            contract.ConfigField("consensus", "judge_model"),
            contract.ConfigField("consensus", "models"),
            contract.ConfigField("consensus", "threshold"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "assertion_extraction_model"),
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "devil_model"),
            contract.ConfigField("consensus", "judge_model"),
            contract.ConfigField("consensus", "models"),
        }
    )


def test_runtime_scan_tracks_indirect_loop_containers_and_sequence_star_patterns(
    contract, tmp_path: Path
) -> None:
    (tmp_path / "indirect_containers.py").write_text(
        """
def indirect_loop(config, report):
    sections = [config.evaluation]
    aliases = sections
    for section in aliases:
        loop_read = section.stage1_enabled

    pairs = [(config.evaluation, report.evaluation)]
    for section, unrelated in pairs:
        destructured_loop_read = section.stage2_enabled
        unrelated_read = unrelated.satisfaction_threshold

    match [config.consensus, report.consensus]:
        case [consensus, *rest]:
            head_read = consensus.models
            unrelated_star_read = rest[0].threshold

async def indirect_async_loop(config):
    sections = [config.consensus]
    async for section in sections:
        async_loop_read = section.devil_model
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "satisfaction_threshold"),
            contract.ConfigField("consensus", "devil_model"),
            contract.ConfigField("consensus", "models"),
            contract.ConfigField("consensus", "threshold"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("consensus", "devil_model"),
            contract.ConfigField("consensus", "models"),
        }
    )


def test_runtime_scan_models_standard_dict_consumers_key_sensitively(
    contract, tmp_path: Path
) -> None:
    (tmp_path / "dict_consumers.py").write_text(
        """
def dict_consumers(config, report, dynamic_key):
    sections = {
        "primary": config.evaluation,
        "unrelated": report.evaluation,
    }
    for section in sections.values():
        values_read = section.stage1_enabled
    for _, section in sections.items():
        items_read = section.stage2_enabled

    primary_read = sections.get("primary").stage3_enabled
    primary_with_default = sections.get(
        "primary", report.evaluation
    ).uncertainty_threshold
    dynamic_read = sections.get(dynamic_key, report.evaluation).assertion_extraction_model
    missing_read = sections.get("missing").satisfaction_threshold
    missing_default = sections.get("missing", config.consensus).models

    for key in sections.keys():
        non_config_key_read = key.satisfaction_threshold

    reports = {"primary": report.evaluation}
    report_get = reports.get("primary").satisfaction_threshold
    report_dynamic = reports.get(dynamic_key, report.evaluation).satisfaction_threshold
    for report_section in reports.values():
        report_value_read = report_section.satisfaction_threshold

    literals = {"primary": "not config"}
    literal_read = literals.get("primary").satisfaction_threshold

    copied = sections.copy()
    copied_read = copied.get("primary").semantic_model
    constructed = dict(sections)
    constructed_read = constructed["primary"].stage1_enabled

    unioned = reports | {"primary": config.consensus}
    union_read = unioned.get("primary").devil_model
    overridden = sections | {
        "primary": report.evaluation,
        "unrelated": report.evaluation,
    }
    overridden_read = overridden.get("primary").satisfaction_threshold

    defaults = {}
    defaults.setdefault("primary", config.consensus)
    inserted_read = defaults.get("primary").judge_model
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "assertion_extraction_model"),
            contract.ConfigField("evaluation", "satisfaction_threshold"),
            contract.ConfigField("evaluation", "semantic_model"),
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "devil_model"),
            contract.ConfigField("consensus", "judge_model"),
            contract.ConfigField("consensus", "models"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "assertion_extraction_model"),
            contract.ConfigField("evaluation", "semantic_model"),
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "devil_model"),
            contract.ConfigField("consensus", "judge_model"),
            contract.ConfigField("consensus", "models"),
        }
    )


def test_runtime_scan_comprehension_targets_shadow_outer_aliases_in_evaluation_order(
    contract, tmp_path: Path
) -> None:
    (tmp_path / "comprehensions.py").write_text(
        """
def comprehension_shadowing(config, reports):
    section = config.evaluation
    ignored_list = [section.stage1_enabled for section in reports]
    ignored_set = {section.stage2_enabled for section in reports}
    ignored_dict = {section.stage3_enabled: section for section in reports}
    ignored_generator = (section.satisfaction_threshold for section in reports)
    outer_read = section.semantic_model

    sections = [config.evaluation]
    positive_list = [item.uncertainty_threshold for item in sections]
    nested = [
        item.assertion_extraction_model
        for group in [[config.evaluation]]
        for item in group
    ]

async def async_comprehension_shadowing(config, reports):
    section = config.consensus
    ignored = [section.threshold async for section in reports]
    outer_read = section.models
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "assertion_extraction_model"),
            contract.ConfigField("evaluation", "satisfaction_threshold"),
            contract.ConfigField("evaluation", "semantic_model"),
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "models"),
            contract.ConfigField("consensus", "threshold"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "assertion_extraction_model"),
            contract.ConfigField("evaluation", "semantic_model"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "models"),
        }
    )


def test_runtime_scan_import_and_context_manager_bindings_override_config_inference(
    contract, tmp_path: Path
) -> None:
    (tmp_path / "binding_overrides.py").write_text(
        """
def imported_names():
    import report as config
    from report import settings
    imported_config = config.evaluation.stage1_enabled
    imported_settings = settings.consensus.models

def context_names(manager):
    with manager as config:
        context_config = config.evaluation.stage2_enabled
    after_context = config.evaluation.stage3_enabled

async def async_context_names(manager):
    async with manager as settings:
        context_settings = settings.consensus.threshold
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("consensus", "models"),
            contract.ConfigField("consensus", "threshold"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset()


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
