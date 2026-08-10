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


def test_runtime_scan_propagates_mutable_mapping_identity_across_aliases(
    contract, tmp_path: Path
) -> None:
    (tmp_path / "mutable_mapping_aliases.py").write_text(
        """
def dynamic_setdefault(config, key):
    sections = {}
    sections.setdefault(key, config.evaluation)
    for section in sections.values():
        dynamic_read = section.stage1_enabled

def aliased_setdefault(config):
    sections = {}
    alias = sections
    alias.setdefault("x", config.evaluation)
    alias_read = sections["x"].stage2_enabled

def augmented_union(config):
    sections = {}
    alias = sections
    sections |= {"x": config.evaluation}
    augmented_read = alias["x"].stage3_enabled

def constructors(config):
    source = {"x": config.evaluation}
    expanded = dict(**source)
    expanded_read = expanded["x"].assertion_extraction_model
    pairs = dict([("x", config.evaluation)])
    pairs_read = pairs["x"].uncertainty_threshold
    keywords = dict(x=config.evaluation)
    keyword_read = keywords["x"].semantic_model

def direct_mutations(config):
    sections = {}
    alias = sections
    alias["x"] = config.evaluation
    assigned_read = sections["x"].stage1_enabled

    consensus = {}
    consensus_alias = consensus
    consensus_alias.update({"x": config.consensus})
    updated_read = consensus["x"].models

    pair_updates = {}
    pair_alias = pair_updates
    pair_alias.update([("x", config.consensus)])
    pair_update_read = pair_updates["x"].devil_model

def conditional_alias(config, enabled):
    left = {}
    right = {}
    alias = left if enabled else right
    alias.setdefault("x", config.consensus)
    possible_left_read = left.get("x").advocate_model

def dynamic_assignment(config, key):
    sections = {}
    alias = sections
    alias[key] = config.evaluation
    for section in sections.values():
        dynamic_assignment_read = section.stage3_enabled

def provenance_kills(config):
    cleared = {"x": config.evaluation}
    cleared_alias = cleared
    cleared_alias.clear()
    cleared_read = cleared.get("x").satisfaction_threshold

    popped = {"x": config.evaluation}
    popped.pop("x")
    popped_read = popped.get("x").satisfaction_threshold

def unrelated_mutations(report, key):
    sections = {}
    alias = sections
    alias.setdefault(key, report.evaluation)
    alias["x"] = report.evaluation
    alias |= {"y": report.evaluation}
    alias.update({"z": report.evaluation})
    for section in sections.values():
        unrelated_read = section.satisfaction_threshold

    source = {"x": report.evaluation}
    expanded = dict(**source)
    pairs = dict([("x", report.evaluation)])
    expanded_read = expanded["x"].satisfaction_threshold
    pairs_read = pairs["x"].satisfaction_threshold
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
            contract.ConfigField("consensus", "advocate_model"),
            contract.ConfigField("consensus", "devil_model"),
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
            contract.ConfigField("consensus", "models"),
            contract.ConfigField("consensus", "advocate_model"),
            contract.ConfigField("consensus", "devil_model"),
        }
    )


def test_runtime_scan_models_dynamic_key_collisions_in_evaluation_order(
    contract, tmp_path: Path
) -> None:
    (tmp_path / "dynamic_key_collisions.py").write_text(
        """
def dynamic_key_collisions(config, report, key):
    preserved = {"known": report.evaluation}
    preserved.setdefault(key, config.evaluation)
    preserved_known_read = preserved["known"].satisfaction_threshold
    for section in preserved.values():
        inserted_default_read = section.stage1_enabled

    later_dynamic = {
        "known": report.evaluation,
        key: config.evaluation,
    }
    later_dynamic_read = later_dynamic["known"].stage2_enabled

    later_literal = {
        key: config.evaluation,
        "known": report.evaluation,
    }
    later_literal_read = later_literal["known"].satisfaction_threshold

    updated = {"known": report.evaluation}
    updated.update({key: config.evaluation})
    updated_read = updated["known"].stage3_enabled

    unioned = {"known": report.evaluation} | {key: config.evaluation}
    unioned_read = unioned["known"].uncertainty_threshold

    assigned = {"known": report.evaluation}
    assigned[key] = config.evaluation
    assigned_read = assigned["known"].assertion_extraction_model

    missing = {"other": config.evaluation}
    unreachable_missing_read = missing["known"].semantic_model
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
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "assertion_extraction_model"),
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
        }
    )


def test_runtime_scan_joins_dynamic_mapping_keys_and_missing_defaults(
    contract, tmp_path: Path
) -> None:
    field = contract.ConfigField("evaluation", "stage1_enabled")
    fields = frozenset({field})

    positive = tmp_path / "positive"
    positive.mkdir()
    (positive / "conditional_mapping.py").write_text(
        """
def conditional_dynamic(config, report, key, enabled):
    sections = {key: config.evaluation} if enabled else {"x": report.evaluation}
    return sections["x"].stage1_enabled

def conditional_default(config, report, enabled):
    sections = {} if enabled else {"x": report.evaluation}
    return sections.get("x", config.evaluation).stage1_enabled
""",
        encoding="utf-8",
    )
    assert contract.runtime_reads(positive, fields) == fields

    negative = tmp_path / "negative"
    negative.mkdir()
    (negative / "missing_or_unrelated.py").write_text(
        """
def conditional_unrelated(report, enabled):
    sections = {} if enabled else {"x": report.evaluation}
    return sections["x"].stage1_enabled
""",
        encoding="utf-8",
    )
    assert contract.runtime_reads(negative, fields) == frozenset()


@pytest.mark.parametrize(
    ("mutation", "receiver_read"),
    [
        (
            'receiver["known"] = report.evaluation',
            'receiver["known"].satisfaction_threshold',
        ),
        (
            'receiver.update({"known": report.evaluation})',
            'receiver["known"].satisfaction_threshold',
        ),
        (
            'receiver |= {"known": report.evaluation}',
            'receiver["known"].satisfaction_threshold',
        ),
        (
            "receiver.clear()",
            'receiver.get("known", report.evaluation).satisfaction_threshold',
        ),
        (
            'receiver.pop("known")',
            'receiver.get("known", report.evaluation).satisfaction_threshold',
        ),
    ],
    ids=["assignment", "update", "union", "clear", "pop"],
)
def test_runtime_scan_strongly_mutates_conditional_receiver_but_not_possible_aliases(
    contract,
    tmp_path: Path,
    mutation: str,
    receiver_read: str,
) -> None:
    (tmp_path / "conditional_receiver.py").write_text(
        f"""
def conditional_receiver(config, report, enabled):
    left = {{"known": config.evaluation}}
    right = {{"known": report.evaluation}}
    receiver = left if enabled else right
    {mutation}
    impossible_receiver_read = {receiver_read}
    possible_alias_read = left.get("known", report.evaluation).stage1_enabled
""",
        encoding="utf-8",
    )
    positive_alias = contract.ConfigField("evaluation", "stage1_enabled")
    false_positive = contract.ConfigField("evaluation", "satisfaction_threshold")

    assert contract.runtime_reads(
        tmp_path,
        frozenset({positive_alias, false_positive}),
    ) == frozenset({positive_alias})


def test_runtime_scan_keeps_provenance_when_mutating_a_may_alias(contract, tmp_path: Path) -> None:
    (tmp_path / "may_alias_mutation.py").write_text(
        """
def may_alias_clear(config, report, enabled):
    left = {"x": config.evaluation}
    right = {"x": report.evaluation}
    alias = left if enabled else right
    alias.clear()
    possible_uncleared_read = left.get("x").stage1_enabled

def independent_report_maps(report):
    left = {"x": report.evaluation}
    right = {"x": report.evaluation}
    alias = left
    alias.clear()
    unrelated_read = right.get("x").satisfaction_threshold

def popped_value(config):
    values = {"x": config.evaluation}
    popped = values.pop("x")
    popped_read = popped.stage2_enabled
    removed_read = values.get("x").satisfaction_threshold
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "satisfaction_threshold"),
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
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


def test_runtime_scan_excludes_static_dead_and_type_only_branches_without_pruning_dynamic(
    contract, tmp_path: Path
) -> None:
    (tmp_path / "reachable_branches.py").write_text(
        """
from typing import TYPE_CHECKING as CHECKING
import typing as typing_alias

if CHECKING:
    type_only_evaluation = config.evaluation.stage1_enabled
if typing_alias.TYPE_CHECKING:
    type_only_consensus = config.consensus.min_models
if False:
    dead_literal = config.evaluation.stage2_enabled
if True:
    live_literal = config.evaluation.stage3_enabled
else:
    dead_else = config.consensus.threshold
if runtime_flag:
    dynamic_body = config.evaluation.uncertainty_threshold
else:
    dynamic_else = config.consensus.models
maybe_type_checking = CHECKING if runtime_flag else another_runtime_flag
if maybe_type_checking:
    dynamic_type_checking_collision = config.consensus.diversity_required

dead_short_circuit = False and config.evaluation.satisfaction_threshold
live_short_circuit = False or config.consensus.advocate_model
dead_expression = (
    config.consensus.devil_model if False else config.evaluation.semantic_model
)
while False:
    dead_loop = config.consensus.judge_model
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "satisfaction_threshold"),
            contract.ConfigField("evaluation", "semantic_model"),
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "advocate_model"),
            contract.ConfigField("consensus", "devil_model"),
            contract.ConfigField("consensus", "diversity_required"),
            contract.ConfigField("consensus", "judge_model"),
            contract.ConfigField("consensus", "min_models"),
            contract.ConfigField("consensus", "models"),
            contract.ConfigField("consensus", "threshold"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "semantic_model"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "advocate_model"),
            contract.ConfigField("consensus", "diversity_required"),
            contract.ConfigField("consensus", "models"),
        }
    )


def test_runtime_scan_tracks_section_annotations_and_local_call_arguments_without_collisions(
    contract, tmp_path: Path
) -> None:
    (tmp_path / "section_helpers.py").write_text(
        """
from typing import TYPE_CHECKING

from ouroboros.config.models import ConsensusConfig, EvaluationConfig
from ouroboros.evaluation.consensus import ConsensusConfig as RuntimeConsensusConfig

EvaluationAlias = EvaluationConfig

if TYPE_CHECKING:
    from ouroboros.config.models import EvaluationConfig as EvaluationSection

def typed_evaluation(section: EvaluationConfig):
    return section.stage1_enabled

def typed_consensus(section: ConsensusConfig | None):
    return section.models

def type_only_alias(section: EvaluationSection):
    return section.stage2_enabled

def assigned_alias(section: EvaluationAlias):
    return section.assertion_extraction_model

def untyped_helper(section):
    return section.stage3_enabled

def identity(section):
    return section

def caller(config):
    untyped_helper(config.evaluation)
    untyped_helper(**{"section": config.evaluation})
    return identity(config.consensus).judge_model

def colliding_runtime_type(section: RuntimeConsensusConfig):
    return section.min_models

def arbitrary_config_annotation(section: ProjectConfig):
    return section.satisfaction_threshold

def wrapped_section_is_not_the_section(sections: list[EvaluationConfig]):
    return sections.uncertainty_threshold
""",
        encoding="utf-8",
    )
    fields = frozenset(
        {
            contract.ConfigField("evaluation", "satisfaction_threshold"),
            contract.ConfigField("evaluation", "assertion_extraction_model"),
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("evaluation", "uncertainty_threshold"),
            contract.ConfigField("consensus", "judge_model"),
            contract.ConfigField("consensus", "min_models"),
            contract.ConfigField("consensus", "models"),
        }
    )

    assert contract.runtime_reads(tmp_path, fields) == frozenset(
        {
            contract.ConfigField("evaluation", "assertion_extraction_model"),
            contract.ConfigField("evaluation", "stage1_enabled"),
            contract.ConfigField("evaluation", "stage2_enabled"),
            contract.ConfigField("evaluation", "stage3_enabled"),
            contract.ConfigField("consensus", "judge_model"),
            contract.ConfigField("consensus", "models"),
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
