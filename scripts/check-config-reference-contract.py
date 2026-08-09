#!/usr/bin/env python3
"""Keep evaluation/consensus schema, runtime wiring, and docs in agreement.

Every field in the user-facing ``EvaluationConfig`` and ``ConsensusConfig``
schema must have exactly one truthful disposition:

* production code reads it from the corresponding config section;
* ``docs/config-reference.md`` marks it as inert and names the effective
  control; or
* it is present in the explicit, justified schema-only allowlist below.

The scan is syntax-aware. It inspects Python attribute loads rather than source
text, and it reads structured Markdown table rows plus JSON contract markers.
That lets the same check catch both directions of drift: a new unwired field,
and a previously inert field that becomes wired while the reference still says
it does nothing.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ouroboros.config._model_defaults import (  # noqa: E402
    DEFAULT_CONSENSUS_OPUS_MODEL,
    DEFAULT_OPUS_MODEL,
)
from ouroboros.config.models import ConsensusConfig, EvaluationConfig  # noqa: E402


@dataclass(frozen=True, order=True)
class ConfigField:
    """One field in a named user-facing config section."""

    section: str
    name: str

    @property
    def dotted(self) -> str:
        return f"{self.section}.{self.name}"


@dataclass(frozen=True)
class ReferenceRow:
    """Relevant cells from one config-reference field row."""

    default: str
    description: str


@dataclass(frozen=True)
class InertMarker:
    """Machine-readable companion to one visible inert-field description."""

    field: ConfigField
    effective_control: str


@dataclass(frozen=True)
class ContractReport:
    """All violations found in one complete contract audit."""

    violations: tuple[str, ...]
    runtime_reads: frozenset[ConfigField]


TRACKED_SECTIONS = frozenset({"evaluation", "consensus"})
REFERENCE_PATH = Path("docs/config-reference.md")
_SECTION_HEADING = re.compile(r"^## `(?P<section>evaluation|consensus)`\s*$")
_FIELD_ROW = re.compile(r"^\|\s*`(?P<field>[a-z][a-z0-9_]*)`\s*\|")
_INERT_MARKER = re.compile(r"<!--\s*config-field-contract:\s*(?P<payload>\{[^\r\n]*\})\s*-->")

# Escape hatch for a deliberately schema-only field that should not be called
# inert in the public reference. Keep this small. Every entry must carry a
# concrete rationale; the audit rejects stale entries once production reads the
# field. The eight fields motivating #1998 are documented, so none belong here.
SCHEMA_ONLY_ALLOWLIST: Mapping[ConfigField, str] = {}

# Bounded guard for the two Opus identifiers that intentionally use different
# normalized forms. This catches a blanket replacement without turning the
# checker into a general-purpose documentation-value synchronizer.
DOCUMENTED_DEFAULTS: Mapping[ConfigField, str] = {
    ConfigField("evaluation", "semantic_model"): DEFAULT_OPUS_MODEL,
    ConfigField("consensus", "advocate_model"): DEFAULT_CONSENSUS_OPUS_MODEL,
}


def schema_fields() -> frozenset[ConfigField]:
    """Enumerate the authoritative Pydantic fields for the two sections."""

    return frozenset(
        ConfigField(section, name)
        for section, model in (
            ("evaluation", EvaluationConfig),
            ("consensus", ConsensusConfig),
        )
        for name in model.model_fields
    )


_CONFIG_ROOT = "<config-root>"
_UNKNOWN_STATE: frozenset[str] = frozenset()
_CONFIG_NAME = re.compile(
    r"(?:^|_)(?:cfg|config|configs|configuration|settings)$",
    flags=re.IGNORECASE,
)
_CONFIG_FACTORY = re.compile(
    r"(?:^|_)(?:build|create|get|load|read|resolve)_(?:config|configuration|settings)$",
    flags=re.IGNORECASE,
)


def _looks_like_config_name(name: str) -> bool:
    return _CONFIG_NAME.search(name) is not None


def _callable_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _annotation_names_config(annotation: ast.AST | None) -> bool:
    if annotation is None:
        return False
    return any(
        (
            isinstance(node, ast.Name)
            and ("config" in node.id.lower() or "settings" in node.id.lower())
        )
        or (
            isinstance(node, ast.Attribute)
            and ("config" in node.attr.lower() or "settings" in node.attr.lower())
        )
        or (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and ("config" in node.value.lower() or "settings" in node.value.lower())
        )
        for node in ast.walk(annotation)
    )


class _RuntimeReadVisitor(ast.NodeVisitor):
    """Collect reads proven to originate from application configuration.

    Each scope tracks abstract provenance rather than matching arbitrary
    ``*.evaluation.field`` text. A value can be a config root, one or more
    tracked section aliases, or unknown. Conditional control flow joins the
    possible states, while an unconditional reassignment replaces them.
    """

    def __init__(self, fields: frozenset[ConfigField]) -> None:
        self._fields = fields
        # An explicit empty state is a deliberate shadow. Without it, a
        # function parameter or local reassignment could inherit an outer alias.
        self._states: list[dict[str, frozenset[str]]] = [{}]
        self.reads: set[ConfigField] = set()

    def _name_state(self, name: str) -> frozenset[str]:
        for scope in reversed(self._states):
            if name in scope:
                return scope[name]
        if _looks_like_config_name(name):
            return frozenset({_CONFIG_ROOT})
        return _UNKNOWN_STATE

    def _expression_state(self, node: ast.AST) -> frozenset[str]:
        if isinstance(node, ast.Name):
            return self._name_state(node.id)
        if isinstance(node, ast.Attribute):
            owner_state = self._expression_state(node.value)
            if node.attr in TRACKED_SECTIONS and _CONFIG_ROOT in owner_state:
                return frozenset({node.attr})
            if _looks_like_config_name(node.attr):
                return frozenset({_CONFIG_ROOT})
            return _UNKNOWN_STATE
        if isinstance(node, ast.Call):
            callable_name = _callable_name(node.func)
            if callable_name is not None and _CONFIG_FACTORY.search(callable_name):
                return frozenset({_CONFIG_ROOT})
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in TRACKED_SECTIONS
                and _CONFIG_ROOT in self._expression_state(node.args[0])
            ):
                return frozenset({node.args[1].value})
            return _UNKNOWN_STATE
        if isinstance(node, ast.Subscript):
            state = self._expression_state(node.value)
            if _CONFIG_ROOT in state:
                return frozenset({_CONFIG_ROOT})
            return _UNKNOWN_STATE
        if isinstance(node, ast.IfExp):
            return self._expression_state(node.body) | self._expression_state(node.orelse)
        if isinstance(node, ast.BoolOp):
            state = _UNKNOWN_STATE
            for value in node.values:
                state |= self._expression_state(value)
            return state
        if isinstance(node, ast.NamedExpr):
            return self._expression_state(node.value)
        return _UNKNOWN_STATE

    def _record(self, section: str, name: str) -> None:
        field = ConfigField(section, name)
        if field in self._fields:
            self.reads.add(field)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            for section in self._expression_state(node.value) & TRACKED_SECTIONS:
                self._record(section, node.attr)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # ``getattr(config.evaluation, "field")`` is still an explicit read.
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            for section in self._expression_state(node.args[0]) & TRACKED_SECTIONS:
                self._record(section, node.args[1].value)
        self.generic_visit(node)

    def _bind_aliases(self, targets: Iterable[ast.expr], value: ast.AST) -> None:
        state = self._expression_state(value)
        for target in targets:
            if isinstance(target, ast.Name):
                self._states[-1][target.id] = state

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        self._bind_aliases(node.targets, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            self._bind_aliases((node.target,), node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind_aliases((node.target,), node.value)

    def _visit_branch(
        self,
        statements: Iterable[ast.stmt],
        initial: dict[str, frozenset[str]],
    ) -> dict[str, frozenset[str]]:
        self._states[-1] = dict(initial)
        for statement in statements:
            self.visit(statement)
        return dict(self._states[-1])

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        initial = dict(self._states[-1])
        body_state = self._visit_branch(node.body, initial)
        else_state = self._visit_branch(node.orelse, initial) if node.orelse else initial
        self._states[-1] = {
            name: body_state.get(name, _UNKNOWN_STATE) | else_state.get(name, _UNKNOWN_STATE)
            for name in body_state.keys() | else_state.keys()
        }

    @staticmethod
    def _argument_state(argument: ast.arg) -> frozenset[str]:
        if _looks_like_config_name(argument.arg) or _annotation_names_config(argument.annotation):
            return frozenset({_CONFIG_ROOT})
        return _UNKNOWN_STATE

    def _scoped_arguments(self, arguments: ast.arguments) -> dict[str, frozenset[str]]:
        scoped = {
            argument.arg: self._argument_state(argument)
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        }
        if arguments.vararg is not None:
            scoped[arguments.vararg.arg] = self._argument_state(arguments.vararg)
        if arguments.kwarg is not None:
            scoped[arguments.kwarg.arg] = self._argument_state(arguments.kwarg)
        return scoped

    def _visit_scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Defaults and decorators execute in the containing scope.
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

        self._states.append(self._scoped_arguments(node.args))
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._states.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        self._states.append(self._scoped_arguments(node.args))
        try:
            self.visit(node.body)
        finally:
            self._states.pop()


def runtime_reads(source_root: Path, fields: frozenset[ConfigField]) -> frozenset[ConfigField]:
    """Return config fields loaded from their named sections in production Python."""

    reads: set[ConfigField] = set()
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _RuntimeReadVisitor(fields)
        visitor.visit(tree)
        reads.update(visitor.reads)
    return frozenset(reads)


def _split_markdown_row(line: str) -> tuple[str, ...]:
    """Split a Markdown table row without treating escaped pipes as separators."""

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in line.strip().strip("|"):
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            current.append(character)
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    cells.append("".join(current).strip())
    return tuple(cells)


def parse_reference_rows(text: str) -> dict[ConfigField, ReferenceRow]:
    """Parse field rows from the two tracked config-reference sections."""

    section: str | None = None
    rows: dict[ConfigField, ReferenceRow] = {}
    for line in text.splitlines():
        heading = _SECTION_HEADING.match(line)
        if heading is not None:
            section = heading.group("section")
            continue
        if line.startswith("## "):
            section = None
            continue
        field_match = _FIELD_ROW.match(line)
        if section is None or field_match is None:
            continue
        cells = _split_markdown_row(line)
        if len(cells) != 4:
            raise ValueError(f"malformed config-reference row: {line}")
        field = ConfigField(section, field_match.group("field"))
        if field in rows:
            raise ValueError(f"duplicate config-reference row for {field.dotted}")
        rows[field] = ReferenceRow(default=cells[2], description=cells[3])
    return rows


def parse_inert_markers(text: str) -> dict[ConfigField, InertMarker]:
    """Parse JSON markers that make inert documentation machine-checkable."""

    markers: dict[ConfigField, InertMarker] = {}
    for match in _INERT_MARKER.finditer(text):
        payload = json.loads(match.group("payload"))
        if set(payload) != {"section", "field", "status", "effective_control"}:
            raise ValueError("config-field-contract marker has an invalid schema")
        if payload["status"] != "inert":
            raise ValueError("config-field-contract marker status must be inert")
        if payload["section"] not in TRACKED_SECTIONS:
            raise ValueError("config-field-contract marker has an unknown section")
        if not isinstance(payload["field"], str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", payload["field"]
        ):
            raise ValueError("config-field-contract marker has an invalid field name")
        field = ConfigField(payload["section"], payload["field"])
        effective_control = payload["effective_control"]
        if not isinstance(effective_control, str) or not effective_control.strip():
            raise ValueError(f"inert marker for {field.dotted} needs an effective control")
        if field in markers:
            raise ValueError(f"duplicate inert marker for {field.dotted}")
        markers[field] = InertMarker(field=field, effective_control=effective_control)
    return markers


def _literal_default(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("`") and normalized.endswith("`"):
        normalized = normalized[1:-1].strip()
    if normalized.startswith('"') and normalized.endswith('"'):
        normalized = normalized[1:-1]
    return normalized


def opus_default_violations(direct_model: str, consensus_model: str) -> tuple[str, ...]:
    """Validate the intentionally different direct and OpenRouter Opus ids."""

    direct_pattern = re.compile(r"claude-opus-(?P<major>[1-9][0-9]*)-(?P<minor>0|[1-9][0-9]*)")
    consensus_pattern = re.compile(
        r"openrouter/anthropic/claude-opus-"
        r"(?P<major>[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)"
    )
    direct_match = direct_pattern.fullmatch(direct_model)
    consensus_match = consensus_pattern.fullmatch(consensus_model)
    violations: list[str] = []

    if direct_model == consensus_model:
        violations.append("Opus defaults must use distinct direct and OpenRouter identifiers")
    if direct_match is None:
        violations.append(
            "evaluation.semantic_model: invalid direct Opus default "
            f"{direct_model!r}; expected 'claude-opus-<major>-<minor>'"
        )
    if consensus_match is None:
        violations.append(
            "consensus.advocate_model: invalid OpenRouter Opus default "
            f"{consensus_model!r}; expected "
            "'openrouter/anthropic/claude-opus-<major>.<minor>'"
        )
    if direct_match is not None and consensus_match is not None:
        direct_version = direct_match.group("major", "minor")
        consensus_version = consensus_match.group("major", "minor")
        if direct_version != consensus_version:
            expected_consensus = (
                f"openrouter/anthropic/claude-opus-{direct_version[0]}.{direct_version[1]}"
            )
            violations.append(
                "consensus.advocate_model: OpenRouter Opus default "
                f"{consensus_model!r} does not correspond to direct default {direct_model!r}; "
                f"expected {expected_consensus!r}"
            )
    return tuple(violations)


def audit_contract(
    *,
    fields: frozenset[ConfigField],
    reads: frozenset[ConfigField],
    rows: Mapping[ConfigField, ReferenceRow],
    markers: Mapping[ConfigField, InertMarker],
    allowlist: Mapping[ConfigField, str],
    documented_defaults: Mapping[ConfigField, str],
) -> ContractReport:
    """Classify every field and return precise bidirectional drift failures."""

    violations: list[str] = []
    marker_fields = frozenset(markers)
    allowlisted_fields = frozenset(allowlist)

    for stale in sorted((marker_fields | allowlisted_fields | frozenset(rows)) - fields):
        violations.append(f"{stale.dotted}: docs/allowlist names no schema field")

    for field, reason in sorted(allowlist.items()):
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            violations.append(f"{field.dotted}: schema-only allowlist needs a concrete rationale")
        if field in reads:
            violations.append(
                f"{field.dotted}: production-wired field remains schema-only allowlisted"
            )
        if field in marker_fields:
            violations.append(f"{field.dotted}: field is both documented inert and allowlisted")

    for field in sorted(fields):
        dispositions = (
            int(field in reads) + int(field in marker_fields) + int(field in allowlisted_fields)
        )
        if dispositions == 0:
            violations.append(
                f"{field.dotted}: no production read, inert documentation, or schema-only rationale"
            )
        elif dispositions > 1:
            violations.append(f"{field.dotted}: conflicting config-field dispositions")

        row = rows.get(field)
        if row is None:
            violations.append(f"{field.dotted}: missing docs/config-reference.md field row")
            continue
        visibly_inert = "currently inert" in row.description.lower()
        if field in reads and visibly_inert:
            violations.append(f"{field.dotted}: production-wired field is still documented inert")
        if field in marker_fields:
            marker = markers[field]
            if not visibly_inert:
                violations.append(
                    f"{field.dotted}: inert marker lacks visible 'Currently inert' text"
                )
            if "effective control:" not in row.description.lower():
                violations.append(f"{field.dotted}: inert docs do not label the effective control")
            if marker.effective_control not in row.description:
                violations.append(
                    f"{field.dotted}: visible docs omit marker effective control "
                    f"{marker.effective_control!r}"
                )
        elif visibly_inert:
            violations.append(f"{field.dotted}: visible inert text lacks a structured marker")

    for field, expected in sorted(documented_defaults.items()):
        if field not in fields:
            violations.append(f"{field.dotted}: default contract names no schema field")
            continue
        row = rows.get(field)
        if row is not None and _literal_default(row.default) != expected:
            violations.append(
                f"{field.dotted}: documented default {_literal_default(row.default)!r} "
                f"does not match {expected!r}"
            )

    return ContractReport(tuple(sorted(set(violations))), reads)


def audit_repository(repo_root: Path = REPO_ROOT) -> ContractReport:
    """Audit the checked-out repository contract."""

    fields = schema_fields()
    reference = (repo_root / REFERENCE_PATH).read_text(encoding="utf-8")
    report = audit_contract(
        fields=fields,
        reads=runtime_reads(repo_root / "src" / "ouroboros", fields),
        rows=parse_reference_rows(reference),
        markers=parse_inert_markers(reference),
        allowlist=SCHEMA_ONLY_ALLOWLIST,
        documented_defaults=DOCUMENTED_DEFAULTS,
    )
    return ContractReport(
        tuple(
            sorted(
                set(report.violations)
                | set(opus_default_violations(DEFAULT_OPUS_MODEL, DEFAULT_CONSENSUS_OPUS_MODEL))
            )
        ),
        report.runtime_reads,
    )


def main() -> int:
    try:
        report = audit_repository()
    except (OSError, SyntaxError, ValueError) as error:
        print(f"Config reference contract could not run: {error}", file=sys.stderr)
        return 2
    if report.violations:
        print("Config reference contract violations:")
        for violation in report.violations:
            print(f"- {violation}")
        return 1
    reads = ", ".join(field.dotted for field in sorted(report.runtime_reads))
    print(f"Config reference contract OK ({len(report.runtime_reads)} production reads: {reads})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
