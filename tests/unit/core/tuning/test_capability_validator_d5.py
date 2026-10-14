from __future__ import annotations

import ast
import inspect
import re
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from benchbox.core.tuning.capability_registry import (
    PLATFORM_ALIASES,
    PLATFORM_TUNING_CAPABILITIES,
    get_capability,
    is_not_rendered_at_execution,
    known_registry_platforms,
    resolve_platform_key,
)
from benchbox.core.tuning.interface import TableTuning, TuningColumn, TuningType, UnifiedTuningConfiguration

pytestmark = [
    pytest.mark.unit,
    pytest.mark.fast,
]


def _partitioned_config() -> UnifiedTuningConfiguration:
    config = UnifiedTuningConfiguration()
    config.disable_all_constraints()
    config.table_tunings["orders"] = TableTuning(
        table_name="orders",
        partitioning=[TuningColumn(name="o_orderdate", type="DATE", order=1)],
    )
    return config


def _distributed_config() -> UnifiedTuningConfiguration:
    config = UnifiedTuningConfiguration()
    config.disable_all_constraints()
    config.table_tunings["lineitem"] = TableTuning(
        table_name="lineitem",
        distribution=[TuningColumn(name="l_orderkey", type="INTEGER", order=1)],
    )
    return config


class TestRequestedButNotRenderedIsAnError:
    def test_partitioning_on_postgresql_errors(self) -> None:
        errors, _warnings = _partitioned_config().validate_for_platform_detailed("postgresql")
        assert any("requested partitioning is not applied at execution on postgresql" in e for e in errors)

    def test_partitioning_on_postgres_family_errors(self) -> None:
        for platform in ("cedardb", "citus", "paradedb", "pg-duckdb"):
            errors, _warnings = _partitioned_config().validate_for_platform_detailed(platform)
            assert any("is not applied at execution" in e for e in errors), platform

    def test_distribution_on_redshift_errors(self) -> None:
        errors, _warnings = _distributed_config().validate_for_platform_detailed("redshift")
        assert any("requested distribution is not applied at execution on redshift" in e for e in errors)

    def test_layout_on_trino_and_athena_errors(self) -> None:
        for platform in ("trino", "athena"):
            errors, _warnings = _partitioned_config().validate_for_platform_detailed(platform)
            assert any("is not applied at execution" in e for e in errors), platform

    def test_starburst_alias_resolves_to_trino_rules(self) -> None:
        assert resolve_platform_key("Starburst") == "trino"
        errors, _warnings = _partitioned_config().validate_for_platform_detailed("starburst")
        assert any("is not applied at execution" in e for e in errors)

    def test_rendering_registry_entry_does_not_error(self) -> None:
        errors, warnings = _distributed_config().validate_for_platform_detailed("starrocks")
        assert errors == []
        assert warnings == []

    def test_doris_layout_and_constraints_error(self) -> None:
        config = UnifiedTuningConfiguration()
        errors, _warnings = config.validate_for_platform_detailed("doris")
        assert any("requested primary_keys is not applied at execution" in e for e in errors)
        assert any("requested foreign_keys is not applied at execution" in e for e in errors)

    def test_datafusion_constraints_error(self) -> None:
        config = UnifiedTuningConfiguration()
        errors, _warnings = config.validate_for_platform_detailed("datafusion")
        assert len(errors) == 4

    def test_motherduck_sorting_renders_but_constraints_error(self) -> None:
        sort_config = UnifiedTuningConfiguration()
        sort_config.disable_all_constraints()
        sort_config.table_tunings["orders"] = TableTuning(
            table_name="orders",
            sorting=[TuningColumn(name="o_orderkey", type="INTEGER", order=1)],
        )
        errors, _warnings = sort_config.validate_for_platform_detailed("motherduck")
        assert errors == []

        config = UnifiedTuningConfiguration()
        errors, _warnings = config.validate_for_platform_detailed("motherduck")
        assert any("is not applied at execution" in e for e in errors)

    def test_fabric_warehouse_partitioning_errors(self) -> None:
        errors, _warnings = _partitioned_config().validate_for_platform_detailed("fabric-warehouse")
        assert any("is not applied at execution" in e for e in errors)

    def test_stripped_constraint_types_error(self) -> None:
        for platform in ("doris", "databend", "starrocks"):
            config = UnifiedTuningConfiguration()
            config.disable_all_constraints()
            config.unique_constraints.enabled = True
            errors, _warnings = config.validate_for_platform_detailed(platform)
            assert any("requested unique_constraints is not applied at execution" in e for e in errors), platform

    def test_vacuous_unique_check_types_error(self) -> None:
        for platform in (
            "duckdb",
            "databricks",
            "snowflake",
            "sqlite",
            "postgresql",
            "mysql",
            "singlestore",
            "cedardb",
            "citus",
            "paradedb",
            "fabric-warehouse",
            "athena",
            "azure-synapse",
            "timescaledb",
            "redshift",
            "firebolt",
            "questdb",
            "pg-duckdb",
            "pg-mooncake",
            "trino",
            "presto",
        ):
            for tuning_attr in ("unique_constraints", "check_constraints"):
                config = UnifiedTuningConfiguration()
                config.disable_all_constraints()
                getattr(config, tuning_attr).enabled = True
                errors, _warnings = config.validate_for_platform_detailed(platform)
                assert any(f"requested {tuning_attr} is not applied at execution" in e for e in errors), (
                    platform,
                    tuning_attr,
                )

    def test_bigquery_unique_and_clickhouse_fk_check_error(self) -> None:
        for platform, tuning_attr in (
            ("bigquery", "unique_constraints"),
            ("clickhouse", "foreign_keys"),
            ("clickhouse", "check_constraints"),
        ):
            config = UnifiedTuningConfiguration()
            config.disable_all_constraints()
            getattr(config, tuning_attr).enabled = True
            errors, _warnings = config.validate_for_platform_detailed(platform)
            assert any(f"requested {tuning_attr} is not applied at execution" in e for e in errors), (
                platform,
                tuning_attr,
            )

    def test_dropped_pk_fk_types_error(self) -> None:
        for platform, tuning_attr in (
            ("firebolt", "primary_keys"),
            ("firebolt", "foreign_keys"),
            ("questdb", "primary_keys"),
            ("questdb", "foreign_keys"),
            ("trino", "primary_keys"),
            ("presto", "primary_keys"),
        ):
            config = UnifiedTuningConfiguration()
            config.disable_all_constraints()
            getattr(config, tuning_attr).enabled = True
            errors, _warnings = config.validate_for_platform_detailed(platform)
            assert any(f"requested {tuning_attr} is not applied at execution" in e for e in errors), (
                platform,
                tuning_attr,
            )

    def test_rendered_pk_fk_types_do_not_error(self) -> None:
        for platform, tuning_attr in (
            ("pg-duckdb", "primary_keys"),
            ("pg-duckdb", "foreign_keys"),
            ("pg-mooncake", "primary_keys"),
            ("pg-mooncake", "foreign_keys"),
            ("trino", "foreign_keys"),
            ("presto", "foreign_keys"),
        ):
            config = UnifiedTuningConfiguration()
            config.disable_all_constraints()
            getattr(config, tuning_attr).enabled = True
            errors, _warnings = config.validate_for_platform_detailed(platform)
            assert errors == [], (platform, tuning_attr, errors)


_LIFECYCLE_ROOTS = frozenset(
    {
        "create_schema",
        "load_data",
        "apply_unified_tuning",
        "apply_table_tunings",
        "apply_post_load_tunings",
        "run_post_load_tunings",
        "configure_for_benchmark",
    }
)

_EXECUTE_ATTRS = frozenset({"execute", "executescript", "exec"})

_QUERY_ATTRS = frozenset({"sql", "query"})

_STATEMENT_EXEC_RECEIVERS = frozenset({"connection", "conn", "client", "cursor", "engine", "db"})

_STATEMENT_EXEC_ARG_MARKERS = frozenset({"ddl", "create", "table", "schema", "statement"})

_STATEMENT_EXEC_METHODS = frozenset(
    {
        "create_schema",
        "apply_unified_tuning",
        "apply_table_tunings",
        "apply_post_load_tunings",
        "run_post_load_tunings",
        "configure_for_benchmark",
        "apply_constraint_configuration",
        "_execute_schema_statements",
        "_execute_sql_statement",
    }
)

_TUNING_READ_NAMES = frozenset(
    {
        "tuning_config",
        "effective_config",
        "effective_tuning",
        "table_tuning",
        "table_tunings",
        "unified_config",
        "get_columns_by_type",
        "_get_constraint_configuration",
        "get_effective_tuning_configuration",
    }
)

_STRIP_CALLS = frozenset({"strip_primary_keys", "strip_foreign_keys"})

_TYPE_EVIDENCE_TOKENS = {
    "partitioning": (
        "partition by",
        "partition_by",
        "partitioned by",
        "partitioned_by",
    ),
    "sorting": ("sort_key", "sortkey", "sort key", "ctas"),
    "clustering": ("cluster by", "cluster_by", "clustering"),
    "distribution": ("distributed by", "distribution", "diststyle", "distkey"),
    "primary_keys": ("primary_key", "primary key"),
    "foreign_keys": ("foreign_key", "foreign key"),
    "unique_constraints": ("unique constraint", "unique_constraint", "unique index", "unique_index"),
    "check_constraints": ("check constraint", "check_constraint"),
    "z_ordering": ("z_order", "zorder"),
    "liquid_clustering": ("liquid_clustering", "liquid clustering"),
    "auto_optimize": ("auto_optimize", "autooptimize"),
    "auto_compact": ("auto_compact", "autocompact"),
    "bloom_filters": ("bloom",),
    "materialized_views": ("materialized",),
}


def _module_functions(adapter_cls: type) -> dict[str, list[tuple[str, ast.FunctionDef]]]:
    cache: dict[str, dict[str, list[tuple[str, ast.FunctionDef]]]] = _module_functions.__dict__.setdefault("cache", {})
    found: dict[str, list[tuple[str, ast.FunctionDef]]] = {}
    for klass in type.mro(adapter_cls):
        module_name = getattr(klass, "__module__", "")
        if not module_name.startswith("benchbox."):
            continue
        if module_name not in cache:
            try:
                module = ast.parse(textwrap.dedent(inspect.getsource(sys.modules[module_name])))
            except (OSError, TypeError, SyntaxError, KeyError):
                cache[module_name] = {}
                continue
            entries: dict[str, list[tuple[str, ast.FunctionDef]]] = {}
            for node in module.body:
                if isinstance(node, ast.FunctionDef):
                    entries.setdefault(node.name, []).append((module_name, node))
            cache[module_name] = entries
        for name, nodes in cache[module_name].items():
            found.setdefault(name, []).extend(nodes)
    return found


def _collect_definers(adapter_cls: type) -> tuple[list[type], dict[str, list[tuple[str, ast.FunctionDef]]]]:
    mro = [klass for klass in type.mro(adapter_cls) if klass is not object]
    definers: dict[str, list[tuple[str, ast.FunctionDef]]] = {}
    for klass in mro:
        try:
            source = inspect.getsource(klass)
        except (OSError, TypeError):
            continue
        try:
            module = ast.parse(textwrap.dedent(source))
        except SyntaxError:
            continue
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == klass.__name__:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        definers.setdefault(item.name, []).append((klass.__name__, item))
    return mro, definers


_FOLLOW_ATTRS = frozenset(
    {
        "create_schema",
        "load_data",
        "apply_unified_tuning",
        "apply_table_tunings",
        "apply_post_load_tunings",
        "run_post_load_tunings",
        "configure_for_benchmark",
        "apply_constraint_configuration",
        "apply_ctas_sort",
        "get_effective_tuning_configuration",
        "get_create_tables_sql",
    }
)


def _receiver_root(value: ast.AST) -> str | None:
    while isinstance(value, ast.Attribute):
        value = value.value
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "super":
        return "super"
    return None


def _module_import_map(module_name: str) -> dict[str, str]:
    cache: dict[str, dict[str, str]] = _module_import_map.__dict__.setdefault("cache", {})
    if module_name not in cache:
        mapping: dict[str, str] = {}
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(sys.modules[module_name])))
        except (OSError, TypeError, SyntaxError, KeyError):
            cache[module_name] = mapping
            return mapping
        for item in ast.walk(tree):
            if isinstance(item, ast.ClassDef):
                mapping.setdefault(item.name, module_name)
            elif isinstance(item, ast.ImportFrom) and item.level == 0 and item.module:
                for alias in item.names:
                    mapping.setdefault(alias.asname or alias.name, item.module)
        cache[module_name] = mapping
    return cache[module_name]


def _composed_methods(class_name: str, module_name: str) -> dict[str, ast.FunctionDef]:
    cache: dict[tuple[str, str], dict[str, ast.FunctionDef]] = _composed_methods.__dict__.setdefault("cache", {})
    key = (module_name, class_name)
    if key not in cache:
        methods: dict[str, ast.FunctionDef] = {}
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(sys.modules[module_name])))
        except (OSError, TypeError, SyntaxError, KeyError):
            cache[key] = methods
            return methods
        for item in tree.body:
            if isinstance(item, ast.ClassDef) and item.name == class_name:
                for entry in item.body:
                    if isinstance(entry, ast.FunctionDef):
                        methods.setdefault(entry.name, entry)
        cache[key] = methods
    return cache[key]


def _mro_definer(
    attr: str,
    definers: dict[str, list[tuple[str, ast.FunctionDef]]],
    mro: list[type],
    skip_modules: frozenset[str],
) -> str | None:
    for klass in mro:
        if klass.__module__ in skip_modules:
            continue
        if any(entry[0] == klass.__name__ for entry in definers.get(attr, [])):
            return klass.__name__
    return None


def _composed_locals(node: ast.FunctionDef, body_module: str) -> dict[str, str]:
    import_map = _module_import_map(body_module)
    locals_map: dict[str, str] = {}
    for child in ast.walk(node):
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(child, ast.Assign) and len(child.targets) == 1:
            target, value = child.targets[0], child.value
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            target, value = child.target, child.value
        if (
            isinstance(target, ast.Name)
            and target.id not in ("self", "adapter", "cls")
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in import_map
            and import_map[value.func.id].startswith("benchbox.")
        ):
            locals_map[target.id] = value.func.id
    return locals_map


def _enqueue_composed_call(
    func: ast.expr,
    locals_map: dict[str, str],
    body_module: str,
    composed: dict[str, tuple[str, dict[str, ast.FunctionDef]]],
    queue: list[tuple[str, str]],
) -> None:
    if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
        return
    if func.value.id not in locals_map:
        return
    resolved_class = locals_map[func.value.id]
    resolved_module = _module_import_map(body_module).get(resolved_class)
    if resolved_module is None:
        return
    methods = _composed_methods(resolved_class, resolved_module)
    if methods:
        composed.setdefault(resolved_class, (resolved_module, methods))
    if func.attr in methods:
        queue.append((resolved_class, func.attr))


def _enqueue_self_attr(
    func: ast.Attribute,
    class_name: str,
    owned: dict[str, ast.FunctionDef],
    in_composed: bool,
    definers: dict[str, list[tuple[str, ast.FunctionDef]]],
    mro: list[type],
    skip_modules: frozenset[str],
    queue: list[tuple[str, str]],
) -> None:
    if func.attr in owned:
        queue.append((class_name, func.attr))
        return
    if in_composed:
        return
    resolved = _mro_definer(func.attr, definers, mro, skip_modules)
    if resolved is not None:
        queue.append((resolved, func.attr))


def _enqueue_super_attr(
    func: ast.Attribute,
    class_name: str,
    class_index: dict[str, int],
    mro: list[type],
    definers: dict[str, list[tuple[str, ast.FunctionDef]]],
    skip_modules: frozenset[str],
    queue: list[tuple[str, str]],
) -> None:
    inner = func.value
    if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)):
        return
    if inner.func.id != "super" or class_name not in class_index:
        return
    position = class_index[class_name]
    for klass in mro[position + 1 :]:
        if klass.__module__ in skip_modules:
            continue
        if any(item[0] == klass.__name__ for item in definers.get(func.attr, [])):
            queue.append((klass.__name__, func.attr))
            break


def _enqueue_follow_attr(
    func: ast.Attribute,
    definers: dict[str, list[tuple[str, ast.FunctionDef]]],
    mro: list[type],
    skip_modules: frozenset[str],
    helpers: dict[str, list[tuple[str, ast.FunctionDef]]],
    helper_nodes: dict[tuple[str, str], ast.FunctionDef],
    queue: list[tuple[str, str]],
) -> None:
    if _receiver_root(func.value) not in ("self", "adapter"):
        return
    resolved = _mro_definer(func.attr, definers, mro, skip_modules)
    if resolved is not None:
        queue.append((resolved, func.attr))
    for position, (module_name, helper) in enumerate(helpers.get(func.attr, [])):
        if module_name in skip_modules:
            continue
        key = (func.attr, f"{helper.name}#{position}")
        helper_nodes.setdefault(key, helper)
        queue.append(key)


def _enqueue_callees(
    node: ast.FunctionDef,
    class_name: str,
    definers: dict[str, list[tuple[str, ast.FunctionDef]]],
    class_index: dict[str, int],
    mro: list[type],
    helpers: dict[str, list[tuple[str, ast.FunctionDef]]],
    helper_nodes: dict[tuple[str, str], ast.FunctionDef],
    queue: list[tuple[str, str]],
    *,
    skip_modules: frozenset[str] = frozenset(),
    body_module: str | None = None,
    composed: dict[str, tuple[str, dict[str, ast.FunctionDef]]] | None = None,
) -> None:
    entry = (composed or {}).get(class_name)
    owned = entry[1] if entry is not None else {}
    in_composed = entry is not None and class_name not in class_index
    locals_map = _composed_locals(node, body_module) if body_module is not None else {}
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            _enqueue_self_attr(func, class_name, owned, in_composed, definers, mro, skip_modules, queue)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call):
            _enqueue_super_attr(func, class_name, class_index, mro, definers, skip_modules, queue)
        elif isinstance(func, ast.Attribute) and func.attr in _FOLLOW_ATTRS:
            _enqueue_follow_attr(func, definers, mro, skip_modules, helpers, helper_nodes, queue)
        elif isinstance(func, ast.Name) and func.id in helpers:
            for position, (module_name, helper) in enumerate(helpers[func.id]):
                if module_name in skip_modules:
                    continue
                key = (func.id, f"{helper.name}#{position}")
                helper_nodes.setdefault(key, helper)
                queue.append(key)
        if composed is not None and body_module is not None:
            _enqueue_composed_call(func, locals_map, body_module, composed, queue)


def _reachable_bodies(adapter_cls: type, skip_modules: frozenset[str] = frozenset()) -> list[ast.FunctionDef]:
    mro, definers = _collect_definers(adapter_cls)
    class_index = {klass.__name__: position for position, klass in enumerate(mro)}
    by_name: dict[str, type] = {}
    for klass in mro:
        by_name.setdefault(klass.__name__, klass)

    def _kept(class_name: str) -> bool:
        owner = by_name.get(class_name)
        return owner is None or owner.__module__ not in skip_modules

    helpers = _module_functions(adapter_cls)
    helper_nodes: dict[tuple[str, str], ast.FunctionDef] = {}
    helper_modules: dict[tuple[str, str], str] = {}
    for helper_name, nodes in helpers.items():
        for position, (module_name, helper) in enumerate(nodes):
            key = (helper_name, f"{helper.name}#{position}")
            helper_nodes[key] = helper
            helper_modules[key] = module_name
    composed: dict[str, tuple[str, dict[str, ast.FunctionDef]]] = {}
    reached: dict[tuple[str, str], ast.FunctionDef] = {}
    queue: list[tuple[str, str]] = []
    for name, entries in definers.items():
        if name in _LIFECYCLE_ROOTS:
            seeded = next((owner for owner, _ in entries if _kept(owner)), None)
            if seeded is not None:
                queue.append((seeded, name))
    while queue:
        class_name, name = queue.pop()
        if (class_name, name) in reached:
            continue
        if (class_name, name) in helper_nodes:
            reached[(class_name, name)] = helper_nodes[(class_name, name)]
            _enqueue_callees(
                helper_nodes[(class_name, name)],
                class_name,
                definers,
                class_index,
                mro,
                helpers,
                helper_nodes,
                queue,
                skip_modules=skip_modules,
                body_module=helper_modules[(class_name, name)],
                composed=composed,
            )
            continue
        entries = [entry for entry in definers.get(name, []) if entry[0] == class_name and _kept(entry[0])]
        if entries:
            node = entries[0][1]
            reached[(class_name, name)] = node
            _enqueue_callees(
                node,
                class_name,
                definers,
                class_index,
                mro,
                helpers,
                helper_nodes,
                queue,
                skip_modules=skip_modules,
                body_module=by_name[class_name].__module__ if class_name in by_name else None,
                composed=composed,
            )
            continue
        if class_name in composed and name in composed[class_name][1]:
            node = composed[class_name][1][name]
            reached[(class_name, name)] = node
            _enqueue_callees(
                node,
                class_name,
                definers,
                class_index,
                mro,
                helpers,
                helper_nodes,
                queue,
                skip_modules=skip_modules,
                body_module=composed[class_name][0],
                composed=composed,
            )
            continue
    inline: list[ast.FunctionDef] = list(reached.values())
    return inline


def _early_exit_block(block: list[ast.stmt]) -> bool:
    return bool(block) and all(isinstance(stmt, (ast.Return, ast.Raise, ast.Continue, ast.Break)) for stmt in block)


def _strip_conditional(child: ast.Call, parents: dict[int, ast.AST], body: ast.FunctionDef) -> bool:
    descendant: ast.AST = child
    node = parents.get(id(child))
    while node is not None and node is not body:
        if isinstance(node, (ast.IfExp, ast.comprehension)):
            return True
        if isinstance(node, ast.If):
            if descendant is node.test:
                return True
            scope = node.body if any(descendant is stmt for stmt in node.body) else node.orelse
            if not _early_exit_block(scope):
                return True
        descendant = node
        node = parents.get(id(node))
    return False


def _sub_replacement_reemits_match(replacement: ast.AST | None) -> bool:
    if replacement is None:
        return True
    if isinstance(replacement, ast.Constant) and isinstance(replacement.value, str):
        return "\\1" in replacement.value or "\\g" in replacement.value
    return True


def _execution_receiver(value: ast.AST) -> bool:
    root = _receiver_root(value)
    if root in _STATEMENT_EXEC_RECEIVERS:
        return True
    if root not in ("self", "adapter", "cls"):
        return False
    try:
        dotted = ast.unparse(value).lower()
    except (ValueError, SyntaxError):
        return False
    return dotted.split(".")[-1].strip("_") in _STATEMENT_EXEC_RECEIVERS


def _statement_argument(call: ast.Call) -> bool:
    try:
        haystack = " ".join(
            [ast.unparse(argument) for argument in call.args]
            + [ast.unparse(keyword.value) for keyword in call.keywords]
        ).lower()
    except (ValueError, SyntaxError):
        return False
    return any(marker in haystack for marker in _STATEMENT_EXEC_ARG_MARKERS)


def _scan_call(
    child: ast.Call,
    parents: dict[int, ast.AST],
    acc: dict[str, bool],
    enclosing: str,
    body: ast.FunctionDef,
) -> None:
    func = child.func
    if isinstance(func, ast.Attribute) and func.attr in _EXECUTE_ATTRS:
        if not isinstance(parents.get(id(child)), ast.Return):
            acc["sql_exec"] = True
    if isinstance(func, ast.Attribute) and func.attr in _QUERY_ATTRS:
        if (
            enclosing in _STATEMENT_EXEC_METHODS
            and not isinstance(parents.get(id(child)), ast.Return)
            and _execution_receiver(func.value)
            and _statement_argument(child)
        ):
            acc["sql_exec"] = True
    if isinstance(func, ast.Name) and func.id in _STRIP_CALLS:
        if not _strip_conditional(child, parents, body):
            if func.id == "strip_primary_keys":
                acc["strip_pk"] = True
            else:
                acc["strip_fk"] = True
    if isinstance(func, ast.Attribute) and func.attr == "sub":
        pattern = child.args[0] if child.args else None
        replacement = child.args[1] if len(child.args) > 1 else None
        lowered = pattern.value.lower() if isinstance(pattern, ast.Constant) and isinstance(pattern.value, str) else ""
        kinds = set()
        if "primary" in lowered and "key" in lowered:
            kinds.add("pk")
        if "foreign" in lowered and "key" in lowered:
            kinds.add("fk")
        if kinds and not _sub_replacement_reemits_match(replacement) and not _strip_conditional(child, parents, body):
            if "pk" in kinds:
                acc["strip_pk"] = True
            if "fk" in kinds:
                acc["strip_fk"] = True
    if isinstance(func, ast.Attribute) and func.attr == "get_create_tables_sql":
        acc["tuning_read"] = True
    for keyword in child.keywords:
        if keyword.arg == "tuning_config":
            acc["tuning_read"] = True
            if isinstance(func, ast.Attribute) and func.attr == "get_create_tables_sql":
                acc["ddl_handoff"] = True


def _signals(
    bodies: list[ast.FunctionDef],
) -> tuple[bool, bool, bool, bool, bool, list[str]]:
    acc = {
        "tuning_read": False,
        "sql_exec": False,
        "ddl_handoff": False,
        "strip_pk": False,
        "strip_fk": False,
    }
    texts: list[str] = []
    for body in bodies:
        try:
            texts.append(ast.unparse(body).lower())
        except (ValueError, SyntaxError):
            continue
        parents: dict[int, ast.AST] = {
            id(child): node for node in ast.walk(body) for child in ast.iter_child_nodes(node)
        }
        for child in ast.walk(body):
            if isinstance(child, ast.Name) and child.id in _TUNING_READ_NAMES:
                acc["tuning_read"] = True
            if isinstance(child, ast.Call):
                _scan_call(child, parents, acc, body.name, body)
    return (
        acc["tuning_read"],
        acc["sql_exec"],
        acc["ddl_handoff"],
        acc["strip_pk"],
        acc["strip_fk"],
        texts,
    )


_TYPE_EVIDENCE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    tuning_value: tuple(re.compile(r"\b" + re.escape(token) + r"\b") for token in tokens)
    for tuning_value, tokens in _TYPE_EVIDENCE_TOKENS.items()
}


def _rendered_gate_passes(tuning_read: bool, sql_exec: bool, evidenced: bool, override: bool) -> bool:
    return override or ((tuning_read and sql_exec) and evidenced)


def _noop_gate_excused(
    tuning_value: str, strip_pk: bool, strip_fk: bool, reg_pk: bool, reg_fk: bool, justified: bool
) -> bool:
    if tuning_value == "primary_keys" and (strip_pk or reg_pk):
        return True
    if tuning_value == "foreign_keys" and (strip_fk or reg_fk):
        return True
    return justified


def _type_evidenced(
    tuning_type_value: str, texts: list[str], ddl_handoff: bool, strip_pk: bool, strip_fk: bool
) -> bool:
    patterns = _TYPE_EVIDENCE_PATTERNS.get(tuning_type_value, ())
    if any(pattern.search(text) for text in texts for pattern in patterns):
        return True
    if tuning_type_value in ("primary_keys", "foreign_keys") and ddl_handoff:
        if tuning_type_value == "primary_keys" and not strip_pk:
            return True
        if tuning_type_value == "foreign_keys" and not strip_fk:
            return True
    return False


_STRIP_VERBS = frozenset({"strip", "remove", "drop"})

_KIND_STRIP_STEMS = {
    "primary_keys": ("primary",),
    "foreign_keys": ("foreign",),
    "unique_constraints": ("unique",),
    "check_constraints": ("check",),
}


def _loaded_ddl_rewrite_rules() -> None:
    done: bool = _loaded_ddl_rewrite_rules.__dict__.setdefault("done", False)
    if done:
        return
    import importlib
    import pkgutil

    import benchbox.sql_compat.rules.ddl_optimize as package

    for module in pkgutil.iter_modules(package.__path__):
        try:
            importlib.import_module(f"{package.__name__}.{module.name}")
        except ModuleNotFoundError:
            continue
    _loaded_ddl_rewrite_rules.__dict__["done"] = True


def _applied_kind_strip_rules(platform_key: str) -> list[tuple[str, str]]:
    _loaded_ddl_rewrite_rules()
    from benchbox.sql_compat.context import Phase
    from benchbox.sql_compat.registry import REGISTRY

    keys = {platform_key} | {alias for alias, target in PLATFORM_ALIASES.items() if target == platform_key}
    matches: list[tuple[str, str]] = []
    for key in keys:
        rules = REGISTRY.resolve_platform_rules(Phase.DDL_OPTIMIZE, key)
        for decision in rules:
            payload = decision.payload
            if getattr(payload, "governance_only", True):
                continue
            haystack = " ".join(
                part
                for part in (
                    str(getattr(payload, "transformer_id", "")),
                    str(getattr(decision, "rule_id", "")),
                    str(getattr(payload, "description", "")),
                )
                if part
            ).lower()
            if not any(verb in haystack for verb in _STRIP_VERBS):
                continue
            for tuning_value, stems in _KIND_STRIP_STEMS.items():
                if any(stem in haystack for stem in stems):
                    matches.append((tuning_value, str(getattr(payload, "transformer_id", ""))))
    return matches


def _registered_kind_strips(platform_key: str) -> tuple[bool, bool]:
    matches = {kind for kind, _ in _applied_kind_strip_rules(platform_key)}
    return ("primary_keys" in matches, "foreign_keys" in matches)


def _absence_holds(texts: list[str], phrases: tuple[str, ...]) -> bool:
    return all(phrase not in text for text in texts for phrase in phrases)


def _all_adapter_texts(adapter_cls: type) -> list[str]:
    _mro, definers = _collect_definers(adapter_cls)
    helpers = _module_functions(adapter_cls)
    texts: list[str] = []
    for entries in definers.values():
        for _, node in entries:
            try:
                texts.append(ast.unparse(node).lower())
            except (ValueError, SyntaxError):
                continue
    for nodes in helpers.values():
        for _, node in nodes:
            try:
                texts.append(ast.unparse(node).lower())
            except (ValueError, SyntaxError):
                continue
    return texts


def _adapter_class(platform_key: str) -> type | None:
    from benchbox.platforms.athena import AthenaAdapter
    from benchbox.platforms.azure_synapse import AzureSynapseAdapter
    from benchbox.platforms.bigquery import BigQueryAdapter
    from benchbox.platforms.cedardb import CedarDBAdapter
    from benchbox.platforms.citus import CitusAdapter
    from benchbox.platforms.clickhouse_server import ClickHouseServerAdapter
    from benchbox.platforms.databend.adapter import DatabendAdapter
    from benchbox.platforms.databricks import DatabricksAdapter
    from benchbox.platforms.datafusion import DataFusionAdapter
    from benchbox.platforms.doris import DorisAdapter
    from benchbox.platforms.duckdb import DuckDBAdapter
    from benchbox.platforms.fabric_warehouse import FabricWarehouseAdapter
    from benchbox.platforms.firebolt import FireboltAdapter
    from benchbox.platforms.motherduck import MotherDuckAdapter
    from benchbox.platforms.paradedb import ParadeDBAdapter
    from benchbox.platforms.pg_duckdb import PgDuckDBAdapter
    from benchbox.platforms.pg_mooncake import PgMooncakeAdapter
    from benchbox.platforms.postgresql import PostgreSQLAdapter
    from benchbox.platforms.presto import PrestoAdapter
    from benchbox.platforms.questdb import QuestDBAdapter
    from benchbox.platforms.redshift import RedshiftAdapter
    from benchbox.platforms.singlestore import SingleStoreAdapter
    from benchbox.platforms.snowflake import SnowflakeAdapter
    from benchbox.platforms.spark import SparkAdapter
    from benchbox.platforms.sqlite import SQLiteAdapter
    from benchbox.platforms.starburst import StarburstAdapter
    from benchbox.platforms.starrocks.adapter import StarRocksAdapter
    from benchbox.platforms.timescaledb import TimescaleDBAdapter
    from benchbox.platforms.trino import TrinoAdapter

    mapping: dict[str, type] = {
        "clickhouse": ClickHouseServerAdapter,
        "athena": AthenaAdapter,
        "azure-synapse": AzureSynapseAdapter,
        "bigquery": BigQueryAdapter,
        "cedardb": CedarDBAdapter,
        "citus": CitusAdapter,
        "databend": DatabendAdapter,
        "databricks": DatabricksAdapter,
        "datafusion": DataFusionAdapter,
        "doris": DorisAdapter,
        "duckdb": DuckDBAdapter,
        "fabric-warehouse": FabricWarehouseAdapter,
        "firebolt": FireboltAdapter,
        "motherduck": MotherDuckAdapter,
        "mysql": SingleStoreAdapter,
        "paradedb": ParadeDBAdapter,
        "pg-duckdb": PgDuckDBAdapter,
        "pg-mooncake": PgMooncakeAdapter,
        "postgresql": PostgreSQLAdapter,
        "presto": PrestoAdapter,
        "questdb": QuestDBAdapter,
        "redshift": RedshiftAdapter,
        "singlestore": SingleStoreAdapter,
        "snowflake": SnowflakeAdapter,
        "spark": SparkAdapter,
        "sqlite": SQLiteAdapter,
        "starrocks": StarRocksAdapter,
        "starburst": StarburstAdapter,
        "synapse": AzureSynapseAdapter,
        "timescaledb": TimescaleDBAdapter,
        "trino": TrinoAdapter,
    }
    return mapping.get(platform_key)


_SKIP_NO_ADAPTER_CLASS = frozenset()

_BLOOM_ABSENCE_PHRASES = ("bloom",)

_MATERIALIZED_VIEW_CREATION_PHRASES = (
    "create materialized",
    "or replace materialized",
    "materialized_view(",
)

_ABSENCE_CLAIMS: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {
    ("databend", "materialized_views"): ("databend", _MATERIALIZED_VIEW_CREATION_PHRASES),
    ("databricks", "bloom_filters"): ("databricks", _BLOOM_ABSENCE_PHRASES),
    ("databricks", "materialized_views"): ("databricks", _MATERIALIZED_VIEW_CREATION_PHRASES),
    ("snowflake", "materialized_views"): ("snowflake", _MATERIALIZED_VIEW_CREATION_PHRASES),
    ("bigquery", "materialized_views"): ("bigquery", _MATERIALIZED_VIEW_CREATION_PHRASES),
    ("redshift", "materialized_views"): ("redshift", _MATERIALIZED_VIEW_CREATION_PHRASES),
    ("postgresql", "bloom_filters"): ("postgresql", _BLOOM_ABSENCE_PHRASES),
    ("postgresql", "materialized_views"): ("postgresql", _MATERIALIZED_VIEW_CREATION_PHRASES),
    ("cedardb", "bloom_filters"): ("cedardb", _BLOOM_ABSENCE_PHRASES),
    ("cedardb", "materialized_views"): ("cedardb", _MATERIALIZED_VIEW_CREATION_PHRASES),
    ("citus", "bloom_filters"): ("citus", _BLOOM_ABSENCE_PHRASES),
    ("citus", "materialized_views"): ("citus", _MATERIALIZED_VIEW_CREATION_PHRASES),
    ("paradedb", "bloom_filters"): ("paradedb", _BLOOM_ABSENCE_PHRASES),
    ("paradedb", "materialized_views"): ("paradedb", _MATERIALIZED_VIEW_CREATION_PHRASES),
    ("clickhouse", "materialized_views"): ("clickhouse", _MATERIALIZED_VIEW_CREATION_PHRASES),
    ("motherduck", "materialized_views"): ("motherduck", _MATERIALIZED_VIEW_CREATION_PHRASES),
}

_PLATFORM_SKIP_MODULES: dict[str, frozenset[str]] = {
    "mysql": frozenset({"benchbox.platforms.singlestore"}),
}

_RENDERED_OVERRIDE_JUSTIFICATIONS: dict[tuple[str, str], str] = {
    (
        "snowflake",
        "partitioning",
    ): "Snowflake has no partition clause; apply_table_tunings reads PARTITIONING columns and issues them "
    "as ALTER TABLE ... CLUSTER BY when no clustering columns are set (snowflake.py: apply_table_tunings). "
    "Clustering tokens no longer evidence partitioning, so this rendering is attested here.",
}

_NONE_JUSTIFICATION_OVERRIDES: dict[tuple[str, str], str] = {
    (
        "trino",
        "partitioning",
    ): "Preview generator uncalled at execution; apply path log-only (trino.py: generate_tuning_clause at 137, apply_table_tunings at 173).",
    (
        "trino",
        "distribution",
    ): "Preview generator uncalled at execution; apply path log-only (trino.py: generate_tuning_clause at 137, apply_table_tunings at 173).",
    (
        "trino",
        "sorting",
    ): "Preview generator uncalled at execution; apply path log-only (trino.py: generate_tuning_clause at 137, apply_table_tunings at 173).",
    (
        "trino",
        "clustering",
    ): "Preview generator uncalled at execution; apply path log-only (trino.py: generate_tuning_clause at 137, apply_table_tunings at 173).",
    (
        "presto",
        "partitioning",
    ): "Preview generator uncalled at execution; apply path log-only (presto.py: generate_tuning_clause at 154, apply_table_tunings at 176).",
    (
        "presto",
        "distribution",
    ): "Preview generator uncalled at execution; apply path log-only (presto.py: generate_tuning_clause at 154, apply_table_tunings at 176).",
    (
        "presto",
        "sorting",
    ): "Preview generator uncalled at execution; apply path log-only (presto.py: generate_tuning_clause at 154, apply_table_tunings at 176).",
    (
        "presto",
        "clustering",
    ): "Preview generator uncalled at execution; apply path log-only (presto.py: generate_tuning_clause at 154, apply_table_tunings at 176).",
    (
        "athena",
        "partitioning",
    ): "External-table rewrite never adds PARTITIONED BY; tuning clause uncalled (athena.py: _convert_to_external_table).",
    ("athena", "distribution"): "Inherited generator output excluded from TBLPROPERTIES (generators/trino.py:268-274).",
    ("athena", "sorting"): "Inherited generator output excluded from TBLPROPERTIES (generators/trino.py:268-274).",
    ("athena", "clustering"): "No clustering clause; folded into dropped sorted_by (generators/trino.py:268-274).",
    (
        "firebolt",
        "partitioning",
    ): "Preview generator uncalled at execution; apply path log-only (firebolt.py: generate_tuning_clause at 1138, apply_table_tunings at 1164).",
    (
        "firebolt",
        "distribution",
    ): "Preview generator uncalled at execution; apply path log-only (firebolt.py: generate_tuning_clause at 1138, apply_table_tunings at 1164).",
    (
        "firebolt",
        "sorting",
    ): "Firebolt sorts within segments automatically; generator log-only (generators/firebolt.py:74-92).",
    ("firebolt", "clustering"): "Clustering via PRIMARY INDEX; generator log-only (generators/firebolt.py:74-92).",
    ("azure-synapse", "partitioning"): "Execution path never adds a PARTITION clause (azure_synapse.py).",
    (
        "azure-synapse",
        "distribution",
    ): "Real DDL uses the engine default, never tuned columns (azure_synapse.py:988-990).",
    ("azure-synapse", "sorting"): "Sorting via index type; generator log-only (generators/azure_synapse.py).",
    (
        "azure-synapse",
        "clustering",
    ): "Clustering via DISTRIBUTION/INDEX; generator log-only (generators/azure_synapse.py).",
    (
        "questdb",
        "partitioning",
    ): "Execution uses adapter-local hardcoded maps, ignoring tuned columns (questdb.py:428-443).",
    ("questdb", "sorting"): "Sorts by designated timestamp automatically; generator log-only (generators/questdb.py).",
    ("questdb", "distribution"): "Single-node engine; generator warning-logs only (generators/questdb.py).",
    (
        "timescaledb",
        "partitioning",
    ): "Execution converts only literal 'time' columns, ignoring tuning (timescaledb.py:327-329).",
    ("timescaledb", "sorting"): "compress_orderby never set from tuning (timescaledb.py:379-393).",
    ("timescaledb", "distribution"): "compress_segmentby hardcoded empty (timescaledb.py:379-393).",
    ("timescaledb", "clustering"): "Merged into hardcoded-empty compress_segmentby (timescaledb.py:379-393).",
    ("pg-duckdb", "partitioning"): "Inherits PostgreSQLAdapter no-ops; adapter defines no overrides (pg_duckdb.py).",
    (
        "pg-duckdb",
        "clustering",
    ): "Inherits PostgreSQLAdapter no-ops; CLUSTER filtered from preview (pg_duckdb.py).",
    ("pg-duckdb", "sorting"): "Inherits PostgreSQLAdapter no-ops; CREATE INDEX preview-only (pg_duckdb.py).",
    (
        "pg-mooncake",
        "partitioning",
    ): "Generator always empty by design; tuning promoted away (pg_mooncake.py).",
    (
        "pg-mooncake",
        "clustering",
    ): "Generator always empty by design; tuning promoted away (pg_mooncake.py).",
    (
        "pg-mooncake",
        "sorting",
    ): "Generator always empty by design; tuning promoted away (pg_mooncake.py).",
    (
        "postgresql",
        "partitioning",
    ): "Preview generator uncalled; generate_tuning_clause returns '' (postgresql.py:968-981).",
    (
        "postgresql",
        "clustering",
    ): "Preview generator uncalled; generate_tuning_clause returns '' (postgresql.py:968-981).",
    ("cedardb", "partitioning"): "Inherits PostgreSQLAdapter no-ops with no DDL override (cedardb.py).",
    ("cedardb", "clustering"): "Inherits PostgreSQLAdapter no-ops with no DDL override (cedardb.py).",
    ("citus", "partitioning"): "Inherits PostgreSQLAdapter no-ops with no DDL override (citus.py).",
    ("citus", "clustering"): "Inherits PostgreSQLAdapter no-ops with no DDL override (citus.py).",
    ("paradedb", "partitioning"): "Inherits PostgreSQLAdapter no-ops with no DDL override (paradedb.py).",
    ("paradedb", "clustering"): "Inherits PostgreSQLAdapter no-ops with no DDL override (paradedb.py).",
    ("doris", "partitioning"): "NoOpTableTuningMixin; preview generator uncalled (doris.py:187).",
    ("doris", "sorting"): "NoOpTableTuningMixin; preview generator uncalled (doris.py:187).",
    ("doris", "distribution"): "NoOpTableTuningMixin; preview generator uncalled (doris.py:187).",
    (
        "databend",
        "partitioning",
    ): "Only CLUSTERING is rendered; generate/apply ignore other types (databend/adapter.py:612-651).",
    (
        "databend",
        "sorting",
    ): "Only CLUSTERING is rendered; generate/apply ignore other types (databend/adapter.py:612-651).",
    (
        "databend",
        "distribution",
    ): "Only CLUSTERING is rendered; generate/apply ignore other types (databend/adapter.py:612-651).",
    (
        "databend",
        "materialized_views",
    ): "Only CLUSTERING is rendered; materialized views never produced or consumed from tuning (databend/adapter.py).",
    ("redshift", "partitioning"): "No CREATE TABLE partition clause; recreation never happens (redshift.py:2248).",
    ("redshift", "distribution"): "Recreation never happens; maintenance only (redshift.py:2248).",
    ("bigquery", "partitioning"): "Recreation hint only; table never re-created (bigquery.py:2428).",
    ("bigquery", "clustering"): "Recreation hint only; table never re-created (bigquery.py:2428).",
    ("databricks", "distribution"): "No user-managed distribution key; distribution columns "
    "feed ZORDER locality via _build_zorder_columns (databricks/adapter.py).",
    ("databricks", "partitioning"): "DatabricksAdapter.generate_tuning_clause builds PARTITIONED BY "
    "from tuned columns (databricks/adapter.py:3005-3048) but has no production call site tree-wide; "
    "execution builds Delta DDL without partitioning columns (_convert_to_delta_table at "
    "databricks/adapter.py:2672-2729) and only logs them (apply_table_tunings at "
    "databricks/adapter.py:3090).",
    ("databricks", "bloom_filters"): "DatabricksAdapter executes tuning only through ZORDER, OPTIMIZE, "
    "Liquid and CLUSTER statements (_apply_zorder_optimization, _apply_delta_optimize, "
    "_apply_liquid_clustering in databricks/adapter.py); no Bloom-index statement is constructed.",
    ("databricks", "materialized_views"): "DatabricksAdapter builds only Delta CREATE TABLE with "
    "TBLPROPERTIES plus post-load OPTIMIZE/ALTER statements (databricks/adapter.py); no "
    "materialized-view creation exists in its lifecycle.",
    ("snowflake", "materialized_views"): "SnowflakeAdapter tuning executes ALTER TABLE ... CLUSTER BY "
    "and constraint-bearing CREATE TABLE only (snowflake.py); no materialized-view creation exists "
    "in its lifecycle.",
    ("bigquery", "materialized_views"): "BigQueryAdapter executes client.query DDL plus recreation-hint "
    "logging only (bigquery.py); no materialized-view creation exists in its lifecycle.",
    ("redshift", "materialized_views"): "RedshiftAdapter tuning executes CTAS-sort, VACUUM and "
    "maintenance statements only (redshift.py); no materialized-view creation exists in its lifecycle.",
    ("postgresql", "bloom_filters"): "PostgreSQLAdapter executes constraint-bearing CREATE TABLE with "
    "no-op apply paths only (postgresql.py); no Bloom-filter creation exists in its lifecycle.",
    ("postgresql", "materialized_views"): "PostgreSQLAdapter executes constraint-bearing CREATE TABLE "
    "with no-op apply paths only (postgresql.py); no materialized-view creation exists in its "
    "lifecycle.",
    ("cedardb", "bloom_filters"): "Inherits PostgreSQLAdapter no-ops with no DDL override (cedardb.py).",
    ("cedardb", "materialized_views"): "Inherits PostgreSQLAdapter no-ops with no DDL override (cedardb.py).",
    ("citus", "bloom_filters"): "Inherits PostgreSQLAdapter no-ops with no DDL override (citus.py).",
    ("citus", "materialized_views"): "Inherits PostgreSQLAdapter no-ops with no DDL override (citus.py).",
    ("paradedb", "bloom_filters"): "Inherits PostgreSQLAdapter no-ops with no DDL override (paradedb.py).",
    ("paradedb", "materialized_views"): "Inherits PostgreSQLAdapter no-ops with no DDL override (paradedb.py).",
    ("duckdb", "partitioning"): "The DuckDB generator logs a COPY TO hint only; no lifecycle path "
    "applies partitioning (generators/duckdb.py).",
    ("clickhouse", "materialized_views"): "ClickHouseAdapter renders ORDER BY/partition DDL and "
    "executes statements only (clickhouse_server.py); no materialized-view creation exists in its "
    "lifecycle.",
    ("clickhouse", "unique_constraints"): "ClickHouse has no UNIQUE constraint syntax; the workload "
    "selects DUPLICATE versus PRIMARY KEY models instead (clickhouse/workload.py). Vacuity pinned "
    "tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("motherduck", "partitioning"): "tuning_config dropped at create_schema (motherduck.py:307-318).",
    ("motherduck", "clustering"): "No post-create CLUSTER step consumes tuning (motherduck.py).",
    ("motherduck", "distribution"): "Single-writer cloud DuckDB; nothing renders distribution columns (motherduck.py).",
    ("motherduck", "materialized_views"): "MotherDuckAdapter executes tuning-aware DDL with CTAS-sort "
    "only (motherduck.py); no materialized-view creation exists in its lifecycle.",
    ("motherduck", "primary_keys"): "tuning_config dropped; applier pass-through no-op (motherduck.py:354-355).",
    ("motherduck", "foreign_keys"): "tuning_config dropped; applier pass-through no-op (motherduck.py:354-355).",
    ("motherduck", "unique_constraints"): "tuning_config dropped; applier pass-through no-op (motherduck.py:354-355). "
    "Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("motherduck", "check_constraints"): "tuning_config dropped; applier pass-through no-op (motherduck.py:354-355). "
    "Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("datafusion", "primary_keys"): "Schema built explicitly without constraints (datafusion.py:558-564).",
    ("datafusion", "foreign_keys"): "Schema built explicitly without constraints (datafusion.py:558-564).",
    ("datafusion", "unique_constraints"): "Schema built explicitly without constraints (datafusion.py:558-564). "
    "Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("datafusion", "check_constraints"): "Schema built explicitly without constraints (datafusion.py:558-564). "
    "Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("doris", "primary_keys"): "Adapter strips PRIMARY KEY clauses from every CREATE TABLE (doris.py:803-805).",
    ("doris", "foreign_keys"): "Adapter strips foreign keys from every CREATE TABLE (doris.py:802).",
    ("databend", "foreign_keys"): "Adapter strips foreign keys from every CREATE TABLE (databend/adapter.py:680).",
    ("databend", "primary_keys"): "Adapter strips PRIMARY KEY clauses via inline regex (databend/adapter.py:678).",
    ("doris", "unique_constraints"): "NoOp adapter; UNIQUE never produced or consumed from tuning (doris.py). "
    "Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("doris", "check_constraints"): "NoOp adapter; CHECK never produced or consumed from tuning (doris.py). "
    "Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("databend", "unique_constraints"): "Only CLUSTERING renders; UNIQUE never produced or consumed from tuning "
    "(databend/adapter.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("databend", "check_constraints"): "Only CLUSTERING renders; CHECK never produced or consumed from tuning "
    "(databend/adapter.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    (
        "starrocks",
        "unique_constraints",
    ): "Workload rewrite handles PARTITION/DISTRIBUTION/ORDER BY plus PK-model only "
    "(starrocks/workload.py:109-198). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    (
        "starrocks",
        "check_constraints",
    ): "Workload rewrite handles PARTITION/DISTRIBUTION/ORDER BY plus PK-model only "
    "(starrocks/workload.py:109-198). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    (
        "mysql",
        "partitioning",
    ): "Mysql-wire shared tuning-aware DDL path renders constraints; partitioning unrendered "
    "(NoOpTableTuningMixin, base/mysql_wire.py:514-522). Checked against SingleStoreAdapter, the "
    "concrete mysql-wire representative (no dedicated MySQL class exists).",
    ("singlestore", "partitioning"): "NoOpTableTuningMixin: generate returns '', apply is a no-op (singlestore.py:75).",
    (
        "fabric-warehouse",
        "partitioning",
    ): "T-SQL execution ignores tuned partition columns (fabric_warehouse.py:857-883).",
    ("duckdb", "unique_constraints"): "DuckDBAdapter.apply_constraint_configuration only logs UNIQUE flag "
    "state (duckdb.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("duckdb", "check_constraints"): "DuckDBAdapter.apply_constraint_configuration only logs CHECK flag "
    "state (duckdb.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("databricks", "unique_constraints"): "DatabricksAdapter executes tuning only through ZORDER, OPTIMIZE, "
    "Liquid and CLUSTER statements (databricks/adapter.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("databricks", "check_constraints"): "DatabricksAdapter executes tuning only through ZORDER, OPTIMIZE, "
    "Liquid and CLUSTER statements (databricks/adapter.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("snowflake", "unique_constraints"): "SnowflakeAdapter tuning executes ALTER TABLE ... CLUSTER BY and "
    "constraint-bearing CREATE TABLE carrying PK/FK only (snowflake.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("snowflake", "check_constraints"): "SnowflakeAdapter tuning executes ALTER TABLE ... CLUSTER BY and "
    "constraint-bearing CREATE TABLE carrying PK/FK only (snowflake.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("bigquery", "unique_constraints"): "BigQuery DDL emits PRIMARY KEY (...) NOT ENFORCED plus FOREIGN KEY "
    "clauses only (bigquery.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("bigquery", "check_constraints"): "BigQuery DDL emits PRIMARY KEY (...) NOT ENFORCED plus FOREIGN KEY "
    "clauses only (bigquery.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("sqlite", "unique_constraints"): "The shared tuning-aware DDL path forwards tuning_config; SQLite "
    "execution builds no UNIQUE clause from the flags (base/data_loading.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("sqlite", "check_constraints"): "The shared tuning-aware DDL path forwards tuning_config; SQLite "
    "execution builds no CHECK clause from the flags (base/data_loading.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("postgresql", "unique_constraints"): "PostgreSQLAdapter renders constraint-bearing CREATE TABLE from "
    "PK/FK flags only (postgresql.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("postgresql", "check_constraints"): "PostgreSQLAdapter renders constraint-bearing CREATE TABLE from "
    "PK/FK flags only (postgresql.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("mysql", "unique_constraints"): "The mysql-wire tuning-aware DDL path renders PK/FK from flags only "
    "(base/mysql_wire.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("mysql", "check_constraints"): "The mysql-wire tuning-aware DDL path renders PK/FK from flags only "
    "(base/mysql_wire.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("singlestore", "unique_constraints"): "SingleStoreAdapter inherits the mysql-wire tuning-aware DDL path "
    "rendering PK from flags only (singlestore.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("singlestore", "check_constraints"): "SingleStoreAdapter inherits the mysql-wire tuning-aware DDL path "
    "rendering PK from flags only (singlestore.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("cedardb", "unique_constraints"): "CedarDBAdapter inherits PostgreSQLAdapter execution rendering PK/FK "
    "from flags only (cedardb.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("cedardb", "check_constraints"): "CedarDBAdapter inherits PostgreSQLAdapter execution rendering PK/FK "
    "from flags only (cedardb.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("citus", "unique_constraints"): "CitusAdapter inherits PostgreSQLAdapter execution rendering PK/FK from "
    "flags only (citus.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("citus", "check_constraints"): "CitusAdapter inherits PostgreSQLAdapter execution rendering PK/FK from "
    "flags only (citus.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("paradedb", "unique_constraints"): "ParadeDBAdapter inherits PostgreSQLAdapter execution rendering PK/FK "
    "from flags only (paradedb.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("paradedb", "check_constraints"): "ParadeDBAdapter inherits PostgreSQLAdapter execution rendering PK/FK "
    "from flags only (paradedb.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("fabric-warehouse", "unique_constraints"): "FabricWarehouseAdapter executes T-SQL carrying PK/FK only "
    "(fabric_warehouse.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("fabric-warehouse", "check_constraints"): "FabricWarehouseAdapter executes T-SQL carrying PK/FK only "
    "(fabric_warehouse.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("athena", "unique_constraints"): "AthenaAdapter._convert_to_external_table builds CREATE EXTERNAL TABLE "
    "without UNIQUE clauses (athena.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("athena", "check_constraints"): "AthenaAdapter._convert_to_external_table builds CREATE EXTERNAL TABLE "
    "without CHECK clauses (athena.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("azure-synapse", "unique_constraints"): "AzureSynapseAdapter real DDL carries PK/FK with the engine-default "
    "DISTRIBUTION only (azure_synapse.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("azure-synapse", "check_constraints"): "AzureSynapseAdapter real DDL carries PK/FK with the engine-default "
    "DISTRIBUTION only (azure_synapse.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("timescaledb", "unique_constraints"): "TimescaleDBAdapter hypertable and compression paths consume no "
    "UNIQUE tuning (timescaledb.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
    ("timescaledb", "check_constraints"): "TimescaleDBAdapter hypertable and compression paths consume no CHECK "
    "tuning (timescaledb.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("redshift", "unique_constraints"): "RedshiftAdapter tuning executes CTAS-sort, VACUUM and maintenance "
    "statements only (redshift.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("redshift", "check_constraints"): "RedshiftAdapter tuning executes CTAS-sort, VACUUM and maintenance "
    "statements only (redshift.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("questdb", "unique_constraints"): "QuestDB schema enhancement maps hardcoded timestamp/partition columns "
    "only (questdb.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("questdb", "check_constraints"): "QuestDB schema enhancement maps hardcoded timestamp/partition columns "
    "only (questdb.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("questdb", "primary_keys"): "QuestDBAdapter.create_schema strips PRIMARY KEY clauses via "
    "_strip_pk_constraints (questdb.py); pinned by test_questdb_create_schema_strips_constraints.",
    ("questdb", "foreign_keys"): "QuestDBAdapter.create_schema strips FOREIGN KEY clauses via ddl_helpers "
    "strip_foreign_keys (questdb.py); pinned by test_questdb_create_schema_strips_constraints.",
    ("pg-duckdb", "unique_constraints"): "PgDuckDBAdapter inherits PostgreSQLAdapter execution carrying PK/FK "
    "only (pg_duckdb.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("pg-duckdb", "check_constraints"): "PgDuckDBAdapter inherits PostgreSQLAdapter execution carrying PK/FK "
    "only (pg_duckdb.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("pg-mooncake", "unique_constraints"): "PgMooncakeAdapter inherits PostgreSQLAdapter execution carrying "
    "PK/FK only (pg_mooncake.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("pg-mooncake", "check_constraints"): "PgMooncakeAdapter inherits PostgreSQLAdapter execution carrying "
    "PK/FK only (pg_mooncake.py). Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("trino", "unique_constraints"): "TrinoAdapter renders no UNIQUE clause at execution (trino.py). Vacuity "
    "pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("trino", "check_constraints"): "TrinoAdapter renders no CHECK clause at execution (trino.py). Vacuity "
    "pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("trino", "primary_keys"): "PrestoTrinoAdapterBase._optimize_table_definition strips PRIMARY KEY clauses "
    "from every CREATE TABLE (presto_trino_adapter_base.py); pinned by "
    "test_trino_base_optimize_strips_primary_keys.",
    ("presto", "unique_constraints"): "PrestoAdapter renders no UNIQUE clause at execution (presto.py). Vacuity "
    "pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("presto", "check_constraints"): "PrestoAdapter renders no CHECK clause at execution (presto.py). Vacuity "
    "pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("presto", "primary_keys"): "PrestoTrinoAdapterBase._optimize_table_definition strips PRIMARY KEY clauses "
    "from every CREATE TABLE (presto_trino_adapter_base.py); pinned by "
    "test_trino_base_optimize_strips_primary_keys.",
    ("firebolt", "primary_keys"): "FireboltAdapter._optimize_table_definition strips PRIMARY KEY clauses from "
    "every CREATE TABLE (firebolt.py).",
    ("firebolt", "foreign_keys"): "FireboltAdapter._optimize_table_definition strips FOREIGN KEY clauses from "
    "every CREATE TABLE (firebolt.py).",
    ("firebolt", "unique_constraints"): "FireboltAdapter renders no UNIQUE clause at execution (firebolt.py). "
    "Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("firebolt", "check_constraints"): "FireboltAdapter renders no CHECK clause at execution (firebolt.py). "
    "Vacuity pinned tree-wide by test_unique_check_rendering_is_vacuous_in_tree.",
    ("clickhouse", "foreign_keys"): "ClickHouseWorkloadMixin._optimize_table_definition preserves FOREIGN KEY "
    "clauses the engine rejects (clickhouse/workload.py); pinned by "
    "test_clickhouse_optimize_preserves_foreign_keys.",
    ("clickhouse", "check_constraints"): "The ClickHouse workload column parser terminates column types at "
    "CHECK keywords without rendering them (clickhouse/workload.py). Vacuity pinned tree-wide by "
    "test_unique_check_rendering_is_vacuous_in_tree.",
}


_BEHAVIORAL_STRIP_TESTS: dict[tuple[str, str], str] = {
    ("questdb", "primary_keys"): "test_questdb_create_schema_strips_constraints",
    ("questdb", "foreign_keys"): "test_questdb_create_schema_strips_constraints",
    ("trino", "primary_keys"): "test_trino_base_optimize_strips_primary_keys",
    ("presto", "primary_keys"): "test_trino_base_optimize_strips_primary_keys",
    ("clickhouse", "foreign_keys"): "test_clickhouse_optimize_preserves_foreign_keys",
    ("azure-synapse", "distribution"): "test_azure_synapse_optimize_uses_engine_default",
}


_NO_TUNING_INPUT_TESTS: dict[tuple[str, str], str] = {
    ("athena", "partitioning"): "test_athena_rewrite_takes_no_tuning",
    ("athena", "sorting"): "test_athena_rewrite_takes_no_tuning",
    ("athena", "clustering"): "test_athena_rewrite_takes_no_tuning",
    ("questdb", "partitioning"): "test_questdb_enhancement_takes_no_tuning",
}


_SCHEMA_IGNORES_CONSTRAINTS_TESTS: dict[tuple[str, str], str] = {
    ("datafusion", "primary_keys"): "test_datafusion_schema_ignores_constraints",
    ("datafusion", "foreign_keys"): "test_datafusion_schema_ignores_constraints",
}


_DROPS_TUNING_TESTS: dict[tuple[str, str], str] = {
    ("motherduck", "primary_keys"): "test_motherduck_create_schema_drops_tuning",
    ("motherduck", "foreign_keys"): "test_motherduck_create_schema_drops_tuning",
    ("motherduck", "clustering"): "test_motherduck_create_schema_drops_tuning",
}


_IGNORES_TUNING_METHODS: dict[tuple[str, str], str] = {
    ("timescaledb", "sorting"): "test_timescaledb_compression_ignores_tuning",
    ("timescaledb", "clustering"): "test_timescaledb_compression_ignores_tuning",
}


_GENERATOR_QUIET_FIELDS: dict[tuple[str, str], str] = {
    ("databricks", "distribution"): "distribute_by",
    ("firebolt", "sorting"): "sort_by",
    ("firebolt", "clustering"): "cluster_by",
    ("questdb", "sorting"): "sort_by",
    ("questdb", "distribution"): "distribute_by",
    ("azure-synapse", "sorting"): "sort_by",
    ("azure-synapse", "clustering"): "cluster_by",
}


_DATABEND_QUIET_KINDS: frozenset[str] = frozenset({"partitioning", "sorting", "distribution"})


_LAYOUT_QUIET: frozenset[tuple[str, str]] = frozenset(
    {
        ("athena", "distribution"),
        ("duckdb", "partitioning"),
        ("fabric-warehouse", "partitioning"),
        ("motherduck", "partitioning"),
        ("motherduck", "distribution"),
        ("azure-synapse", "partitioning"),
        ("redshift", "partitioning"),
        ("timescaledb", "partitioning"),
        ("timescaledb", "distribution"),
    }
)


_VACUITY_TEST_NAME = "test_unique_check_rendering_is_vacuous_in_tree"


_UNCALLED_GENERATOR_KEYS = frozenset(
    {
        ("trino", "partitioning"),
        ("trino", "distribution"),
        ("trino", "sorting"),
        ("trino", "clustering"),
        ("presto", "partitioning"),
        ("presto", "distribution"),
        ("presto", "sorting"),
        ("presto", "clustering"),
        ("firebolt", "partitioning"),
        ("firebolt", "distribution"),
        ("doris", "partitioning"),
        ("doris", "sorting"),
        ("doris", "distribution"),
        ("bigquery", "partitioning"),
        ("bigquery", "clustering"),
        ("redshift", "partitioning"),
        ("redshift", "distribution"),
        ("databricks", "partitioning"),
        ("azure-synapse", "partitioning"),
        ("azure-synapse", "distribution"),
    }
)

_RETURNS_EMPTY_GENERATOR_KEYS = frozenset(
    {
        ("postgresql", "partitioning"),
        ("postgresql", "clustering"),
        ("cedardb", "partitioning"),
        ("cedardb", "clustering"),
        ("citus", "partitioning"),
        ("citus", "clustering"),
        ("paradedb", "partitioning"),
        ("paradedb", "clustering"),
        ("pg-duckdb", "partitioning"),
        ("pg-duckdb", "clustering"),
        ("pg-duckdb", "sorting"),
        ("pg-mooncake", "partitioning"),
        ("pg-mooncake", "clustering"),
        ("pg-mooncake", "sorting"),
        ("singlestore", "partitioning"),
        ("mysql", "partitioning"),
    }
)


def _mro_method_node(adapter_cls: type, name: str) -> ast.FunctionDef | None:
    mro, definers = _collect_definers(adapter_cls)
    for klass in mro:
        for owner, node in definers.get(name, []):
            if owner == klass.__name__:
                return node
    return None


def _lifecycle_generator_calls(adapter_cls: type) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for body in _reachable_bodies(adapter_cls):
        called = sorted(
            {
                child.func.attr
                for child in ast.walk(body)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in ("generate_tuning_clause", "generate_tuning_clauses")
            }
        )
        if called:
            found.setdefault(body.name, []).extend(called)
    return found


_VACUITY_SIGNATURE_ONLY: dict[tuple[str, str], str] = {
    ("tpcds_obt", "TPCDSOBT"): "get_create_tables_sql accepts no tuning_config",
    ("core.joinorder.benchmark", "JoinOrderSchema"): "schema helper; get_create_tables_sql accepts no tuning_config",
}


_DISCOVERY_IMPORT_SKIPS: dict[str, str] = {}


def _benchmark_classes_in(module: object, module_key: str, seen: set[int]) -> list[tuple[str, str, type]]:
    found: list[tuple[str, str, type]] = []
    for attribute in sorted(vars(module)):
        value = vars(module)[attribute]
        if (
            inspect.isclass(value)
            and hasattr(value, "get_create_tables_sql")
            and value.__module__.split(".")[0] == "benchbox"
            and id(value) not in seen
        ):
            seen.add(id(value))
            found.append((module_key, attribute, value))
    return found


def _discovered_benchmarks() -> list[tuple[str, str, type]]:
    import importlib
    import pkgutil

    import benchbox
    import benchbox.core

    found: list[tuple[str, str, type]] = []
    seen: set[int] = set()
    for module in sorted(pkgutil.iter_modules(benchbox.__path__), key=lambda entry: entry.name):
        imported = importlib.import_module(f"benchbox.{module.name}")
        found.extend(_benchmark_classes_in(imported, module.name, seen))
    failures: dict[str, str] = {}
    for module in sorted(pkgutil.walk_packages(benchbox.core.__path__, "benchbox.core."), key=lambda e: e.name):
        if module.name in _DISCOVERY_IMPORT_SKIPS:
            continue
        try:
            imported = importlib.import_module(module.name)
        except ImportError as exc:
            failures[module.name] = str(exc)
            continue
        found.extend(_benchmark_classes_in(imported, module.name.removeprefix("benchbox."), seen))
    assert not failures, f"benchbox.core modules failed to import; fix or list in _DISCOVERY_IMPORT_SKIPS: {failures}"
    return found


def _vacuous_capable_benchmarks() -> list[tuple[str, str, object]]:
    capable: list[tuple[str, str, object]] = []
    for module_name, class_name, benchmark_cls in _discovered_benchmarks():
        if (module_name, class_name) in _VACUITY_SIGNATURE_ONLY:
            parameters = inspect.signature(benchmark_cls.get_create_tables_sql).parameters
            assert "tuning_config" not in parameters, (
                f"{module_name}.{class_name} now accepts tuning_config; cover it behaviorally"
            )
            continue
        try:
            benchmark = benchmark_cls()
        except Exception as exc:
            pytest.fail(
                f"{module_name}.{class_name} needs explicit construction ({exc}); construct it in "
                "_vacuous_capable_benchmarks or classify it in _VACUITY_SIGNATURE_ONLY"
            )
        parameters = inspect.signature(benchmark.get_create_tables_sql).parameters
        assert "tuning_config" in parameters, (
            f"{module_name}.{class_name} accepts no tuning_config; classify it in _VACUITY_SIGNATURE_ONLY"
        )
        capable.append((module_name, class_name, benchmark))
    assert capable, "no behaviorally covered benchmarks; the vacuity proof is empty"
    return capable


_PARITY_ALIAS_TARGETS: dict[str, str] = {"starburst": "trino"}


def _parity_targets() -> list[tuple[str, str]]:
    targets = [
        (platform_key, platform_key) for platform_key in sorted(known_registry_platforms() - _SKIP_NO_ADAPTER_CLASS)
    ]
    targets.extend(sorted(_PARITY_ALIAS_TARGETS.items()))
    return targets


class TestRegistryExecutionParity:
    @pytest.mark.parametrize("adapter_key,registry_key", _parity_targets())
    def test_rendered_entries_have_executing_implementation(self, adapter_key: str, registry_key: str) -> None:
        adapter_cls = _adapter_class(resolve_platform_key(adapter_key))
        assert adapter_cls is not None, f"no adapter class mapped for registry platform '{adapter_key}'"
        bodies = _reachable_bodies(adapter_cls, _PLATFORM_SKIP_MODULES.get(adapter_key, frozenset()))
        assert bodies, f"no lifecycle roots reachable on {adapter_cls.__name__}"
        tuning_read, sql_exec, ddl_handoff, strip_pk, strip_fk, texts = _signals(bodies)
        for tuning_type, capability in PLATFORM_TUNING_CAPABILITIES[registry_key].items():
            if is_not_rendered_at_execution(capability):
                continue
            key = (registry_key, tuning_type.value)
            evidenced = _type_evidenced(tuning_type.value, texts, ddl_handoff, strip_pk, strip_fk)
            assert _rendered_gate_passes(tuning_read, sql_exec, evidenced, key in _RENDERED_OVERRIDE_JUSTIFICATIONS), (
                f"{adapter_key}/{tuning_type.value} is registered '{capability.rendered_via}' "
                f"({capability.mechanism_id}) but the {adapter_cls.__name__} lifecycle shows "
                f"tuning_read={tuning_read} sql_exec={sql_exec} type_evidenced={evidenced}; "
                f"add execution or correct the registry"
            )

    @pytest.mark.parametrize("adapter_key,registry_key", _parity_targets())
    def test_noop_entries_are_justified(self, adapter_key: str, registry_key: str) -> None:
        adapter_cls = _adapter_class(resolve_platform_key(adapter_key))
        assert adapter_cls is not None, f"no adapter class mapped for registry platform '{adapter_key}'"
        bodies = _reachable_bodies(adapter_cls, _PLATFORM_SKIP_MODULES.get(adapter_key, frozenset()))
        _tuning_read, _sql_exec, _ddl_handoff, strip_pk, strip_fk, _texts = _signals(bodies)
        reg_pk, reg_fk = _registered_kind_strips(registry_key)
        for tuning_type, capability in PLATFORM_TUNING_CAPABILITIES[registry_key].items():
            if not is_not_rendered_at_execution(capability):
                continue
            key = (registry_key, tuning_type.value)
            assert _noop_gate_excused(
                tuning_type.value, strip_pk, strip_fk, reg_pk, reg_fk, key in _NONE_JUSTIFICATION_OVERRIDES
            ), (
                f"{adapter_key}/{tuning_type.value} is registered not-rendered but the "
                f"{adapter_cls.__name__} lifecycle shows no strip, no applied DDL-rewrite kind strip, "
                f"and no documented justification; document it in _NONE_JUSTIFICATION_OVERRIDES or "
                f"correct the registry"
            )

    def test_alias_adapters_are_parity_targets(self) -> None:
        for alias, target in PLATFORM_ALIASES.items():
            if target not in known_registry_platforms():
                continue
            alias_cls = _adapter_class(resolve_platform_key(alias))
            target_cls = _adapter_class(resolve_platform_key(target))
            if alias_cls is None or target_cls is None or alias_cls is target_cls:
                continue
            assert _PARITY_ALIAS_TARGETS.get(alias) == target, (
                f"alias '{alias}' resolves to a distinct adapter class ({alias_cls.__name__} vs "
                f"{target_cls.__name__}) but has no parity target; add it to _PARITY_ALIAS_TARGETS"
            )

    def test_rendered_gate_rejects_signal_only_claims(self) -> None:
        assert not _rendered_gate_passes(True, True, False, False)
        assert not _rendered_gate_passes(True, False, True, False)
        assert not _rendered_gate_passes(False, True, True, False)
        assert _rendered_gate_passes(True, True, True, False)
        assert _rendered_gate_passes(False, False, False, True)

    def test_noop_gate_requires_matching_kind(self) -> None:
        assert _noop_gate_excused("primary_keys", True, False, False, False, False)
        assert _noop_gate_excused("primary_keys", False, False, True, False, False)
        assert _noop_gate_excused("foreign_keys", False, True, False, False, False)
        assert _noop_gate_excused("foreign_keys", False, False, False, True, False)
        assert not _noop_gate_excused("partitioning", True, True, True, True, False)
        assert not _noop_gate_excused("primary_keys", False, True, False, True, False)
        assert not _noop_gate_excused("foreign_keys", True, False, True, False, False)
        assert _noop_gate_excused("partitioning", False, False, False, False, True)

    def test_unique_check_rendering_is_vacuous_in_tree(self) -> None:
        for module_name, class_name, benchmark in _vacuous_capable_benchmarks():
            plain = UnifiedTuningConfiguration()
            plain.disable_all_constraints()
            toggled = UnifiedTuningConfiguration()
            toggled.disable_all_constraints()
            toggled.unique_constraints.enabled = True
            toggled.check_constraints.enabled = True
            assert benchmark.get_create_tables_sql(tuning_config=plain) == benchmark.get_create_tables_sql(
                tuning_config=toggled
            ), (
                f"{module_name}.{class_name} DDL now varies with UNIQUE/CHECK flags; the not-rendered "
                "UNIQUE/CHECK registry entries must be re-audited against execution"
            )

    def test_benchmark_ddl_ignores_layout_flags_in_tree(self) -> None:
        for module_name, class_name, benchmark in _vacuous_capable_benchmarks():
            plain = UnifiedTuningConfiguration()
            plain.disable_all_constraints()
            toggled = UnifiedTuningConfiguration()
            toggled.disable_all_constraints()
            toggled.table_tunings["orders"] = TableTuning(
                table_name="orders",
                partitioning=[TuningColumn(name="o_orderdate", type="DATE", order=1)],
                sorting=[TuningColumn(name="o_orderkey", type="INTEGER", order=1)],
                clustering=[TuningColumn(name="o_orderkey", type="INTEGER", order=1)],
                distribution=[TuningColumn(name="o_orderkey", type="INTEGER", order=1)],
            )
            assert benchmark.get_create_tables_sql(tuning_config=plain) == benchmark.get_create_tables_sql(
                tuning_config=toggled
            ), (
                f"{module_name}.{class_name} DDL now varies with layout tuning flags; the not-rendered "
                "layout registry entries must be re-audited against execution"
            )

    def test_vacuity_discovery_covers_all_benchmarks(self) -> None:
        found = {(module_name, class_name) for module_name, class_name, _ in _discovered_benchmarks()}
        assert found, "benchmark discovery is empty; the vacuity proof covers nothing"
        known = {
            ("tpch", "TPCH"),
            ("tpcds", "TPCDS"),
            ("ssb", "SSB"),
            ("clickbench", "ClickBench"),
            ("core.ai_primitives", "AIPrimitivesBenchmark"),
        }
        assert known <= found, f"discovery lost known benchmarks: {sorted(known - found)}"
        for key in _VACUITY_SIGNATURE_ONLY:
            assert key in found, f"signature-only benchmark {key} disappeared from discovery"
        for module_name, class_name in sorted(found - set(_VACUITY_SIGNATURE_ONLY)):
            assert any(
                module_name == seen_module and class_name == seen_class
                for seen_module, seen_class, _ in _vacuous_capable_benchmarks()
            ), (
                f"{module_name}.{class_name} is not covered by the vacuity proof; construct it in "
                "_vacuous_capable_benchmarks or classify it in _VACUITY_SIGNATURE_ONLY"
            )

    def test_rendered_overrides_cover_only_current_entries(self) -> None:
        current = {
            (platform_key, tuning_type.value)
            for platform_key, entries in PLATFORM_TUNING_CAPABILITIES.items()
            for tuning_type, capability in entries.items()
            if not is_not_rendered_at_execution(capability)
        }
        assert set(_RENDERED_OVERRIDE_JUSTIFICATIONS) <= current, (
            f"stale rendered justifications: {sorted(set(_RENDERED_OVERRIDE_JUSTIFICATIONS) - current)}"
        )

    def test_justifications_cover_only_current_noop_entries(self) -> None:
        current = {
            (platform_key, tuning_type.value)
            for platform_key, entries in PLATFORM_TUNING_CAPABILITIES.items()
            for tuning_type, capability in entries.items()
            if is_not_rendered_at_execution(capability)
        }
        assert set(_NONE_JUSTIFICATION_OVERRIDES) <= current, (
            f"stale justifications: {sorted(set(_NONE_JUSTIFICATION_OVERRIDES) - current)}"
        )

    def test_absence_claims_hold(self) -> None:
        for (platform_key, tuning_type), (adapter_key, phrases) in _ABSENCE_CLAIMS.items():
            adapter_cls = _adapter_class(resolve_platform_key(adapter_key))
            assert adapter_cls is not None
            texts = _all_adapter_texts(adapter_cls)
            assert texts, f"{platform_key}/{tuning_type}: absence scan corpus is empty"
            assert _absence_holds(texts, phrases), (
                f"{platform_key}/{tuning_type}: absence claim broken, an absence phrase now appears "
                f"in a {adapter_cls.__name__} adapter method; re-audit the registry entry"
            )

    def test_justification_citations_resolve(self) -> None:
        import re

        root = Path(__file__).resolve().parents[4]
        cited = re.compile(r"[\w\-./]+\.py")
        lines = re.compile(r":(\d+)(?:-(\d+))?")
        for key, reason in _NONE_JUSTIFICATION_OVERRIDES.items():
            matches = cited.findall(reason)
            assert matches, f"{key}: justification cites no source file"
            for match in matches:
                candidates = [
                    root / match,
                    root / "benchbox" / match,
                    root / "benchbox" / "platforms" / match,
                    root / "benchbox" / "core" / "tuning" / match,
                    root / "benchbox" / "core" / "tuning" / "generators" / match,
                ]
                path = next((candidate for candidate in candidates if candidate.is_file()), None)
                assert path is not None, f"{key}: cited file does not exist: {match}"
                total = sum(1 for _ in path.open())
                for number in lines.findall(reason):
                    for bound in number:
                        if bound:
                            assert int(bound) <= total, f"{key}: cited line out of range in {match}"

    def test_noop_justifications_are_machine_backed(self) -> None:
        signals_by_platform = {}
        for platform_key in known_registry_platforms():
            adapter_cls = _adapter_class(resolve_platform_key(platform_key))
            assert adapter_cls is not None
            bodies = _reachable_bodies(adapter_cls, _PLATFORM_SKIP_MODULES.get(platform_key, frozenset()))
            _tuning_read, _sql_exec, _ddl_handoff, strip_pk, strip_fk, texts = _signals(bodies)
            reg_pk, reg_fk = _registered_kind_strips(platform_key)
            signals_by_platform[platform_key] = (strip_pk, strip_fk, reg_pk, reg_fk, texts)
        for key, reason in _NONE_JUSTIFICATION_OVERRIDES.items():
            platform_key, tuning_value = key
            assert self._justification_backed(key, reason, signals_by_platform), (
                f"{key}: justification has no machine-checked backing; add the key to a backing set "
                f"(_UNCALLED_GENERATOR_KEYS, _RETURNS_EMPTY_GENERATOR_KEYS, _ABSENCE_CLAIMS, "
                f"_BEHAVIORAL_STRIP_TESTS, _NO_TUNING_INPUT_TESTS, _DROPS_TUNING_TESTS, "
                f"_IGNORES_TUNING_METHODS, _SCHEMA_IGNORES_CONSTRAINTS_TESTS, "
                f"_GENERATOR_QUIET_FIELDS, _DATABEND_QUIET_KINDS, _LAYOUT_QUIET) or correct the registry"
            )

    def _justification_backed(
        self,
        key: tuple[str, str],
        reason: str,
        signals_by_platform: dict[str, tuple[bool, bool, bool, bool, list[str]]],
    ) -> bool:
        platform_key, tuning_value = key
        if key in _UNCALLED_GENERATOR_KEYS or key in _RETURNS_EMPTY_GENERATOR_KEYS:
            return True
        if key in _ABSENCE_CLAIMS:
            return True
        for mapping in (
            _BEHAVIORAL_STRIP_TESTS,
            _NO_TUNING_INPUT_TESTS,
            _DROPS_TUNING_TESTS,
            _SCHEMA_IGNORES_CONSTRAINTS_TESTS,
        ):
            if key in mapping:
                return callable(getattr(type(self), mapping[key], None))
        if platform_key == "databend" and tuning_value in _DATABEND_QUIET_KINDS:
            return callable(getattr(type(self), "test_databend_generate_ignores_non_clustering", None))
        if key in _IGNORES_TUNING_METHODS:
            return callable(getattr(type(self), _IGNORES_TUNING_METHODS[key], None))
        if key in _GENERATOR_QUIET_FIELDS:
            return callable(getattr(type(self), "test_quiet_generators_emit_no_clause", None))
        if tuning_value in ("primary_keys", "foreign_keys"):
            strip_pk, strip_fk, reg_pk, reg_fk, _texts = signals_by_platform[platform_key]
            return _noop_gate_excused(tuning_value, strip_pk, strip_fk, reg_pk, reg_fk, False)
        if tuning_value in ("unique_constraints", "check_constraints"):
            strip_pk, strip_fk, _reg_pk, _reg_fk, texts = signals_by_platform[platform_key]
            kinds = {kind for kind, _ in _applied_kind_strip_rules(platform_key)}
            return (
                _VACUITY_TEST_NAME in reason
                and tuning_value not in kinds
                and not _type_evidenced(tuning_value, texts, False, strip_pk, strip_fk)
            )
        if key in _LAYOUT_QUIET:
            _strip_pk, _strip_fk, _reg_pk, _reg_fk, texts = signals_by_platform[platform_key]
            return not _type_evidenced(tuning_value, texts, False, False, False)
        return False

    def test_questdb_create_schema_strips_constraints(self) -> None:
        from benchbox.platforms.base.ddl_helpers import strip_foreign_keys
        from benchbox.platforms.questdb import QuestDBAdapter

        adapter = QuestDBAdapter.__new__(QuestDBAdapter)
        create = "CREATE TABLE orders (o_orderkey INTEGER, o_custkey INTEGER, PRIMARY KEY (o_orderkey));"
        assert "PRIMARY KEY" not in QuestDBAdapter._strip_pk_constraints(adapter, create)
        assert "o_orderkey INTEGER" in QuestDBAdapter._strip_pk_constraints(adapter, create)
        foreign = "CREATE TABLE orders (o_orderkey INTEGER, FOREIGN KEY (o_orderkey) REFERENCES c (x));"
        assert "FOREIGN KEY" not in strip_foreign_keys(foreign)

    def test_trino_base_optimize_strips_primary_keys(self) -> None:
        from benchbox.platforms.presto_trino_adapter_base import PrestoTrinoAdapterBase

        adapter = PrestoTrinoAdapterBase.__new__(PrestoTrinoAdapterBase)
        adapter.table_format = "hive"
        adapter.catalog = "catalog"
        create = "CREATE TABLE orders (o_orderkey INTEGER, PRIMARY KEY (o_orderkey));"
        assert "PRIMARY KEY" not in PrestoTrinoAdapterBase._optimize_table_definition(adapter, create)
        foreign = "CREATE TABLE orders (o_orderkey INTEGER, FOREIGN KEY (o_orderkey) REFERENCES c (x));"
        assert "FOREIGN KEY" in PrestoTrinoAdapterBase._optimize_table_definition(adapter, foreign)

    def test_clickhouse_optimize_preserves_foreign_keys(self) -> None:
        from benchbox.platforms.clickhouse.workload import ClickHouseWorkloadMixin

        mixin = ClickHouseWorkloadMixin.__new__(ClickHouseWorkloadMixin)
        create = (
            "CREATE TABLE orders (o_orderkey INTEGER, FOREIGN KEY (o_orderkey) REFERENCES c (x)) "
            "ENGINE = MergeTree() ORDER BY tuple();"
        )
        assert "FOREIGN KEY" in ClickHouseWorkloadMixin._optimize_table_definition(mixin, create, None)

    def test_azure_synapse_optimize_uses_engine_default(self) -> None:
        from benchbox.platforms.azure_synapse import AzureSynapseAdapter

        adapter = AzureSynapseAdapter.__new__(AzureSynapseAdapter)
        adapter.schema = "dbo"
        adapter.distribution_default = "ROUND_ROBIN"
        create = "CREATE TABLE orders (o_orderkey INTEGER, tuned_dist INTEGER);"
        rendered = AzureSynapseAdapter._optimize_table_definition(adapter, create)
        assert "DISTRIBUTION = ROUND_ROBIN" in rendered
        assert "tuned_dist" not in rendered.split("WITH (DISTRIBUTION")[1]
        assert "PARTITION" not in rendered.upper()

    def test_athena_rewrite_takes_no_tuning(self) -> None:
        from benchbox.platforms.athena import AthenaAdapter

        parameters = inspect.signature(AthenaAdapter._convert_to_external_table).parameters
        assert "tuning_config" not in parameters and "table_tuning" not in parameters
        node = _mro_method_node(AthenaAdapter, "_convert_to_external_table")
        assert node is not None
        lowered = ast.unparse(node).lower()
        assert "tuning_config" not in lowered and "table_tuning" not in lowered

    def test_questdb_enhancement_takes_no_tuning(self) -> None:
        from benchbox.platforms.questdb import QuestDBAdapter

        parameters = inspect.signature(QuestDBAdapter._apply_questdb_schema_enhancements).parameters
        assert "tuning_config" not in parameters and "table_tuning" not in parameters
        adapter = QuestDBAdapter.__new__(QuestDBAdapter)
        adapter.partition_by = None
        create = "CREATE TABLE lineitem (l_orderkey INTEGER, l_shipdate DATE);"
        rendered = QuestDBAdapter._apply_questdb_schema_enhancements(adapter, create)
        assert "PARTITION BY MONTH" in rendered.upper()

    def test_motherduck_create_schema_drops_tuning(self) -> None:
        from benchbox.platforms.motherduck import MotherDuckAdapter

        node = _mro_method_node(MotherDuckAdapter, "create_schema")
        assert node is not None
        lowered = ast.unparse(node).lower()
        assert "tuning_config" not in lowered
        assert "get_columns_by_type" not in lowered
        reads = [
            ast.unparse(child).lower()
            for body in _reachable_bodies(MotherDuckAdapter)
            for child in ast.walk(body)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "get_columns_by_type"
            and "clustering" in ast.unparse(child).lower()
        ]
        assert not reads, "motherduck CTAS path reads clustering columns"

    def test_timescaledb_compression_ignores_tuning(self) -> None:
        from benchbox.platforms.timescaledb import TimescaleDBAdapter

        node = _mro_method_node(TimescaleDBAdapter, "_add_compression_policy")
        assert node is not None
        lowered = ast.unparse(node).lower()
        assert "tuning_config" not in lowered
        assert "table_tuning" not in lowered
        assert "get_columns_by_type" not in lowered
        assert "compress_orderby" not in lowered

    def test_datafusion_schema_ignores_constraints(self) -> None:
        from benchbox.platforms.datafusion import DataFusionAdapter

        parameters = inspect.signature(DataFusionAdapter._get_benchmark_schema).parameters
        assert "tuning_config" not in parameters and "enable_primary_keys" not in parameters
        assert "enable_foreign_keys" not in parameters
        node = _mro_method_node(DataFusionAdapter, "create_schema")
        assert node is not None
        lowered = ast.unparse(node).lower()
        assert "does not enforce primary key or foreign key constraints" in lowered

    def test_quiet_generators_emit_no_clause(self) -> None:
        from benchbox.core.tuning.ddl_generator import get_ddl_generator

        for (platform_key, tuning_value), field in sorted(_GENERATOR_QUIET_FIELDS.items()):
            tuning: dict[str, list[TuningColumn]] = {
                tuning_value: [TuningColumn(name="tuned_column", type="INTEGER", order=1)]
            }
            clauses = get_ddl_generator(platform_key).generate_tuning_clauses(TableTuning(table_name="t", **tuning))
            assert getattr(clauses, field) is None, (
                f"{platform_key}/{tuning_value} generator emits a {field} clause from tuned columns"
            )

    def test_databend_generate_ignores_non_clustering(self) -> None:
        from benchbox.platforms.databend.adapter import DatabendAdapter

        for tuning_value in sorted(_DATABEND_QUIET_KINDS):
            tuning: dict[str, list[TuningColumn]] = {
                tuning_value: [TuningColumn(name="tuned_column", type="INTEGER", order=1)]
            }
            assert DatabendAdapter.generate_tuning_clause(None, TableTuning(table_name="t", **tuning)) == ""

    def test_skip_set_is_empty(self) -> None:
        assert frozenset() == _SKIP_NO_ADAPTER_CLASS

    def test_preview_generators_uncalled(self) -> None:
        for platform_key, tuning_value in sorted(_UNCALLED_GENERATOR_KEYS):
            adapter_cls = _adapter_class(resolve_platform_key(platform_key))
            assert adapter_cls is not None
            calls = _lifecycle_generator_calls(adapter_cls)
            assert not calls, (
                f"{platform_key}/{tuning_value} is justified as preview-generator-uncalled but the "
                f"{adapter_cls.__name__} lifecycle calls tuning generators: {calls}"
            )

    def test_empty_generators_return_empty(self) -> None:
        for platform_key, tuning_value in sorted(_RETURNS_EMPTY_GENERATOR_KEYS):
            adapter_cls = _adapter_class(resolve_platform_key(platform_key))
            assert adapter_cls is not None
            node = _mro_method_node(adapter_cls, "generate_tuning_clause")
            assert node is not None, f"{platform_key}: generate_tuning_clause missing from adapter MRO"
            non_empty = [
                ast.unparse(stmt.value) if stmt.value is not None else "bare return"
                for stmt in ast.walk(node)
                if isinstance(stmt, ast.Return)
                and not (isinstance(stmt.value, ast.Constant) and stmt.value.value == "")
            ]
            assert not non_empty, (
                f"{platform_key}/{tuning_value} is justified as generator-returns-empty but the MRO "
                f"generate_tuning_clause returns: {non_empty}"
            )

    def test_uncalled_keys_cover_only_current_noop_entries(self) -> None:
        current = {
            (platform_key, tuning_type.value)
            for platform_key, entries in PLATFORM_TUNING_CAPABILITIES.items()
            for tuning_type, capability in entries.items()
            if is_not_rendered_at_execution(capability)
        }
        assert set(_UNCALLED_GENERATOR_KEYS) <= current, (
            f"stale uncalled-generator keys: {sorted(set(_UNCALLED_GENERATOR_KEYS) - current)}"
        )

    def test_returns_empty_keys_cover_only_current_noop_entries(self) -> None:
        current = {
            (platform_key, tuning_type.value)
            for platform_key, entries in PLATFORM_TUNING_CAPABILITIES.items()
            for tuning_type, capability in entries.items()
            if is_not_rendered_at_execution(capability)
        }
        assert set(_RETURNS_EMPTY_GENERATOR_KEYS) <= current, (
            f"stale returns-empty keys: {sorted(set(_RETURNS_EMPTY_GENERATOR_KEYS) - current)}"
        )

    def test_rendered_constraints_survive_registered_ddl_rewrites(self) -> None:
        for platform_key, entries in PLATFORM_TUNING_CAPABILITIES.items():
            for tuning_type, capability in entries.items():
                if is_not_rendered_at_execution(capability):
                    continue
                if tuning_type.value not in _KIND_STRIP_STEMS:
                    continue
                matches = _applied_kind_strip_rules(platform_key)
                relevant = [transformer for kind, transformer in matches if kind == tuning_type.value]
                assert not relevant, (
                    f"{platform_key}/{tuning_type.value} is registered rendered but an applied "
                    f"DDL_OPTIMIZE rewrite strips it: {relevant}; correct the registry"
                )

    def test_kind_strip_checker_flags_singlestore_fk(self) -> None:
        matches = _applied_kind_strip_rules("singlestore")
        assert ("foreign_keys", "singlestore_strip_foreign_keys") in matches
        assert not [kind for kind, _ in matches if kind == "primary_keys"]

    def test_kind_strip_mapping_is_exact(self) -> None:
        observed = {
            (platform_key, kind, transformer)
            for platform_key in known_registry_platforms()
            for kind, transformer in _applied_kind_strip_rules(platform_key)
        }
        assert observed == {("singlestore", "foreign_keys", "singlestore_strip_foreign_keys")}, (
            f"DDL_OPTIMIZE kind-strip mapping changed: {sorted(observed)}; re-audit the noop gate excuses"
        )

    def test_gate_rejects_rewriting_adapters_unique_check(self) -> None:
        for platform_key in ("starrocks", "doris", "databend"):
            adapter_cls = _adapter_class(resolve_platform_key(platform_key))
            assert adapter_cls is not None
            tuning_read, sql_exec, ddl_handoff, strip_pk, strip_fk, texts = _signals(_reachable_bodies(adapter_cls))
            assert tuning_read and sql_exec and ddl_handoff
            assert strip_pk and strip_fk
            assert not _type_evidenced("unique_constraints", texts, ddl_handoff, strip_pk, strip_fk), platform_key
            assert not _type_evidenced("check_constraints", texts, ddl_handoff, strip_pk, strip_fk), platform_key

    def test_incidental_mentions_evidence_nothing(self) -> None:
        assert not _type_evidenced("unique_constraints", ["unique_identifier column"], False, False, False)
        assert not _type_evidenced("check_constraints", ["check digit validation"], True, True, True)
        assert _type_evidenced("unique_constraints", ["add unique constraint"], False, False, False)
        assert not _type_evidenced("auto_optimize", ["optimize the query plan"], True, False, False)
        assert not _type_evidenced("sorting", ["sorted_rows = sort_results(rows)"], True, False, False)
        assert _type_evidenced("sorting", ["sort key clause"], True, False, False)
        assert not _type_evidenced("check_constraints", ["check the logs for errors"], True, False, False)

    def test_clustering_alone_does_not_evidence_partitioning(self) -> None:
        clustering_only = ["sql = f'alter table t cluster by (a)'", "cluster_by = table_tuning.clustering"]
        assert _type_evidenced("clustering", clustering_only, False, False, False)
        assert not _type_evidenced("partitioning", clustering_only, False, False, False)
        assert _type_evidenced("partitioning", ["create table t (a int) partition by (a)"], False, False, False)
        assert not _type_evidenced("sorting", ["rows arrive pre-sorted"], True, False, False)
        assert not _type_evidenced("clustering", ["cluster status: healthy"], True, False, False)
        assert not _type_evidenced("partitioning", ["partition count"], True, False, False)
        assert not _type_evidenced("distribution", ["redistribute the load"], True, False, False)
        assert not _type_evidenced("clustering", ["liquid_clustering_enabled = True"], True, False, False)
        assert not _type_evidenced("sorting", ["calls apply_ctas_sort next"], True, False, False)
        assert _type_evidenced("clustering", ["alter table t cluster by (c)"], True, False, False)
        assert _type_evidenced("partitioning", ["partition by range (d)"], True, False, False)

    def test_mysql_evidence_rests_on_shared_wire_base(self) -> None:
        from benchbox.platforms.base.mysql_wire import MySqlWireLifecycleMixin, NoOpTableTuningMixin
        from benchbox.platforms.singlestore import SingleStoreAdapter

        adapter_cls = _adapter_class("mysql")
        assert adapter_cls is SingleStoreAdapter
        assert issubclass(adapter_cls, MySqlWireLifecycleMixin)
        assert issubclass(adapter_cls, NoOpTableTuningMixin)
        skip = _PLATFORM_SKIP_MODULES["mysql"]
        bodies = _reachable_bodies(adapter_cls, skip)
        full = _reachable_bodies(adapter_cls)
        assert bodies, "no shared-base lifecycle reachable for mysql"
        excluded = {ast.unparse(node) for node in full} - {ast.unparse(node) for node in bodies}
        assert excluded, "mysql skip set excludes no singlestore-defined lifecycle"
        tuning_read, sql_exec, ddl_handoff, strip_pk, strip_fk, texts = _signals(bodies)
        assert tuning_read and sql_exec and ddl_handoff
        assert not strip_pk and not strip_fk
        for tuning_type, capability in PLATFORM_TUNING_CAPABILITIES["mysql"].items():
            if is_not_rendered_at_execution(capability):
                continue
            assert _type_evidenced(tuning_type.value, texts, ddl_handoff, strip_pk, strip_fk), (
                f"mysql/{tuning_type.value} lacks shared-base evidence"
            )
        singlestore_path = Path(__file__).resolve().parents[4] / "benchbox" / "platforms" / "singlestore.py"
        module_tree = ast.parse(singlestore_path.read_text(encoding="utf-8"))
        singlestore_defs = {
            (node.name, ast.unparse(node))
            for node in ast.walk(module_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for body in bodies:
            assert (body.name, ast.unparse(body)) not in singlestore_defs, (
                f"mysql evidence includes singlestore-defined {body.name}"
            )

    def test_absence_predicate_flags_planted_creation(self) -> None:
        clean = ["def create_schema(self):", "cursor.execute(create_table_sql)"]
        planted = clean + ["cursor.execute('create materialized view mv as select 1')"]
        assert _absence_holds(clean, ("create materialized",))
        assert not _absence_holds(planted, ("create materialized",))
        assert _absence_holds(clean, ("bloom",))
        assert not _absence_holds(clean + ["execute('create bloomfilter index')"], ("bloom",))

    def test_returned_and_query_scoped_execution_credit(self) -> None:
        def _exec_signal(source: str) -> bool:
            nodes = [node for node in ast.walk(ast.parse(textwrap.dedent(source))) if isinstance(node, ast.FunctionDef)]
            return _signals(nodes)[1]

        assert _exec_signal("def run(self):\n    cursor.execute(create_table_sql)\n")
        assert not _exec_signal("def run(self):\n    return cursor.execute(select_sql)\n")
        assert _exec_signal("def create_schema(self):\n    client.query(create_table_sql)\n")
        assert not _exec_signal("def probe(self):\n    client.query(select_sql)\n")
        assert not _exec_signal("def count(self):\n    return client.query(count_sql)\n")
        assert not _exec_signal("def create_schema(self):\n    builder.query(create_table_sql)\n")
        assert not _exec_signal("def create_schema(self):\n    self.preview.query(create_table_sql)\n")
        assert not _exec_signal("def create_schema(self):\n    connection.query(preview)\n")
        assert not _exec_signal("def create_schema(self):\n    connection.query(count_sql)\n")
        assert _exec_signal("def create_schema(self):\n    self.connection.query(create_table_sql)\n")

    def test_pk_fk_flags_drive_ddl_content(self) -> None:
        from benchbox.ssb import SSBBenchmark
        from benchbox.tpcds import TPCDSBenchmark
        from benchbox.tpch import TPCHBenchmark

        for benchmark in (TPCHBenchmark(), TPCDSBenchmark(), SSBBenchmark()):
            name = type(benchmark).__name__
            disabled = UnifiedTuningConfiguration()
            disabled.disable_all_constraints()
            pk_only = UnifiedTuningConfiguration()
            pk_only.disable_all_constraints()
            pk_only.primary_keys.enabled = True
            plain = benchmark.get_create_tables_sql(tuning_config=disabled)
            keyed = benchmark.get_create_tables_sql(tuning_config=pk_only)
            assert keyed != plain, f"{name} DDL ignores the primary_keys flag"
            assert "PRIMARY KEY" in keyed, f"{name} DDL carries no PRIMARY KEY clause"
            assert "PRIMARY KEY" not in plain, f"{name} DDL carries PRIMARY KEY with flags off"
        for benchmark in (TPCHBenchmark(), TPCDSBenchmark()):
            name = type(benchmark).__name__
            disabled = UnifiedTuningConfiguration()
            disabled.disable_all_constraints()
            fk_only = UnifiedTuningConfiguration()
            fk_only.disable_all_constraints()
            fk_only.foreign_keys.enabled = True
            plain = benchmark.get_create_tables_sql(tuning_config=disabled)
            keyed = benchmark.get_create_tables_sql(tuning_config=fk_only)
            assert keyed != plain, f"{name} DDL ignores the foreign_keys flag"
            assert "FOREIGN KEY" in keyed, f"{name} DDL carries no FOREIGN KEY clause"


class TestShippedTemplatesValidateClean:
    @pytest.mark.parametrize("platform", ["clickhouse", "databricks", "duckdb", "snowflake"])
    def test_packaged_templates_validate_clean(self, platform: str) -> None:
        from benchbox.core.tuning.packaged_templates import list_packaged_templates

        templates = list_packaged_templates(platform)
        assert templates.get(platform), f"no packaged templates found for {platform}"
        for template_path in templates[platform]:
            config = UnifiedTuningConfiguration.from_dict(yaml.safe_load(template_path.read_text(encoding="utf-8")))
            errors, _warnings = config.validate_for_platform_detailed(platform)
            assert errors == [], f"{template_path}: {errors}"

    @pytest.mark.parametrize(
        "platform",
        ["duckdb", "clickhouse", "snowflake", "bigquery", "postgresql", "databricks", "redshift"],
    )
    def test_sample_configs_validate_clean(self, tmp_path, platform: str) -> None:
        from benchbox.cli.config import ConfigManager

        target = tmp_path / f"{platform}_tuning.yaml"
        ConfigManager().create_sample_unified_tuning_config(target, platform)
        config = UnifiedTuningConfiguration.from_dict(yaml.safe_load(target.read_text(encoding="utf-8")))
        errors, _warnings = config.validate_for_platform_detailed(platform)
        assert errors == [], f"sample for {platform}: {errors}"

    @pytest.mark.parametrize("platform", sorted(known_registry_platforms() | {"spark", "starburst"}))
    def test_fallback_config_validates_clean(self, tmp_path, platform: str) -> None:
        from benchbox.cli.tuning_runtime import build_default_unified_config

        config = build_default_unified_config(platform)
        errors, _warnings = config.validate_for_platform_detailed(platform)
        assert errors == [], f"fallback for {platform}: {errors}"
