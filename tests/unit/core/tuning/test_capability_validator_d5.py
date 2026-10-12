from __future__ import annotations

import ast
import inspect
import sys
import textwrap

import pytest
import yaml

from benchbox.core.tuning.capability_registry import (
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

_EXECUTE_ATTRS = frozenset({"execute", "executescript", "exec", "sql", "query"})

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
    "partitioning": ("partition",),
    "sorting": ("sort",),
    "clustering": ("cluster",),
    "distribution": ("distribut",),
    "primary_keys": ("primary_key", "primary key"),
    "foreign_keys": ("foreign_key", "foreign key"),
    "unique_constraints": ("unique",),
    "check_constraints": ("check",),
    "z_ordering": ("z_order", "zorder"),
    "liquid_clustering": ("liquid",),
    "auto_optimize": ("auto_optimize", "autooptimize", "optimize"),
    "auto_compact": ("auto_compact", "autocompact", "compact"),
    "bloom_filters": ("bloom",),
    "materialized_views": ("materialized",),
}

_CONSTRAINT_TYPES = frozenset({"primary_keys", "foreign_keys", "unique_constraints", "check_constraints"})

_STRIP_KIND = {"primary_keys": "strip_pk", "foreign_keys": "strip_fk"}


def _module_functions(adapter_cls: type) -> dict[str, list[ast.FunctionDef]]:
    cache: dict[str, dict[str, list[ast.FunctionDef]]] = _module_functions.__dict__.setdefault("cache", {})
    found: dict[str, list[ast.FunctionDef]] = {}
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
            entries: dict[str, list[ast.FunctionDef]] = {}
            for node in module.body:
                if isinstance(node, ast.FunctionDef):
                    entries.setdefault(node.name, []).append(node)
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


def _enqueue_callees(
    node: ast.FunctionDef,
    class_name: str,
    definers: dict[str, list[tuple[str, ast.FunctionDef]]],
    class_index: dict[str, int],
    mro: list[type],
    helpers: dict[str, list[ast.FunctionDef]],
    queue: list[tuple[str, str]],
) -> None:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            if func.attr in definers:
                queue.append((definers[func.attr][0][0], func.attr))
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call):
            inner = func.value
            if isinstance(inner.func, ast.Name) and inner.func.id == "super" and class_name in class_index:
                position = class_index[class_name]
                for klass in mro[position + 1 :]:
                    if any(entry[0] == klass.__name__ for entry in definers.get(func.attr, [])):
                        queue.append((klass.__name__, func.attr))
                        break
        elif isinstance(func, ast.Name) and func.id in helpers:
            for position, helper in enumerate(helpers[func.id]):
                queue.append((func.id, f"{helper.name}#{position}"))


def _reachable_bodies(adapter_cls: type) -> list[ast.FunctionDef]:
    mro, definers = _collect_definers(adapter_cls)
    class_index = {klass.__name__: position for position, klass in enumerate(mro)}
    helpers = _module_functions(adapter_cls)
    helper_nodes: dict[tuple[str, str], ast.FunctionDef] = {}
    for helper_name, nodes in helpers.items():
        for position, helper in enumerate(nodes):
            helper_nodes[(helper_name, f"{helper.name}#{position}")] = helper
    reached: dict[tuple[str, str], ast.FunctionDef] = {}
    queue: list[tuple[str, str]] = []
    for name, entries in definers.items():
        if name in _LIFECYCLE_ROOTS:
            queue.append((entries[0][0], name))
    while queue:
        class_name, name = queue.pop()
        if (class_name, name) in reached:
            continue
        if (class_name, name) in helper_nodes:
            reached[(class_name, name)] = helper_nodes[(class_name, name)]
            _enqueue_callees(helper_nodes[(class_name, name)], class_name, definers, class_index, mro, helpers, queue)
            continue
        entries = [entry for entry in definers.get(name, []) if entry[0] == class_name]
        if not entries:
            continue
        node = entries[0][1]
        reached[(class_name, name)] = node
        _enqueue_callees(node, class_name, definers, class_index, mro, helpers, queue)
    inline: list[ast.FunctionDef] = list(reached.values())
    return inline


def _signals(
    bodies: list[ast.FunctionDef],
) -> tuple[bool, bool, bool, bool, bool, bool, list[str]]:
    tuning_read = False
    sql_exec = False
    strips = False
    ddl_handoff = False
    strip_pk = False
    strip_fk = False
    texts: list[str] = []
    for body in bodies:
        try:
            texts.append(ast.unparse(body).lower())
        except (ValueError, SyntaxError):
            continue
        for child in ast.walk(body):
            if isinstance(child, ast.Name) and child.id in _TUNING_READ_NAMES:
                tuning_read = True
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Attribute) and func.attr in _EXECUTE_ATTRS:
                    sql_exec = True
                if isinstance(func, ast.Name) and func.id in _STRIP_CALLS:
                    strips = True
                    if func.id == "strip_primary_keys":
                        strip_pk = True
                    else:
                        strip_fk = True
                if isinstance(func, ast.Attribute) and func.attr == "sub":
                    for arg in child.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            if "primary" in arg.value.lower() and "key" in arg.value.lower():
                                strips = True
                                strip_pk = True
                            if "foreign" in arg.value.lower() and "key" in arg.value.lower():
                                strips = True
                                strip_fk = True
                if isinstance(func, ast.Attribute) and func.attr == "get_create_tables_sql":
                    tuning_read = True
                for keyword in child.keywords:
                    if keyword.arg == "tuning_config":
                        tuning_read = True
                        if isinstance(func, ast.Attribute) and func.attr == "get_create_tables_sql":
                            ddl_handoff = True
    return tuning_read, sql_exec, strips, ddl_handoff, strip_pk, strip_fk, texts


def _type_evidenced(
    tuning_type_value: str, texts: list[str], ddl_handoff: bool, strip_pk: bool, strip_fk: bool
) -> bool:
    tokens = _TYPE_EVIDENCE_TOKENS.get(tuning_type_value, ())
    if any(token in text for text in texts for token in tokens):
        return True
    if tuning_type_value in _CONSTRAINT_TYPES and ddl_handoff:
        kind = _STRIP_KIND.get(tuning_type_value)
        if kind is None:
            return True
        if kind == "strip_pk" and not strip_pk:
            return True
        if kind == "strip_fk" and not strip_fk:
            return True
    return False


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


_SKIP_NO_ADAPTER_CLASS = frozenset({"mysql"})

_RENDERED_OVERRIDE_JUSTIFICATIONS: dict[tuple[str, str], str] = {}

_NONE_JUSTIFICATION_OVERRIDES: dict[tuple[str, str], str] = {
    ("trino", "partitioning"): "Preview generator uncalled at execution; apply path log-only (trino.py:190-298).",
    ("trino", "distribution"): "Preview generator uncalled at execution; apply path log-only (trino.py:190-298).",
    ("trino", "sorting"): "Preview generator uncalled at execution; apply path log-only (trino.py:190-298).",
    ("trino", "clustering"): "Preview generator uncalled at execution; apply path log-only (trino.py:190-298).",
    ("presto", "partitioning"): "Preview generator uncalled at execution; apply path log-only (presto.py:214-282).",
    ("presto", "distribution"): "Preview generator uncalled at execution; apply path log-only (presto.py:214-282).",
    ("presto", "sorting"): "Preview generator uncalled at execution; apply path log-only (presto.py:214-282).",
    ("presto", "clustering"): "Preview generator uncalled at execution; apply path log-only (presto.py:214-282).",
    (
        "athena",
        "partitioning",
    ): "External-table rewrite never adds PARTITIONED BY; tuning clause uncalled (athena.py:627-801).",
    ("athena", "distribution"): "Inherited generator output excluded from TBLPROPERTIES (trino.py:433-437).",
    ("athena", "sorting"): "Inherited generator output excluded from TBLPROPERTIES (trino.py:433-437).",
    ("athena", "clustering"): "No clustering clause; folded into dropped sorted_by.",
    (
        "firebolt",
        "partitioning",
    ): "Preview generator uncalled at execution; apply path log-only (firebolt.py:1267-1421).",
    (
        "firebolt",
        "distribution",
    ): "Preview generator uncalled at execution; apply path log-only (firebolt.py:1267-1421).",
    ("firebolt", "sorting"): "Firebolt sorts within segments automatically; generator log-only (firebolt.py:126-133).",
    ("firebolt", "clustering"): "Clustering via PRIMARY INDEX; generator log-only (firebolt.py:135-142).",
    ("azure-synapse", "partitioning"): "Execution path never adds a PARTITION clause (azure_synapse.py:1046-1069).",
    (
        "azure-synapse",
        "distribution",
    ): "Real DDL uses engine default, never tuned columns (azure_synapse.py:1046-1069).",
    ("azure-synapse", "sorting"): "Sorting via index type; generator log-only (azure_synapse.py:173-181).",
    (
        "azure-synapse",
        "clustering",
    ): "Clustering via DISTRIBUTION/INDEX; generator log-only (azure_synapse.py:183-189).",
    (
        "questdb",
        "partitioning",
    ): "Execution uses adapter-local hardcoded maps, ignoring tuned columns (questdb.py:521-558).",
    ("questdb", "sorting"): "Sorts by designated timestamp automatically; generator log-only (questdb.py:185-191).",
    ("questdb", "distribution"): "Single-node engine; generator warning-logs only (questdb.py:193-199).",
    (
        "timescaledb",
        "partitioning",
    ): "Execution converts only literal 'time' columns, ignoring tuning (timescaledb.py:378-467).",
    ("timescaledb", "sorting"): "compress_orderby never set from tuning (timescaledb.py:476-521).",
    ("timescaledb", "distribution"): "compress_segmentby hardcoded empty (timescaledb.py:476-521).",
    ("timescaledb", "clustering"): "Merged into hardcoded-empty compress_segmentby (timescaledb.py:177-186).",
    ("pg-duckdb", "partitioning"): "Inherits PostgreSQLAdapter no-ops; adapter defines no overrides (pg_duckdb.py).",
    (
        "pg-duckdb",
        "clustering",
    ): "Inherits PostgreSQLAdapter no-ops; CLUSTER filtered from preview (pg_duckdb.py:83-88).",
    ("pg-duckdb", "sorting"): "Inherits PostgreSQLAdapter no-ops; CREATE INDEX preview-only (pg_duckdb.py:83-88).",
    (
        "pg-mooncake",
        "partitioning",
    ): "Generator always empty by design; tuning promoted away (pg_mooncake.py:61-81,196-204).",
    (
        "pg-mooncake",
        "clustering",
    ): "Generator always empty by design; tuning promoted away (pg_mooncake.py:61-81,196-204).",
    (
        "pg-mooncake",
        "sorting",
    ): "Generator always empty by design; tuning promoted away (pg_mooncake.py:61-81,196-204).",
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
    ("redshift", "partitioning"): "No CREATE TABLE partition clause; recreation never happens (redshift.py:2512-2636).",
    ("redshift", "distribution"): "Recreation never happens; maintenance only (redshift.py:2607-2624).",
    ("bigquery", "partitioning"): "Recreation hint only; table never re-created (bigquery.py:2013-2027).",
    ("bigquery", "clustering"): "Recreation hint only; table never re-created (bigquery.py:2013-2027).",
    ("databricks", "distribution"): "No user-managed distribution key; folded into ZORDER upstream.",
    ("databricks", "bloom_filters"): "Compatible per legacy map; no adapter code creates a Bloom filter index.",
    ("databricks", "materialized_views"): "Compatible per legacy map; no adapter code creates a materialized view.",
    ("snowflake", "materialized_views"): "Compatible per legacy map; no adapter implementation.",
    ("bigquery", "materialized_views"): "Compatible per legacy map; no adapter implementation.",
    ("redshift", "materialized_views"): "Compatible per legacy map; no adapter implementation.",
    ("postgresql", "bloom_filters"): "Compatible per legacy map; no adapter implementation.",
    ("postgresql", "materialized_views"): "Compatible per legacy map; no adapter implementation.",
    ("cedardb", "bloom_filters"): "Inherits PostgreSQLAdapter no-ops with no DDL override (cedardb.py).",
    ("cedardb", "materialized_views"): "Inherits PostgreSQLAdapter no-ops with no DDL override (cedardb.py).",
    ("citus", "bloom_filters"): "Inherits PostgreSQLAdapter no-ops with no DDL override (citus.py).",
    ("citus", "materialized_views"): "Inherits PostgreSQLAdapter no-ops with no DDL override (citus.py).",
    ("paradedb", "bloom_filters"): "Inherits PostgreSQLAdapter no-ops with no DDL override (paradedb.py).",
    ("paradedb", "materialized_views"): "Inherits PostgreSQLAdapter no-ops with no DDL override (paradedb.py).",
    ("duckdb", "partitioning"): "Generator logs a COPY TO hint only; no path applies it (capability note).",
    ("clickhouse", "materialized_views"): "Compatible per legacy map; no adapter code creates a materialized view.",
    ("clickhouse", "unique_constraints"): "ClickHouse has no UNIQUE constraint syntax and no adapter code renders one.",
    ("motherduck", "partitioning"): "tuning_config dropped at create_schema (motherduck.py:307-318).",
    ("motherduck", "clustering"): "No post-create CLUSTER step consumes tuning.",
    ("motherduck", "distribution"): "Single-writer cloud DuckDB; nothing renders distribution columns.",
    ("motherduck", "materialized_views"): "Compatible per legacy map; no adapter implementation.",
    ("motherduck", "primary_keys"): "tuning_config dropped; applier pass-through no-op (motherduck.py:354-355).",
    ("motherduck", "foreign_keys"): "tuning_config dropped; applier pass-through no-op (motherduck.py:354-355).",
    ("motherduck", "unique_constraints"): "tuning_config dropped; applier pass-through no-op (motherduck.py:354-355).",
    ("motherduck", "check_constraints"): "tuning_config dropped; applier pass-through no-op (motherduck.py:354-355).",
    ("datafusion", "primary_keys"): "Schema built explicitly without constraints (datafusion.py:558-564).",
    ("datafusion", "foreign_keys"): "Schema built explicitly without constraints (datafusion.py:558-564).",
    ("datafusion", "unique_constraints"): "Schema built explicitly without constraints (datafusion.py:558-564).",
    ("datafusion", "check_constraints"): "Schema built explicitly without constraints (datafusion.py:558-564).",
    ("doris", "primary_keys"): "Adapter strips PRIMARY KEY clauses from every CREATE TABLE (doris.py:803-805).",
    ("doris", "foreign_keys"): "Adapter strips foreign keys from every CREATE TABLE (doris.py:802).",
    ("databend", "foreign_keys"): "Adapter strips foreign keys from every CREATE TABLE (databend/adapter.py:680).",
    ("databend", "primary_keys"): "Adapter strips PRIMARY KEY clauses via inline regex (databend/adapter.py:678).",
    ("doris", "unique_constraints"): "NoOp adapter; UNIQUE never produced or consumed from tuning.",
    ("doris", "check_constraints"): "NoOp adapter; CHECK never produced or consumed from tuning.",
    ("databend", "unique_constraints"): "Only CLUSTERING renders; UNIQUE never produced or consumed from tuning.",
    ("databend", "check_constraints"): "Only CLUSTERING renders; CHECK never produced or consumed from tuning.",
    (
        "starrocks",
        "unique_constraints",
    ): "Workload rewrite handles PARTITION/DISTRIBUTION/ORDER BY plus PK-model only (starrocks/workload.py:109-198).",
    (
        "starrocks",
        "check_constraints",
    ): "Workload rewrite handles PARTITION/DISTRIBUTION/ORDER BY plus PK-model only (starrocks/workload.py:109-198).",
    (
        "mysql",
        "partitioning",
    ): "No MySQLAdapter class exists; mysql-wire adapters are NoOp (base/mysql_wire.py:514-522).",
    ("singlestore", "partitioning"): "NoOpTableTuningMixin: generate returns '', apply is a no-op (singlestore.py:75).",
    (
        "fabric-warehouse",
        "partitioning",
    ): "T-SQL execution ignores tuned partition columns (fabric_warehouse.py:857-883).",
}


class TestRegistryExecutionParity:
    @pytest.mark.parametrize(
        "platform_key",
        sorted(known_registry_platforms() - _SKIP_NO_ADAPTER_CLASS),
    )
    def test_rendered_entries_have_executing_implementation(self, platform_key: str) -> None:
        adapter_cls = _adapter_class(resolve_platform_key(platform_key))
        assert adapter_cls is not None, f"no adapter class mapped for registry platform '{platform_key}'"
        bodies = _reachable_bodies(adapter_cls)
        assert bodies, f"no lifecycle roots reachable on {adapter_cls.__name__}"
        tuning_read, sql_exec, _strips, ddl_handoff, strip_pk, strip_fk, texts = _signals(bodies)
        for tuning_type, capability in PLATFORM_TUNING_CAPABILITIES[platform_key].items():
            if is_not_rendered_at_execution(capability):
                continue
            key = (platform_key, tuning_type.value)
            evidenced = _type_evidenced(tuning_type.value, texts, ddl_handoff, strip_pk, strip_fk)
            assert key in _RENDERED_OVERRIDE_JUSTIFICATIONS or ((tuning_read and sql_exec) and evidenced), (
                f"{platform_key}/{tuning_type.value} is registered '{capability.rendered_via}' "
                f"({capability.mechanism_id}) but the {adapter_cls.__name__} lifecycle shows "
                f"tuning_read={tuning_read} sql_exec={sql_exec} type_evidenced={evidenced}; "
                f"add execution or correct the registry"
            )

    @pytest.mark.parametrize(
        "platform_key",
        sorted(known_registry_platforms() - _SKIP_NO_ADAPTER_CLASS),
    )
    def test_noop_entries_are_justified(self, platform_key: str) -> None:
        adapter_cls = _adapter_class(resolve_platform_key(platform_key))
        assert adapter_cls is not None, f"no adapter class mapped for registry platform '{platform_key}'"
        bodies = _reachable_bodies(adapter_cls)
        _tuning_read, _sql_exec, strips, _ddl_handoff, _strip_pk, _strip_fk, _texts = _signals(bodies)
        for tuning_type, capability in PLATFORM_TUNING_CAPABILITIES[platform_key].items():
            if not is_not_rendered_at_execution(capability):
                continue
            key = (platform_key, tuning_type.value)
            assert strips or key in _NONE_JUSTIFICATION_OVERRIDES, (
                f"{platform_key}/{tuning_type.value} is registered not-rendered but the "
                f"{adapter_cls.__name__} lifecycle shows no strip and no documented justification; "
                f"document it in _NONE_JUSTIFICATION_OVERRIDES or correct the registry"
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

    def test_skip_set_is_minimal_and_documented(self) -> None:
        assert frozenset({"mysql"}) == _SKIP_NO_ADAPTER_CLASS


class TestShippedTemplatesValidateClean:
    @pytest.mark.parametrize("platform", ["clickhouse", "databricks", "duckdb", "snowflake"])
    def test_packaged_templates_validate_clean(self, platform: str) -> None:
        from benchbox.core.tuning.packaged_templates import list_packaged_templates

        templates = list_packaged_templates(platform)
        assert templates.get(platform), f"no packaged templates found for {platform}"
        for template_path in templates[platform]:
            config = UnifiedTuningConfiguration.from_dict(yaml.safe_load(template_path.read_text()))
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
        config = UnifiedTuningConfiguration.from_dict(yaml.safe_load(target.read_text()))
        errors, _warnings = config.validate_for_platform_detailed(platform)
        assert errors == [], f"sample for {platform}: {errors}"

    @pytest.mark.parametrize("platform", sorted(known_registry_platforms() | {"spark", "starburst"}))
    def test_fallback_config_validates_clean(self, tmp_path, platform: str) -> None:
        from benchbox.cli.tuning_runtime import build_default_unified_config

        config = build_default_unified_config(platform)
        errors, _warnings = config.validate_for_platform_detailed(platform)
        assert errors == [], f"fallback for {platform}: {errors}"
