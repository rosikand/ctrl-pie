from __future__ import annotations

import importlib.util
from pathlib import Path
import types
import uuid

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ctrl_pi.db import Base
from ctrl_pi.models import YAMCell, YAMCellArm, YAMSetup


def _arm(
    *,
    position: int,
    logical_id: str,
    role: str,
    pair_id: str,
    side: str,
    stable_identity: str,
) -> YAMCellArm:
    return YAMCellArm(
        position=position,
        logical_id=logical_id,
        name=logical_id.replace("-", " ").title(),
        role=role,
        pair_id=pair_id,
        group_id="bimanual",
        side=side,
        transport_kind="socketcan",
        stable_identity=stable_identity,
        end_effector_kind=(
            "yam_teaching_handle" if role == "leader" else "linear_4310"
        ),
        config={},
    )


def _cell() -> YAMCell:
    return YAMCell(
        id="primary",
        name="Test four-arm cell",
        i2rt_root="/opt/operator/i2rt",
        i2rt_commit="a" * 40,
        pair_ports={"pair-a": 11_333, "pair-b": 11_334},
        auto_restore=False,
        arms=[
            _arm(
                position=0,
                logical_id="leader-right",
                role="leader",
                pair_id="pair-a",
                side="right",
                stable_identity="FAKE-ADAPTER-A",
            ),
            _arm(
                position=1,
                logical_id="follower-right",
                role="follower",
                pair_id="pair-a",
                side="right",
                stable_identity="FAKE-ADAPTER-B",
            ),
            _arm(
                position=2,
                logical_id="leader-left",
                role="leader",
                pair_id="pair-b",
                side="left",
                stable_identity="FAKE-ADAPTER-C",
            ),
            _arm(
                position=3,
                logical_id="follower-left",
                role="follower",
                pair_id="pair-b",
                side="left",
                stable_identity="FAKE-ADAPTER-D",
            ),
        ],
    )


@pytest.fixture
def engine():
    value = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(value)
    return value


def test_cell_models_round_trip_normalized_arm_metadata(engine) -> None:
    with Session(engine) as db:
        db.add(_cell())
        db.commit()
        db.expire_all()

        stored = db.get(YAMCell, "primary")
        assert stored is not None
        assert stored.pair_ports == {"pair-a": 11_333, "pair-b": 11_334}
        assert [arm.logical_id for arm in stored.arms] == [
            "leader-right",
            "follower-right",
            "leader-left",
            "follower-left",
        ]
        assert [(arm.role, arm.pair_id, arm.group_id, arm.side) for arm in stored.arms] == [
            ("leader", "pair-a", "bimanual", "right"),
            ("follower", "pair-a", "bimanual", "right"),
            ("leader", "pair-b", "bimanual", "left"),
            ("follower", "pair-b", "bimanual", "left"),
        ]


def test_cell_models_persist_an_empty_topology_draft(engine) -> None:
    with Session(engine) as db:
        db.add(
            YAMCell(
                id="primary",
                name="Unassigned cell",
                i2rt_root="/opt/operator/i2rt",
                i2rt_commit="b" * 40,
                pair_ports={},
                auto_restore=False,
                arms=[],
            )
        )
        db.commit()
        db.expire_all()

        stored = db.get(YAMCell, "primary")
        assert stored is not None
        assert stored.arms == []


def test_cell_schema_has_no_runtime_socketcan_identity_column(engine) -> None:
    columns = {column["name"] for column in sa.inspect(engine).get_columns("yam_cell_arms")}

    assert "stable_identity" in columns
    assert "can_interface" not in columns
    assert "runtime_interface" not in columns


def test_cell_rejects_duplicate_durable_arm_identity(engine) -> None:
    cell = _cell()
    cell.arms[3].stable_identity = cell.arms[0].stable_identity

    with Session(engine) as db:
        db.add(cell)
        with pytest.raises(IntegrityError):
            db.commit()


def test_cell_rejects_unsupported_transport_role_shape(engine) -> None:
    cell = _cell()
    cell.arms[0].transport_kind = "serial"

    with Session(engine) as db:
        db.add(cell)
        with pytest.raises(IntegrityError):
            db.commit()


def _load_migration() -> types.ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0011_yam_cells.py"
    )
    spec = importlib.util.spec_from_file_location("migration_0011_yam_cells", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_upgrades_cleanly_and_preserves_legacy_setup() -> None:
    migration = _load_migration()
    engine = create_engine("sqlite+pysqlite:///:memory:")
    legacy_values = {
        "id": "primary",
        "mode": "hardware",
        "can_interface": "can7",
        "leader_port": "/dev/serial/by-id/legacy-test",
        "mujoco_xml_path": "/opt/legacy/yam.xml",
        "leader_calibration_id": "legacy-leader",
        "leader_calibration_dir": "/var/lib/legacy/calibration",
        "auto_restore": True,
    }

    with engine.begin() as connection:
        YAMSetup.__table__.create(connection)
        connection.execute(sa.insert(YAMSetup.__table__).values(**legacy_values))
        migration.op = Operations(MigrationContext.configure(connection))

        migration.upgrade()

        inspector = sa.inspect(connection)
        assert {"yam_setups", "yam_cells", "yam_cell_arms"} <= set(
            inspector.get_table_names()
        )
        assert {column["name"] for column in inspector.get_columns("yam_cells")} == {
            column.name for column in YAMCell.__table__.columns
        }
        assert {
            column["name"] for column in inspector.get_columns("yam_cell_arms")
        } == {column.name for column in YAMCellArm.__table__.columns}
        assert connection.execute(
            sa.select(YAMSetup.__table__.c.can_interface).where(
                YAMSetup.__table__.c.id == "primary"
            )
        ).scalar_one() == "can7"
        assert connection.execute(
            sa.select(YAMSetup.__table__.c.auto_restore)
        ).scalar_one() is False

        migration.downgrade()

        assert sa.inspect(connection).get_table_names() == ["yam_setups"]
        assert connection.execute(
            sa.select(YAMSetup.__table__.c.leader_port)
        ).scalar_one() == "/dev/serial/by-id/legacy-test"
