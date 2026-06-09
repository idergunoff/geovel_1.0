"""add persistent cluster calculation cache

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-06-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd3e4f5a6b7c8'
down_revision: Union[str, None] = 'c2d3e4f5a6b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _create_cache_table(table_name: str, owner_column: str, owner_table: str) -> None:
    op.create_table(
        table_name,
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column(owner_column, sa.Integer(), nullable=False),
        sa.Column('cache_key', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('data_hash', sa.String(), nullable=False),
        sa.Column('config_json', sa.Text(), nullable=False),
        sa.Column('result_payload', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint([owner_column], [f'{owner_table}.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(f'ix_{table_name}_{owner_column}', table_name, [owner_column], unique=False)
    op.create_index(f'ix_{table_name}_cache_key', table_name, ['cache_key'], unique=True)


def upgrade() -> None:
    _create_cache_table('cluster_calculation_cache', 'object_set_id', 'object_set')
    _create_cache_table('well_log_cluster_calculation_cache', 'dataset_id', 'well_log_cluster_dataset')


def downgrade() -> None:
    for table_name, owner_column in (
        ('well_log_cluster_calculation_cache', 'dataset_id'),
        ('cluster_calculation_cache', 'object_set_id'),
    ):
        op.drop_index(f'ix_{table_name}_cache_key', table_name=table_name)
        op.drop_index(f'ix_{table_name}_{owner_column}', table_name=table_name)
        op.drop_table(table_name)
