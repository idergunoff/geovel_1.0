"""add well log auto tuning cache

Revision ID: c2d3e4f5a6b7
Revises: 9b21c6e4a1f2
Create Date: 2026-06-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c2d3e4f5a6b7'
down_revision: Union[str, None] = '9b21c6e4a1f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _create_table(name, *columns):
    if op.get_context().as_sql or not sa.inspect(op.get_bind()).has_table(name):
        op.create_table(name, *columns)


def _create_index(name, table, columns, **kwargs):
    indexes = set() if op.get_context().as_sql else {
        index['name'] for index in sa.inspect(op.get_bind()).get_indexes(table)
    }
    if name not in indexes:
        op.create_index(name, table, columns, **kwargs)


def upgrade() -> None:
    _create_table(
        'well_log_cluster_auto_tuning_cache',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=False),
        sa.Column('cache_key', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('top_results', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['dataset_id'], ['well_log_cluster_dataset.id']),
        sa.PrimaryKeyConstraint('id')
    )
    _create_index(
        'ix_well_log_cluster_auto_tuning_cache_dataset_id',
        'well_log_cluster_auto_tuning_cache',
        ['dataset_id'],
        unique=False
    )
    _create_index(
        'ix_well_log_cluster_auto_tuning_cache_cache_key',
        'well_log_cluster_auto_tuning_cache',
        ['cache_key'],
        unique=True
    )


def downgrade() -> None:
    op.drop_index('ix_well_log_cluster_auto_tuning_cache_cache_key', table_name='well_log_cluster_auto_tuning_cache')
    op.drop_index('ix_well_log_cluster_auto_tuning_cache_dataset_id', table_name='well_log_cluster_auto_tuning_cache')
    op.drop_table('well_log_cluster_auto_tuning_cache')
