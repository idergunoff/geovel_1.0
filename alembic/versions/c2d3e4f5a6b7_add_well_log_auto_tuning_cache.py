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


def upgrade() -> None:
    op.create_table(
        'well_log_cluster_auto_tuning_cache',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=False),
        sa.Column('cache_key', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('top_results', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['dataset_id'], ['well_log_cluster_dataset.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        'ix_well_log_cluster_auto_tuning_cache_dataset_id',
        'well_log_cluster_auto_tuning_cache',
        ['dataset_id'],
        unique=False
    )
    op.create_index(
        'ix_well_log_cluster_auto_tuning_cache_cache_key',
        'well_log_cluster_auto_tuning_cache',
        ['cache_key'],
        unique=True
    )


def downgrade() -> None:
    op.drop_index('ix_well_log_cluster_auto_tuning_cache_cache_key', table_name='well_log_cluster_auto_tuning_cache')
    op.drop_index('ix_well_log_cluster_auto_tuning_cache_dataset_id', table_name='well_log_cluster_auto_tuning_cache')
    op.drop_table('well_log_cluster_auto_tuning_cache')
