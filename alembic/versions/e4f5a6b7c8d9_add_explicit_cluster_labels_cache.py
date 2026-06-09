"""add explicit pre-smoothing labels to cluster calculation cache

Revision ID: e4f5a6b7c8d9
Revises: d3e4f5a6b7c8
Create Date: 2026-06-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e4f5a6b7c8d9'
down_revision: Union[str, None] = 'd3e4f5a6b7c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table_name in ('cluster_calculation_cache', 'well_log_cluster_calculation_cache'):
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.add_column(sa.Column('labels_json', sa.Text(), nullable=False, server_default='[]'))
            batch_op.add_column(sa.Column('kept_row_indices_json', sa.Text(), nullable=False, server_default='[]'))
            batch_op.add_column(sa.Column('assignments_json', sa.Text(), nullable=False, server_default='[]'))


def downgrade() -> None:
    for table_name in ('well_log_cluster_calculation_cache', 'cluster_calculation_cache'):
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.drop_column('assignments_json')
            batch_op.drop_column('kept_row_indices_json')
            batch_op.drop_column('labels_json')
