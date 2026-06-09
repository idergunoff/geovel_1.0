"""add cached cluster postprocess results

Revision ID: f5a6b7c8d9e0
Revises: e4f5a6b7c8d9
Create Date: 2026-06-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f5a6b7c8d9e0'
down_revision: Union[str, None] = 'e4f5a6b7c8d9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table_name in ('cluster_calculation_cache', 'well_log_cluster_calculation_cache'):
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.add_column(
                sa.Column('postprocess_results_json', sa.Text(), nullable=False, server_default='{}')
            )


def downgrade() -> None:
    for table_name in ('well_log_cluster_calculation_cache', 'cluster_calculation_cache'):
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.drop_column('postprocess_results_json')
