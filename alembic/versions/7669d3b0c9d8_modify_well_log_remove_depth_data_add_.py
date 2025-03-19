"""Modify well_log: remove depth_data, add begin, end, description

Revision ID: 7669d3b0c9d8
Revises: 0bb5688f605e
Create Date: 2025-03-19 11:46:02.058394

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7669d3b0c9d8'
down_revision: Union[str, None] = '0bb5688f605e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Удаляем depth_data
    op.drop_column('well_log', 'depth_data')

    # Добавляем новые столбцы
    op.add_column('well_log', sa.Column('begin', sa.Float, nullable=True))
    op.add_column('well_log', sa.Column('end', sa.Float, nullable=True))
    op.add_column('well_log', sa.Column('description', sa.Text, nullable=True))


def downgrade():
    # Восстанавливаем depth_data
    op.add_column('well_log', sa.Column('depth_data', sa.Text, nullable=True))

    # Удаляем новые столбцы
    op.drop_column('well_log', 'begin')
    op.drop_column('well_log', 'end')
    op.drop_column('well_log', 'description')
    # ### end Alembic commands ###
