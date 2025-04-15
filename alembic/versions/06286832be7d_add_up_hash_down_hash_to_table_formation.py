"""add_up_hash_down_hash_to_table_formation

Revision ID: 06286832be7d
Revises: 4175701be29c
Create Date: 2025-04-09 14:10:24.820591

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '06286832be7d'
down_revision: Union[str, None] = '4175701be29c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Добавляем столбцы up_hash и down_hash
    op.add_column('formation', sa.Column('up_hash', sa.String(), nullable=True))
    op.add_column('formation', sa.Column('down_hash', sa.String(), nullable=True))


def downgrade():
    # Удаляем столбцы при откате миграции
    op.drop_column('formation', 'down_hash')
    op.drop_column('formation', 'up_hash')
