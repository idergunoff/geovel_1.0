"""add_well_hash_to_table_well

Revision ID: 4175701be29c
Revises: 12305b8e9163
Create Date: 2025-04-04 10:24:46.533877

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4175701be29c'
down_revision: Union[str, None] = '12305b8e9163'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Добавление столбца well_hash в таблицу well
    op.add_column('well', sa.Column('well_hash', sa.String(), nullable=True))

def downgrade():
    # Удаление столбца well_hash из таблицы well
    op.drop_column('well', 'well_hash')
