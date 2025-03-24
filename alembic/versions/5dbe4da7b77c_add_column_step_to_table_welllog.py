"""add column step to table WellLog

Revision ID: 5dbe4da7b77c
Revises: 423446c88581
Create Date: 2025-03-24 16:02:32.586104

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5dbe4da7b77c'
down_revision: Union[str, None] = '423446c88581'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('well_log', sa.Column('step', sa.Float, nullable=True))


def downgrade() -> None:
    op.drop_column('well_log', 'step')
