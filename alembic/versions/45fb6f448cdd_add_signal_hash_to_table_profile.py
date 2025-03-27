"""add_signal_hash_to_table_profile

Revision ID: 45fb6f448cdd
Revises: 5dbe4da7b77c
Create Date: 2025-03-25 08:42:50.174592

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '45fb6f448cdd'
down_revision: Union[str, None] = '5dbe4da7b77c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('profile', sa.Column('signal_hash', sa.String, nullable=True))


def downgrade() -> None:
    op.drop_column('profile', 'signal_hash')