"""add_signal_hash_md5_to_table_profile

Revision ID: 2d4ecc4ae049
Revises: 45fb6f448cdd
Create Date: 2025-03-31 08:37:37.062570

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2d4ecc4ae049'
down_revision: Union[str, None] = '45fb6f448cdd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('profile', sa.Column('signal_hash_md5', sa.String, nullable=True))


def downgrade() -> None:
    op.drop_column('profile', 'signal_hash_md5')
