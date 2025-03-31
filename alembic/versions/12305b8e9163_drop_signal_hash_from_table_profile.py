"""drop_signal_hash_from_table_profile

Revision ID: 12305b8e9163
Revises: 2d4ecc4ae049
Create Date: 2025-03-31 15:06:18.213972

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '12305b8e9163'
down_revision: Union[str, None] = '2d4ecc4ae049'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.drop_column('profile', 'signal_hash')

def downgrade():
    op.add_column('profile', sa.Column('signal_hash', sa.String, nullable=True))
