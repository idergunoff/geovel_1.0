"""merge migration heads

Revision ID: 0d253a325606
Revises: a7c8d9e0f1a2, f5a6b7c8d9e0
Create Date: 2026-08-26 09:25:43.097549

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0d253a325606'
down_revision: Union[str, None] = ('a7c8d9e0f1a2', 'f5a6b7c8d9e0')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
