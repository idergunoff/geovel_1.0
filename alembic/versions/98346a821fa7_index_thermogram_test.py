"""index thermogram test

Revision ID: 98346a821fa7
Revises: df8ce367b1c8
Create Date: 2024-02-22 14:23:40.161803

"""
from typing import Sequence, Union

import sqlalchemy.exc
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '98346a821fa7'
down_revision: Union[str, None] = 'df8ce367b1c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    try:
        op.create_index('idx_id', 'thermogram', ['id'])
    except sqlalchemy.exc.OperationalError:
        pass


def downgrade() -> None:
    op.drop_index('idx_id', 'thermogram')
