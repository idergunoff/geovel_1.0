"""change_object_set_data_report_to_text

Revision ID: 9b1f2a6c4d31
Revises: df983186961d
Create Date: 2026-04-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '9b1f2a6c4d31'
down_revision: Union[str, None] = 'df983186961d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('object_set') as batch_op:
        batch_op.alter_column('data', existing_type=sa.String(), type_=sa.Text(), existing_nullable=True)
        batch_op.alter_column('report', existing_type=sa.String(), type_=sa.Text(), existing_nullable=True)


def downgrade() -> None:
    with op.batch_alter_table('object_set') as batch_op:
        batch_op.alter_column('data', existing_type=sa.Text(), type_=sa.String(), existing_nullable=True)
        batch_op.alter_column('report', existing_type=sa.Text(), type_=sa.String(), existing_nullable=True)
