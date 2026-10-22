"""Persist core interval thickness and geological index.

Revision ID: 2f3a4b5c6d7e
Revises: 1e2f3a4b5c6d
"""
from alembic import op
import sqlalchemy as sa

revision = '2f3a4b5c6d7e'
down_revision = '1e2f3a4b5c6d'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('core_description_interval', sa.Column('thickness', sa.Float(), nullable=True))
    op.add_column('core_description_interval', sa.Column('geological_index_raw', sa.Text(), nullable=True))


def downgrade():
    op.drop_column('core_description_interval', 'geological_index_raw')
    op.drop_column('core_description_interval', 'thickness')
