"""add regression target provenance

Revision ID: b8c9d0e1f2a3
Revises: a7c8d9e0f1a2
"""
from alembic import op
import sqlalchemy as sa

revision = 'b8c9d0e1f2a3'
down_revision = 'a7c8d9e0f1a2'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('markup_reg') as batch_op:
        batch_op.add_column(sa.Column('target_source_type', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('target_source_config', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('target_source_details', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('target_is_manual_override', sa.Boolean(), nullable=True, server_default=sa.false()))


def downgrade():
    with op.batch_alter_table('markup_reg') as batch_op:
        batch_op.drop_column('target_is_manual_override')
        batch_op.drop_column('target_source_details')
        batch_op.drop_column('target_source_config')
        batch_op.drop_column('target_source_type')
