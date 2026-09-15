"""add reusable well parameter choices

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
"""
from alembic import op
import sqlalchemy as sa

revision = 'c9d0e1f2a3b4'
down_revision = 'b8c9d0e1f2a3'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'well_parameter_choice',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('well_id', sa.Integer(), sa.ForeignKey('well.id'), nullable=False),
        sa.Column('source_type', sa.String(), nullable=False),
        sa.Column('canonical_id', sa.Integer(), nullable=False),
        sa.Column('source_id', sa.Integer(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.UniqueConstraint('well_id', 'source_type', 'canonical_id',
                            name='uq_well_parameter_choice'),
    )
    op.create_index('ix_well_parameter_choice_well_id', 'well_parameter_choice', ['well_id'])


def downgrade():
    op.drop_index('ix_well_parameter_choice_well_id', table_name='well_parameter_choice')
    op.drop_table('well_parameter_choice')
