"""add normalized names for well log aliases

Revision ID: 9b21c6e4a1f2
Revises: b1e2c3d4f5a6
Create Date: 2026-05-22 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9b21c6e4a1f2'
down_revision = 'b1e2c3d4f5a6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('canonical_well_log', schema=None) as batch_op:
        batch_op.add_column(sa.Column('canonical_name_norm', sa.String(), nullable=True))

    op.execute("""
        UPDATE canonical_well_log
        SET canonical_name_norm = lower(trim(canonical_name))
        WHERE canonical_name IS NOT NULL
    """)

    with op.batch_alter_table('canonical_well_log', schema=None) as batch_op:
        batch_op.alter_column('canonical_name_norm', existing_type=sa.String(), nullable=False)
        batch_op.create_index('ix_canonical_well_log_canonical_name_norm', ['canonical_name_norm'], unique=True)

    with op.batch_alter_table('alias_well_log', schema=None) as batch_op:
        batch_op.add_column(sa.Column('alias_name_norm', sa.String(), nullable=True))

    op.execute("""
        UPDATE alias_well_log
        SET alias_name_norm = lower(trim(alias_name))
        WHERE alias_name IS NOT NULL
    """)

    with op.batch_alter_table('alias_well_log', schema=None) as batch_op:
        batch_op.alter_column('alias_name_norm', existing_type=sa.String(), nullable=False)
        batch_op.create_index('ix_alias_well_log_alias_name_norm', ['alias_name_norm'], unique=True)


def downgrade() -> None:
    with op.batch_alter_table('alias_well_log', schema=None) as batch_op:
        batch_op.drop_index('ix_alias_well_log_alias_name_norm')
        batch_op.drop_column('alias_name_norm')

    with op.batch_alter_table('canonical_well_log', schema=None) as batch_op:
        batch_op.drop_index('ix_canonical_well_log_canonical_name_norm')
        batch_op.drop_column('canonical_name_norm')
