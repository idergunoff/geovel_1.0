"""Restore saved predictions previously marked stale after a terrain change."""
from alembic import op
import sqlalchemy as sa

revision = '4b5c6d7e8f90'
down_revision = '3a4b5c6d7e8f'
branch_labels = None
depends_on = None


def upgrade():
    if not op.get_context().as_sql:
        inspector = sa.inspect(op.get_bind())
        if not inspector.has_table('profile_model_prediction'):
            return
        if 'is_stale' not in {column['name'] for column in inspector.get_columns('profile_model_prediction')}:
            return
    op.execute(sa.text('UPDATE profile_model_prediction SET is_stale = 0 WHERE is_stale <> 0'))


def downgrade():
    # Original values cannot be reconstructed; predictions stay usable.
    pass
