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
    for table, source in (('canonical_well_log', 'canonical_name'), ('alias_well_log', 'alias_name')):
        normalized = f'{source}_norm'
        inspector = sa.inspect(op.get_bind())
        columns = {column['name']: column for column in inspector.get_columns(table)}
        if normalized not in columns:
            op.add_column(table, sa.Column(normalized, sa.String(), nullable=True))

        # Preserve values already normalized by the application's Unicode casefold.
        op.execute(f'UPDATE {table} SET {normalized} = lower(trim({source})) '
                   f'WHERE {normalized} IS NULL AND {source} IS NOT NULL')

        indexes = {index['name'] for index in inspector.get_indexes(table)}
        index_name = f'ix_{table}_{normalized}'
        with op.batch_alter_table(table, schema=None) as batch_op:
            if normalized not in columns or columns[normalized]['nullable']:
                batch_op.alter_column(normalized, existing_type=sa.String(), nullable=False)
            if index_name not in indexes:
                batch_op.create_index(index_name, [normalized], unique=True)


def downgrade() -> None:
    with op.batch_alter_table('alias_well_log', schema=None) as batch_op:
        batch_op.drop_index('ix_alias_well_log_alias_name_norm')
        batch_op.drop_column('alias_name_norm')

    with op.batch_alter_table('canonical_well_log', schema=None) as batch_op:
        batch_op.drop_index('ix_canonical_well_log_canonical_name_norm')
        batch_op.drop_column('canonical_name_norm')
