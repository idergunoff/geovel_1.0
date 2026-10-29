"""add boundary and well data aliases

Revision ID: a7c8d9e0f1a2
Revises: 9b21c6e4a1f2
"""
from alembic import op
import sqlalchemy as sa

revision = 'a7c8d9e0f1a2'
down_revision = '9b21c6e4a1f2'
branch_labels = None
depends_on = None


def _create_table(name, *columns):
    if op.get_context().as_sql or not sa.inspect(op.get_bind()).has_table(name):
        op.create_table(name, *columns)


def _create_index(name, table, columns, **kwargs):
    indexes = set() if op.get_context().as_sql else {
        index['name'] for index in sa.inspect(op.get_bind()).get_indexes(table)
    }
    if name not in indexes:
        op.create_index(name, table, columns, **kwargs)


def _canonical_table(name):
    _create_table(name,
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('canonical_name', sa.String(), nullable=False),
        sa.Column('canonical_name_norm', sa.String(), nullable=False),
        sa.Column('description', sa.String()))
    _create_index(f'ix_{name}_canonical_name', name, ['canonical_name'], unique=True)
    _create_index(f'ix_{name}_canonical_name_norm', name, ['canonical_name_norm'], unique=True)


def _alias_table(name, canonical):
    _create_table(name,
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('alias_name', sa.String(), nullable=False),
        sa.Column('alias_name_norm', sa.String(), nullable=False),
        sa.Column('canonical_id', sa.Integer(), sa.ForeignKey(f'{canonical}.id'), nullable=False))
    _create_index(f'ix_{name}_alias_name', name, ['alias_name'], unique=True)
    _create_index(f'ix_{name}_alias_name_norm', name, ['alias_name_norm'], unique=True)
    _create_index(f'ix_{name}_canonical_id', name, ['canonical_id'])


def upgrade():
    _canonical_table('canonical_boundary')
    _alias_table('alias_boundary', 'canonical_boundary')
    _canonical_table('canonical_well_option')
    _alias_table('alias_well_option', 'canonical_well_option')


def downgrade():
    op.drop_table('alias_well_option')
    op.drop_table('canonical_well_option')
    op.drop_table('alias_boundary')
    op.drop_table('canonical_boundary')
