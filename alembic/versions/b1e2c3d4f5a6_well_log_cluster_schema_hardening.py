"""well log cluster schema hardening

Revision ID: b1e2c3d4f5a6
Revises: 9b1f2a6c4d31
Create Date: 2026-05-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b1e2c3d4f5a6'
down_revision: Union[str, None] = '9b1f2a6c4d31'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _harden_table(table, columns, indexes=(), unique_constraint=None):
    # create_all() may already have created this table using the newer model.
    inspector = sa.inspect(op.get_bind())
    existing_columns = {column['name']: column for column in inspector.get_columns(table)}
    existing_indexes = {index['name'] for index in inspector.get_indexes(table)}
    existing_constraints = {constraint['name'] for constraint in inspector.get_unique_constraints(table)}
    with op.batch_alter_table(table, schema=None) as batch_op:
        for name, column_type in columns:
            if existing_columns[name]['nullable']:
                batch_op.alter_column(name, existing_type=column_type, nullable=False)
        for name, column_names, unique in indexes:
            if name not in existing_indexes:
                batch_op.create_index(name, column_names, unique=unique)
        if unique_constraint is not None:
            name, column_names = unique_constraint
            if name not in existing_constraints:
                batch_op.create_unique_constraint(name, column_names)


def upgrade() -> None:
    columns = {column['name'] for column in sa.inspect(op.get_bind()).get_columns('well_log_cluster_dataset')}
    if 'created_at' not in columns:
        with op.batch_alter_table('well_log_cluster_dataset', schema=None) as batch_op:
            batch_op.alter_column('created_by', new_column_name='created_at', existing_type=sa.DateTime(), nullable=False)
    _harden_table('well_log_cluster_dataset', [('name', sa.String()), ('created_at', sa.DateTime())])
    _harden_table(
        'well_for_cluster',
        [('dataset_id', sa.Integer()), ('well_id', sa.Integer()), ('top_md', sa.Float()), ('bottom_md', sa.Float())],
        [('ix_well_for_cluster_dataset_id', ['dataset_id'], False),
         ('ix_well_for_cluster_well_id', ['well_id'], False)],
        ('uq_well_for_cluster_dataset_well', ['dataset_id', 'well_id']),
    )
    _harden_table(
        'canonical_well_log', [('canonical_name', sa.String())],
        [('ix_canonical_well_log_canonical_name', ['canonical_name'], True)],
    )
    _harden_table(
        'alias_well_log', [('alias_name', sa.String()), ('canonical_id', sa.Integer())],
        [('ix_alias_well_log_alias_name', ['alias_name'], True),
         ('ix_alias_well_log_canonical_id', ['canonical_id'], False)],
    )
    _harden_table(
        'feature_calculator',
        [('feature_name', sa.String()), ('used_canonical_well_log', sa.Text()),
         ('transform_type', sa.String()), ('params_json', sa.Text()), ('created_at', sa.DateTime())],
        [('ix_feature_calculator_feature_name', ['feature_name'], True)],
    )
    _harden_table(
        'cluster_well_log_parameter', [('dataset_id', sa.Integer()), ('canonical_id', sa.Integer())],
        [('ix_cluster_well_log_parameter_dataset_id', ['dataset_id'], False),
         ('ix_cluster_well_log_parameter_canonical_id', ['canonical_id'], False)],
        ('uq_cluster_well_log_param_dataset_canonical', ['dataset_id', 'canonical_id']),
    )
    _harden_table(
        'cluster_well_log_parameter_from_calculator', [('dataset_id', sa.Integer()), ('calculator_id', sa.Integer())],
        [('ix_cluster_well_log_parameter_from_calculator_dataset_id', ['dataset_id'], False),
         ('ix_cluster_well_log_parameter_from_calculator_calculator_id', ['calculator_id'], False)],
        ('uq_cluster_well_log_param_calc_dataset_calculator', ['dataset_id', 'calculator_id']),
    )
    _harden_table(
        'cluster_well_log_dataset_data', [('dataset_id', sa.Integer()), ('data', sa.Text())],
        [('ix_cluster_well_log_dataset_data_dataset_id', ['dataset_id'], False)],
    )


def downgrade() -> None:
    with op.batch_alter_table('cluster_well_log_dataset_data', schema=None) as batch_op:
        batch_op.drop_index('ix_cluster_well_log_dataset_data_dataset_id')
        batch_op.alter_column('data', existing_type=sa.Text(), nullable=True)
        batch_op.alter_column('dataset_id', existing_type=sa.Integer(), nullable=True)

    with op.batch_alter_table('cluster_well_log_parameter_from_calculator', schema=None) as batch_op:
        batch_op.drop_constraint('uq_cluster_well_log_param_calc_dataset_calculator', type_='unique')
        batch_op.drop_index('ix_cluster_well_log_parameter_from_calculator_calculator_id')
        batch_op.drop_index('ix_cluster_well_log_parameter_from_calculator_dataset_id')
        batch_op.alter_column('calculator_id', existing_type=sa.Integer(), nullable=True)
        batch_op.alter_column('dataset_id', existing_type=sa.Integer(), nullable=True)

    with op.batch_alter_table('cluster_well_log_parameter', schema=None) as batch_op:
        batch_op.drop_constraint('uq_cluster_well_log_param_dataset_canonical', type_='unique')
        batch_op.drop_index('ix_cluster_well_log_parameter_canonical_id')
        batch_op.drop_index('ix_cluster_well_log_parameter_dataset_id')
        batch_op.alter_column('canonical_id', existing_type=sa.Integer(), nullable=True)
        batch_op.alter_column('dataset_id', existing_type=sa.Integer(), nullable=True)

    with op.batch_alter_table('feature_calculator', schema=None) as batch_op:
        batch_op.drop_index('ix_feature_calculator_feature_name')
        batch_op.alter_column('created_at', existing_type=sa.DateTime(), nullable=True)
        batch_op.alter_column('params_json', existing_type=sa.Text(), nullable=True)
        batch_op.alter_column('transform_type', existing_type=sa.String(), nullable=True)
        batch_op.alter_column('used_canonical_well_log', existing_type=sa.Text(), nullable=True)
        batch_op.alter_column('feature_name', existing_type=sa.String(), nullable=True)

    with op.batch_alter_table('alias_well_log', schema=None) as batch_op:
        batch_op.drop_index('ix_alias_well_log_canonical_id')
        batch_op.drop_index('ix_alias_well_log_alias_name')
        batch_op.alter_column('canonical_id', existing_type=sa.Integer(), nullable=True)
        batch_op.alter_column('alias_name', existing_type=sa.String(), nullable=True)

    with op.batch_alter_table('canonical_well_log', schema=None) as batch_op:
        batch_op.drop_index('ix_canonical_well_log_canonical_name')
        batch_op.alter_column('canonical_name', existing_type=sa.String(), nullable=True)

    with op.batch_alter_table('well_for_cluster', schema=None) as batch_op:
        batch_op.drop_constraint('uq_well_for_cluster_dataset_well', type_='unique')
        batch_op.drop_index('ix_well_for_cluster_well_id')
        batch_op.drop_index('ix_well_for_cluster_dataset_id')
        batch_op.alter_column('bottom_md', existing_type=sa.Float(), nullable=True)
        batch_op.alter_column('top_md', existing_type=sa.Float(), nullable=True)
        batch_op.alter_column('well_id', existing_type=sa.Integer(), nullable=True)
        batch_op.alter_column('dataset_id', existing_type=sa.Integer(), nullable=True)

    with op.batch_alter_table('well_log_cluster_dataset', schema=None) as batch_op:
        batch_op.alter_column('created_at', new_column_name='created_by', existing_type=sa.DateTime(), nullable=True)
        batch_op.alter_column('name', existing_type=sa.String(), nullable=True)
