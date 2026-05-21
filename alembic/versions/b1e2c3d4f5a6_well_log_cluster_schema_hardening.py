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


def upgrade() -> None:
    with op.batch_alter_table('well_log_cluster_dataset', schema=None) as batch_op:
        batch_op.alter_column('name', existing_type=sa.String(), nullable=False)
        batch_op.alter_column('created_by', new_column_name='created_at', existing_type=sa.DateTime(), nullable=False)

    with op.batch_alter_table('well_for_cluster', schema=None) as batch_op:
        batch_op.alter_column('dataset_id', existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column('well_id', existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column('top_md', existing_type=sa.Float(), nullable=False)
        batch_op.alter_column('bottom_md', existing_type=sa.Float(), nullable=False)
        batch_op.create_index('ix_well_for_cluster_dataset_id', ['dataset_id'], unique=False)
        batch_op.create_index('ix_well_for_cluster_well_id', ['well_id'], unique=False)
        batch_op.create_unique_constraint('uq_well_for_cluster_dataset_well', ['dataset_id', 'well_id'])

    with op.batch_alter_table('canonical_well_log', schema=None) as batch_op:
        batch_op.alter_column('canonical_name', existing_type=sa.String(), nullable=False)
        batch_op.create_index('ix_canonical_well_log_canonical_name', ['canonical_name'], unique=True)

    with op.batch_alter_table('alias_well_log', schema=None) as batch_op:
        batch_op.alter_column('alias_name', existing_type=sa.String(), nullable=False)
        batch_op.alter_column('canonical_id', existing_type=sa.Integer(), nullable=False)
        batch_op.create_index('ix_alias_well_log_alias_name', ['alias_name'], unique=True)
        batch_op.create_index('ix_alias_well_log_canonical_id', ['canonical_id'], unique=False)

    with op.batch_alter_table('feature_calculator', schema=None) as batch_op:
        batch_op.alter_column('feature_name', existing_type=sa.String(), nullable=False)
        batch_op.alter_column('used_canonical_well_log', existing_type=sa.Text(), nullable=False)
        batch_op.alter_column('transform_type', existing_type=sa.String(), nullable=False)
        batch_op.alter_column('params_json', existing_type=sa.Text(), nullable=False)
        batch_op.alter_column('created_at', existing_type=sa.DateTime(), nullable=False)
        batch_op.create_index('ix_feature_calculator_feature_name', ['feature_name'], unique=True)

    with op.batch_alter_table('cluster_well_log_parameter', schema=None) as batch_op:
        batch_op.alter_column('dataset_id', existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column('canonical_id', existing_type=sa.Integer(), nullable=False)
        batch_op.create_index('ix_cluster_well_log_parameter_dataset_id', ['dataset_id'], unique=False)
        batch_op.create_index('ix_cluster_well_log_parameter_canonical_id', ['canonical_id'], unique=False)
        batch_op.create_unique_constraint('uq_cluster_well_log_param_dataset_canonical', ['dataset_id', 'canonical_id'])

    with op.batch_alter_table('cluster_well_log_parameter_from_calculator', schema=None) as batch_op:
        batch_op.alter_column('dataset_id', existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column('calculator_id', existing_type=sa.Integer(), nullable=False)
        batch_op.create_index('ix_cluster_well_log_parameter_from_calculator_dataset_id', ['dataset_id'], unique=False)
        batch_op.create_index('ix_cluster_well_log_parameter_from_calculator_calculator_id', ['calculator_id'], unique=False)
        batch_op.create_unique_constraint('uq_cluster_well_log_param_calc_dataset_calculator', ['dataset_id', 'calculator_id'])

    with op.batch_alter_table('cluster_well_log_dataset_data', schema=None) as batch_op:
        batch_op.alter_column('dataset_id', existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column('data', existing_type=sa.Text(), nullable=False)
        batch_op.create_index('ix_cluster_well_log_dataset_data_dataset_id', ['dataset_id'], unique=False)


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
