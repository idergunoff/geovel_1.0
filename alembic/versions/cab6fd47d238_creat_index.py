"""creat index

Revision ID: cab6fd47d238
Revises: 98346a821fa7
Create Date: 2024-02-22 14:36:52.529219

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cab6fd47d238'
down_revision: Union[str, None] = '98346a821fa7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(op.f('ix_object_id'), 'georadar_object', ['id'], unique=True)
    op.create_index(op.f('ix_research_id'), 'research', ['id'], unique=True)
    op.create_index(op.f('ix_profile_id'), 'profile', ['id'], unique=True)
    op.create_index(op.f('deep_profile_id'), 'deep_profile', ['id'], unique=True)
    op.create_index(op.f('ix_deep_layer_id'), 'deep_layer', ['id'])
    op.create_index(op.f('ix_binding_id'), 'binding', ['id'])
    op.create_index(op.f('ix_velocity_formation_id'), 'velocity_formation', ['id'])
    op.create_index(op.f('ix_velocity_model_id'), 'velocity_model', ['id'])
    op.create_index(op.f('ix_current_profile_id'), 'current_profile', ['id'])
    op.create_index(op.f('ix_current_profile_min_max_id'), 'current_profile_min_max', ['id'])
    op.create_index(op.f('ix_current_velocity_model_id'), 'current_velocity_model', ['id'])
    op.create_index(op.f('ix_grid_id'), 'grid', ['id'])
    op.create_index(op.f('ix_fft_spectr_id'), 'fft_spectr', ['id'])
    op.create_index(op.f('ix_window_profile_id'), 'window_profile', ['id'])
    op.create_index(op.f('ix_layers_id'), 'layers', ['id'])
    op.create_index(op.f('ix_points_of_layer_id'), 'points_of_layer', ['id'])
    op.create_index(op.f('ix_formation_id'), 'formation', ['id'])
    op.create_index(op.f('ix_well_id'), 'well', ['id'])
    op.create_index(op.f('ix_boundary_id'), 'boundary', ['id'])
    op.create_index(op.f('ix_well_optionally_id'), 'well_optionally', ['id'])
    op.create_index(op.f('ix_well_log_id'), 'well_log', ['id'])
    op.create_index(op.f('ix_analysis_lda_id'), 'analysis_lda', ['id'])
    op.create_index(op.f('ix_parameter_lda_id'), 'parameter_lda', ['id'])
    op.create_index(op.f('ix_marker_lda_id'), 'marker_lda', ['id'])
    op.create_index(op.f('ix_markup_lda_id'), 'markup_lda', ['id'])
    op.create_index(op.f('ix_analysis_mlp_id'), 'analysis_mlp', ['id'])
    op.create_index(op.f('ix_analysis_reg_id'), 'analysis_reg', ['id'])
    op.create_index(op.f('ix_horizontal_well_id'), 'horizontal_well', ['id'])
    op.create_index(op.f('ix_model_formation_ai_id'), 'model_formation_ai', ['id'])
    op.create_index(op.f('ix_trained_model_id'), 'trained_model', ['id'])
    op.create_index(op.f('ix_exploration_id'), 'exploration', ['id'])
    op.create_index(op.f('ix_analysis_exploration_id'), 'analysis_exploration', ['id'])
    op.create_index(op.f('ix_geochem_id'), 'geochem', ['id'])



def downgrade() -> None:
    op.drop_index(op.f('ix_object_id'), table_name='georadar_object')
    op.drop_index(op.f('ix_research_id'), table_name='research')
    op.drop_index(op.f('ix_profile_id'), table_name='profile')
    op.drop_index(op.f('deep_profile_id'), table_name='deep_profile')
    op.drop_index(op.f('ix_deep_layer_id'), table_name='deep_layer')
    op.drop_index(op.f('ix_binding_id'), table_name='binding')
    op.drop_index(op.f('ix_velocity_formation_id'), table_name='velocity_formation')
    op.drop_index(op.f('ix_velocity_model_id'), table_name='velocity_model')
    op.drop_index(op.f('ix_current_profile_id'), table_name='current_profile')
    op.drop_index(op.f('ix_current_profile_min_max_id'), table_name='current_profile_min_max')
    op.drop_index(op.f('ix_current_velocity_model_id'), table_name='current_velocity_model')
    op.drop_index(op.f('ix_grid_id'), table_name='grid')
    op.drop_index(op.f('ix_fft_spectr_id'), table_name='fft_spectr')
    op.drop_index(op.f('ix_window_profile_id'), table_name='window_profile')
    op.drop_index(op.f('ix_layers_id'), table_name='layers')
    op.drop_index(op.f('ix_points_of_layer_id'), table_name='points_of_layer')
    op.drop_index(op.f('ix_formation_id'), table_name='formation')
    op.drop_index(op.f('ix_well_id'), table_name='well')
    op.drop_index(op.f('ix_boundary_id'), table_name='boundary')
    op.drop_index(op.f('ix_well_optionally_id'), table_name='well_optionally')
    op.drop_index(op.f('ix_well_log_id'), table_name='well_log')
    op.drop_index(op.f('ix_analysis_lda_id'), table_name='analysis_lda')
    op.drop_index(op.f('ix_parameter_lda_id'), table_name='parameter_lda')
    op.drop_index(op.f('ix_marker_lda_id'), table_name='marker_lda')
    op.drop_index(op.f('ix_markup_lda_id'), table_name='markup_lda')
    op.drop_index(op.f('ix_analysis_mlp_id'), table_name='analysis_mlp')
    op.drop_index(op.f('ix_analysis_reg_id'), table_name='analysis_reg')
    op.drop_index(op.f('ix_horizontal_well_id'), table_name='horizontal_well')
    op.drop_index(op.f('ix_model_formation_ai_id'), table_name='model_formation_ai')
    op.drop_index(op.f('ix_trained_model_id'), table_name='trained_model')
    op.drop_index(op.f('ix_exploration_id'), table_name='exploration')
    op.drop_index(op.f('ix_analysis_exploration_id'), table_name='analysis_exploration')
    op.drop_index(op.f('ix_geochem_id'), table_name='geochem')





