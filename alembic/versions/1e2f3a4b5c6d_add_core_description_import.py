"""add core description import tables

Revision ID: 1e2f3a4b5c6d
Revises: c9d0e1f2a3b4, 0d253a325606
"""
from alembic import op
import sqlalchemy as sa

revision = '1e2f3a4b5c6d'
down_revision = ('c9d0e1f2a3b4', '0d253a325606')
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'core_description_document',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('well_id', sa.Integer(), sa.ForeignKey('well.id'), nullable=False),
        sa.Column('source_file_name', sa.String(), nullable=False),
        sa.Column('source_file_hash', sa.String(length=64), nullable=False),
        sa.Column('source_format', sa.String(), nullable=False),
        sa.Column('well_name_raw', sa.String()), sa.Column('area_name_raw', sa.String()),
        sa.Column('described_by_raw', sa.String()), sa.Column('described_by', sa.String()),
        sa.Column('parser_version', sa.String(), nullable=False),
        sa.Column('dictionary_version', sa.String()),
        sa.Column('match_method', sa.String(), nullable=False),
        sa.Column('match_confidence', sa.Float(), nullable=False),
        sa.Column('imported_at', sa.DateTime(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('diagnostics', sa.Text(), nullable=False),
        sa.UniqueConstraint('source_file_hash', name='uq_core_description_document_hash'),
    )
    op.create_index('ix_core_description_document_well_id', 'core_description_document', ['well_id'])
    op.create_index('ix_core_description_document_source_file_hash', 'core_description_document',
                    ['source_file_hash'], unique=True)
    op.create_table(
        'core_description_interval',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('document_id', sa.Integer(), sa.ForeignKey('core_description_document.id', ondelete='CASCADE'), nullable=False),
        sa.Column('well_id', sa.Integer(), sa.ForeignKey('well.id'), nullable=False),
        sa.Column('top_depth', sa.Float(), nullable=False),
        sa.Column('bottom_depth', sa.Float(), nullable=False),
        sa.Column('raw_description', sa.Text(), nullable=False),
        sa.Column('normalized_description', sa.Text(), nullable=False),
        sa.Column('oil_saturation', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('needs_review', sa.Boolean(), nullable=False),
        sa.Column('source_table_index', sa.Integer(), nullable=False),
        sa.Column('source_row_index', sa.Integer(), nullable=False),
        sa.Column('manually_edited', sa.Boolean(), nullable=False),
        sa.Column('recognition_details', sa.Text(), nullable=False),
        sa.CheckConstraint('top_depth < bottom_depth', name='ck_core_description_interval_depths'),
    )
    op.create_index('ix_core_description_interval_document_id', 'core_description_interval', ['document_id'])
    op.create_index('ix_core_description_interval_well_id', 'core_description_interval', ['well_id'])
    op.create_index('ix_core_description_interval_well_depths', 'core_description_interval',
                    ['well_id', 'top_depth', 'bottom_depth'])
    op.create_table(
        'core_description_rock',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('interval_id', sa.Integer(), sa.ForeignKey('core_description_interval.id', ondelete='CASCADE'), nullable=False),
        sa.Column('rock_name', sa.String(), nullable=False), sa.Column('role', sa.String(), nullable=False),
        sa.Column('oil_saturation', sa.String()), sa.Column('source_text', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('match_start', sa.Integer()), sa.Column('match_end', sa.Integer()),
    )
    op.create_index('ix_core_description_rock_interval_id', 'core_description_rock', ['interval_id'])


def downgrade():
    op.drop_index('ix_core_description_rock_interval_id', table_name='core_description_rock')
    op.drop_table('core_description_rock')
    op.drop_index('ix_core_description_interval_well_depths', table_name='core_description_interval')
    op.drop_index('ix_core_description_interval_well_id', table_name='core_description_interval')
    op.drop_index('ix_core_description_interval_document_id', table_name='core_description_interval')
    op.drop_table('core_description_interval')
    op.drop_index('ix_core_description_document_source_file_hash', table_name='core_description_document')
    op.drop_index('ix_core_description_document_well_id', table_name='core_description_document')
    op.drop_table('core_description_document')
