"""add abs and depth relief to Profile

Revision ID: d3d9fd3cfa6e
Revises: a3b04e73f9bb
Create Date: 2024-02-20 08:56:02.486696

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd3d9fd3cfa6e'
down_revision: Union[str, None] = 'a3b04e73f9bb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Create a temporary table with the same structure as the original table
    op.create_table(
        'temp_profile',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('research_id', sa.Integer, sa.ForeignKey('research.id')),
        # Add other columns as needed, make sure to match the original table's structure
        sa.Column('title', sa.String),
        sa.Column('signal', sa.Text),
        sa.Column('x_wgs', sa.Text),
        sa.Column('y_wgs', sa.Text),
        sa.Column('x_pulc', sa.Text),
        sa.Column('y_pulc', sa.Text),
        # Add the new columns abs_relief and depth_relief
        sa.Column('abs_relief', sa.Text, nullable=True),
        sa.Column('depth_relief', sa.Text, nullable=True),
    )

    # Copy data from the original table to the temporary table
    op.execute('INSERT INTO temp_profile (id, research_id, title, signal, x_wgs, y_wgs, x_pulc, y_pulc) SELECT id, '
               'research_id, title, signal, x_wgs, y_wgs, x_pulc, y_pulc FROM profile')


    # Drop the original table
    op.drop_table('profile')

    # Rename the temporary table to the original table's name
    op.rename_table('temp_profile', 'profile')


def downgrade():
    # This is a destructive operation, downgrade won't restore the dropped foreign key constraint
    raise NotImplementedError("Downgrade not supported for this migration.")
