"""geochem point add column fake, drop table geochempointfake

Revision ID: df8ce367b1c8
Revises: d3d9fd3cfa6e
Create Date: 2024-02-20 11:02:22.095112

"""
from typing import Sequence, Union

import sqlalchemy.exc
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'df8ce367b1c8'
down_revision: Union[str, None] = 'd3d9fd3cfa6e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'temp_geochem_point',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('geochem_id', sa.Integer, sa.ForeignKey('geochem.id')),
        sa.Column('title', sa.String),
        sa.Column('x_coord', sa.Float),
        sa.Column('y_coord', sa.Float),
        sa.Column('fake', sa.Boolean, default=False),
    )

    op.execute('INSERT INTO temp_geochem_point (id, geochem_id, title, x_coord, y_coord) SELECT id, '
               'geochem_id, title, x_coord, y_coord FROM geochem_point')

    op.drop_table('geochem_point')

    op.rename_table('temp_geochem_point', 'geochem_point')

    try:
        op.drop_table('geochem_point_fake')
    except sqlalchemy.exc.OperationalError:
        pass


def downgrade() -> None:
    op.create_table(
        'geochem_point_fake',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('geochem_point_id', sa.Integer, sa.ForeignKey('geochem_point.id')),
    )

    op.drop_column('geochem_point', 'fake')

