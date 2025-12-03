"""drop_table_lineup_train

Revision ID: df983186961d
Revises: 436670216df8
Create Date: 2025-12-03 10:34:22.864424

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'df983186961d'
down_revision: Union[str, None] = '436670216df8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table('lineup_train')


def downgrade() -> None:
    op.create_table('lineup_train',
                    sa.Column('id', sa.Integer(), primary_key=True),
                    sa.Column('type_ml', sa.String()),
                    sa.Column('analysis_id', sa.Integer()),
                    sa.Column('list_param', sa.Text()),
                    sa.Column('list_param_short', sa.Text()),
                    sa.Column('except_signal', sa.String()),
                    sa.Column('except_crl', sa.String()),
                    sa.Column('text_model', sa.Text()),
                    sa.Column('model_name', sa.String()),
                    sa.Column('pipe', sa.LargeBinary()),
                    sa.Column('over_sampling', sa.String()),
                    sa.Column('random_seed', sa.Integer(), server_default='0'),
                    sa.Column('cvw', sa.Boolean(), server_default='false')
                    )
