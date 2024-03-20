"""update lineup_train add cvw

Revision ID: 384aa0fbbff7
Revises: cab6fd47d238
Create Date: 2024-03-19 15:02:09.725391

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '384aa0fbbff7'
down_revision: Union[str, None] = 'cab6fd47d238'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'temp_lineup_train',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('type_ml', sa.String),
        sa.Column('analysis_id', sa.Integer, sa.ForeignKey('analysis_mlp.id')),
        sa.Column('list_param', sa.Text),
        sa.Column('list_param_short', sa.Text),
        sa.Column('except_signal', sa.String),
        sa.Column('except_crl', sa.String),
        sa.Column('text_model', sa.Text),
        sa.Column('model_name', sa.String),
        sa.Column('pipe', sa.LargeBinary),
        sa.Column('over_sampling', sa.String),
        sa.Column('random_seed', sa.Integer, default=0),
        sa.Column('cvw', sa.Boolean, default=False)
    )

    op.execute('INSERT INTO temp_lineup_train (id, type_ml, analysis_id, list_param, list_param_short, except_signal, '
               'except_crl, text_model, model_name, pipe, over_sampling) SELECT id, type_ml, '
               'analysis_id, list_param, list_param_short, except_signal, except_crl, text_model, model_name, pipe, '
               'over_sampling FROM lineup_train')

    op.drop_table('lineup_train')
    op.rename_table('temp_lineup_train', 'lineup_train')


def downgrade() -> None:
    op.drop_column('lineup_train', 'cvw')
    op.drop_column('lineup_train', 'random_seed')
