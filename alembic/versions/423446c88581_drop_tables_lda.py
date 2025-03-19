"""drop tables LDA

Revision ID: 423446c88581
Revises: 7669d3b0c9d8
Create Date: 2025-03-19 14:55:17.559558

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '423446c88581'
down_revision: Union[str, None] = '7669d3b0c9d8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Удаляем таблицы в правильном порядке, чтобы избежать ошибок связанных с внешними ключами
    op.drop_table('markup_lda')
    op.drop_table('marker_lda')
    op.drop_table('parameter_lda')
    op.drop_table('analysis_lda')


def downgrade():
    # Восстанавливаем таблицы в обратном порядке
    op.create_table(
        'analysis_lda',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('title', sa.String),
        sa.Column('data', sa.Text),
        sa.Column('up_data', sa.Boolean, default=False)
    )

    op.create_table(
        'parameter_lda',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('analysis_id', sa.Integer, sa.ForeignKey('analysis_lda.id')),
        sa.Column('parameter', sa.String)
    )

    op.create_table(
        'marker_lda',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('analysis_id', sa.Integer, sa.ForeignKey('analysis_lda.id')),
        sa.Column('title', sa.String),
        sa.Column('color', sa.String)
    )

    op.create_table(
        'markup_lda',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('analysis_id', sa.Integer, sa.ForeignKey('analysis_lda.id')),
        sa.Column('well_id', sa.Integer, sa.ForeignKey('well.id')),
        sa.Column('profile_id', sa.Integer, sa.ForeignKey('profile.id')),
        sa.Column('formation_id', sa.Integer, sa.ForeignKey('formation.id')),
        sa.Column('marker_id', sa.Integer, sa.ForeignKey('marker_lda.id')),
        sa.Column('list_measure', sa.Text),
        sa.Column('list_fake', sa.Text),
        sa.Column('type_markup', sa.String)
    )
