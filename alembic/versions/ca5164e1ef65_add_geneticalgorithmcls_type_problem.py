"""add GeneticAlgorithmCLS type problem

Revision ID: ca5164e1ef65
Revises: 06286832be7d
Create Date: 2025-04-24 09:07:02.179002

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ca5164e1ef65'
down_revision: Union[str, None] = '06286832be7d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Создаём временную таблицу с новой колонкой type_problem
    op.create_table(
        'temp_genetic_algorithm_cls',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('analysis_id', sa.Integer, sa.ForeignKey('analysis_mlp.id')),
        sa.Column('title', sa.String),
        sa.Column('pipeline', sa.Text),
        sa.Column('checkfile_path', sa.String),
        sa.Column('list_params', sa.Text),
        sa.Column('population_size', sa.Integer),
        sa.Column('comment', sa.Text),
        sa.Column('type_problem', sa.String, nullable=False),  # новая колонка
    )

    # Копируем данные из старой таблицы, добавляя значение 'min' в новую колонку
    op.execute("""
        INSERT INTO temp_genetic_algorithm_cls (
            id, analysis_id, title, pipeline, checkfile_path, list_params, population_size, comment, type_problem
        )
        SELECT
            id, analysis_id, title, pipeline, checkfile_path, list_params, population_size, comment, 'min'
        FROM genetic_algorithm_cls
    """)

    # Удаляем старую таблицу
    op.drop_table('genetic_algorithm_cls')

    # Переименовываем временную таблицу в оригинальное имя
    op.rename_table('temp_genetic_algorithm_cls', 'genetic_algorithm_cls')


def downgrade() -> None:
    op.drop_column('genetic_algorithm_cls', 'type_problem')
