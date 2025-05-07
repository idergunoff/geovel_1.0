"""update_model_hashes_in_trained_models

Revision ID: 436670216df8
Revises: 37ccb495425d
Create Date: 2025-05-06 16:08:39.655099

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text
import os
import hashlib


# revision identifiers, used by Alembic.
revision: str = '436670216df8'
down_revision: Union[str, None] = '37ccb495425d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def get_model_hash(file_path: str, length: int = 12) -> str:
    try:
        # Нормализация пути и контента
        abs_path = os.path.abspath(os.path.normpath(file_path))
        with open(abs_path, 'rb') as f:
            content = f.read().replace(b'\r\n', b'\n')  # Унификация переводов строк
            return hashlib.md5(content).hexdigest()[:length]
    except:
        return ""


def upgrade():
    conn = op.get_bind()
    # Обновляем хэши для всех моделей
    for table in ['trained_model_class', 'trained_model_reg']:
        for id, path in conn.execute(text(f"SELECT id, path_model FROM {table}")):
            if path and os.path.exists(path):
                new_hash = get_model_hash(path)  # Новая реализация
                conn.execute(
                    text(f"UPDATE {table} SET model_hash = :hash WHERE id = :id"),
                    {"hash": new_hash, "id": id}
                )


def downgrade():
    conn = op.get_bind()
    for table in ['trained_model_class', 'trained_model_reg']:
        conn.execute(
            text(f"UPDATE {table} SET model_hash = NULL")  # Обнуляем хэши
        )
