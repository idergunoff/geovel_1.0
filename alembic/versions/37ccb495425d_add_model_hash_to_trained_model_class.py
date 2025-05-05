"""add_model_hash_to_trained_model_class

Revision ID: 37ccb495425d
Revises: ca5164e1ef65
Create Date: 2025-05-05 08:23:16.279800

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from models_db.model import *
import hashlib
import os
import pickle


# revision identifiers, used by Alembic.
revision: str = '37ccb495425d'
down_revision: Union[str, None] = 'ca5164e1ef65'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def get_model_hash(file_path: str, length=12) -> str:
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(pickle.dumps(pickle.load(f))).hexdigest()[:length]
    except:
        return ""


def upgrade():
    op.add_column('trained_model_class', sa.Column('model_hash', sa.String(12)))
    op.add_column('trained_model_reg', sa.Column('model_hash', sa.String(12)))

    # Запонение хэшей для существующих записей:
    conn = op.get_bind()

    # Classification
    rows_class = conn.execute(text("SELECT id, path_model FROM trained_model_class WHERE path_model IS NOT NULL"))

    for id, path in rows_class:
        if path and os.path.exists(path):
            model_hash = get_model_hash(path)
            conn.execute(text("UPDATE trained_model_class SET model_hash = :hash WHERE id = :id"),
                         {'hash': model_hash, 'id': id})

    # Regression
    rows_reg = conn.execute(text("SELECT id, path_model FROM trained_model_reg WHERE path_model IS NOT NULL"))

    for id, path in rows_reg:
        if path and os.path.exists(path):
            model_hash = get_model_hash(path)
            conn.execute(text("UPDATE trained_model_reg SET model_hash = :hash WHERE id = :id"),
                         {'hash': model_hash, 'id': id})


def downgrade():
    # Удаляем столбец model_hash при откате миграции
    op.drop_column('trained_model_class', 'model_hash')
    op.drop_column('trained_model_reg', 'model_hash')






