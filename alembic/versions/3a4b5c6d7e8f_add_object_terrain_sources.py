"""Object terrain assignments and source tracking; keep training snapshots intact."""
from alembic import op
import sqlalchemy as sa

revision = '3a4b5c6d7e8f'
down_revision = '2f3a4b5c6d7e'
branch_labels = None
depends_on = None


def upgrade():
    additions = {
        'georadar_object': [sa.Column('relief_grid_id', sa.Integer(), nullable=True)],
        'common_grid': [sa.Column('comment', sa.Text()), sa.Column('source_name', sa.String()),
                        sa.Column('created_at', sa.DateTime()),
                        sa.Column('is_local', sa.Boolean(), nullable=False, server_default=sa.false())],
        'profile': [sa.Column('relief_source_key', sa.String())],
        'formation': [sa.Column('land_source_key', sa.String())],
        'profile_model_prediction': [sa.Column('is_stale', sa.Boolean(), nullable=False, server_default=sa.false())],
    }
    for table, columns in additions.items():
        if not op.get_context().as_sql and not sa.inspect(op.get_bind()).has_table(table):
            continue
        existing = set() if op.get_context().as_sql else {
            column['name'] for column in sa.inspect(op.get_bind()).get_columns(table)
        }
        for column in columns:
            if column.name not in existing:
                op.add_column(table, column)
    if not op.get_context().as_sql:
        if not sa.inspect(op.get_bind()).has_table('georadar_object'):
            return
        constraints = sa.inspect(op.get_bind()).get_foreign_keys('georadar_object')
        if any(fk['constrained_columns'] == ['relief_grid_id'] for fk in constraints):
            return
    offline_table = None
    if op.get_context().as_sql:
        metadata = sa.MetaData()
        sa.Table('common_grid', metadata, sa.Column('id', sa.Integer(), primary_key=True))
        offline_table = sa.Table('georadar_object', metadata,
                                sa.Column('id', sa.Integer(), primary_key=True),
                                sa.Column('title', sa.String()),
                                sa.Column('relief_grid_id', sa.Integer()))
    with op.batch_alter_table('georadar_object', copy_from=offline_table) as batch:
        batch.create_foreign_key('fk_object_relief_grid', 'common_grid', ['relief_grid_id'], ['id'])


def downgrade():
    with op.batch_alter_table('georadar_object') as batch:
        batch.drop_column('relief_grid_id')
    for table, columns in {
        'profile_model_prediction': ['is_stale'], 'formation': ['land_source_key'],
        'profile': ['relief_source_key'], 'common_grid': ['is_local', 'created_at', 'source_name', 'comment'],
    }.items():
        for column in columns:
            op.drop_column(table, column)
