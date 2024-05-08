"""add_CRL_param_formation

Revision ID: 0bb5688f605e
Revises: 384aa0fbbff7
Create Date: 2024-05-06 16:24:49.314454

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0bb5688f605e'
down_revision: Union[str, None] = '384aa0fbbff7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'temp_formation',

        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('profile_id', sa.Integer, sa.ForeignKey('profile.id')),
        sa.Column('title', sa.String),
        sa.Column('up', sa.Integer, sa.ForeignKey('layers.id')),
        sa.Column('down', sa.Integer, sa.ForeignKey('layers.id')),
        sa.Column('T_top', sa.Text),
        sa.Column('T_bottom', sa.Text),
        sa.Column('dT', sa.Text),
        sa.Column('A_top', sa.Text),
        sa.Column('A_bottom', sa.Text),
        sa.Column('dA', sa.Text),
        sa.Column('A_sum', sa.Text),
        sa.Column('A_mean', sa.Text),
        sa.Column('dVt', sa.Text),
        sa.Column('Vt_top', sa.Text),
        sa.Column('Vt_sum', sa.Text),
        sa.Column('Vt_mean', sa.Text),
        sa.Column('dAt', sa.Text),
        sa.Column('At_top', sa.Text),
        sa.Column('At_sum', sa.Text),
        sa.Column('At_mean', sa.Text),
        sa.Column('dPht', sa.Text),
        sa.Column('Pht_top', sa.Text),
        sa.Column('Pht_sum', sa.Text),
        sa.Column('Pht_mean', sa.Text),
        sa.Column('Wt_top', sa.Text),
        sa.Column('Wt_mean', sa.Text),
        sa.Column('Wt_sum', sa.Text),

        sa.Column('width', sa.Text),
        sa.Column('top', sa.Text),
        sa.Column('land', sa.Text),
        sa.Column('speed', sa.Text),
        sa.Column('speed_cover', sa.Text),

        sa.Column('skew', sa.Text),
        sa.Column('kurt', sa.Text),
        sa.Column('std', sa.Text),
        sa.Column('k_var', sa.Text),

        sa.Column('A_max', sa.Text),
        sa.Column('Vt_max', sa.Text),
        sa.Column('At_max', sa.Text),
        sa.Column('Pht_max', sa.Text),
        sa.Column('Wt_max', sa.Text),

        sa.Column('A_T_max', sa.Text),
        sa.Column('Vt_T_max', sa.Text),
        sa.Column('At_T_max', sa.Text),
        sa.Column('Pht_T_max', sa.Text),
        sa.Column('Wt_T_max', sa.Text),

        sa.Column('A_Sn', sa.Text),
        sa.Column('Vt_Sn', sa.Text),
        sa.Column('At_Sn', sa.Text),
        sa.Column('Pht_Sn', sa.Text),
        sa.Column('Wt_Sn', sa.Text),

        sa.Column('A_wmf', sa.Text),
        sa.Column('Vt_wmf', sa.Text),
        sa.Column('At_wmf', sa.Text),
        sa.Column('Pht_wmf', sa.Text),
        sa.Column('Wt_wmf', sa.Text),

        sa.Column('A_Qf', sa.Text),
        sa.Column('Vt_Qf', sa.Text),
        sa.Column('At_Qf', sa.Text),
        sa.Column('Pht_Qf', sa.Text),
        sa.Column('Wt_Qf', sa.Text),

        sa.Column('A_Sn_wmf', sa.Text),
        sa.Column('Vt_Sn_wmf', sa.Text),
        sa.Column('At_Sn_wmf', sa.Text),
        sa.Column('Pht_Sn_wmf', sa.Text),
        sa.Column('Wt_Sn_wmf', sa.Text),

        sa.Column('k_r', sa.Text),

        # Add the new columns
        sa.Column('CRL_top', sa.Text, nullable=True),
        sa.Column('CRL_bottom', sa.Text, nullable=True),
        sa.Column('dCRL', sa.Text, nullable=True),
        sa.Column('CRL_sum', sa.Text, nullable=True),
        sa.Column('CRL_mean', sa.Text, nullable=True),
        sa.Column('CRL_max', sa.Text, nullable=True),
        sa.Column('CRL_T_max', sa.Text, nullable=True),
        sa.Column('CRL_Sn', sa.Text, nullable=True),
        sa.Column('CRL_wmf', sa.Text, nullable=True),
        sa.Column('CRL_Qf', sa.Text, nullable=True),
        sa.Column('CRL_Sn_wmf', sa.Text, nullable=True),

        sa.Column('CRL_skew', sa.Text, nullable=True),
        sa.Column('CRL_kurt', sa.Text, nullable=True),
        sa.Column('CRL_std', sa.Text, nullable=True),
        sa.Column('CRL_k_var', sa.Text, nullable=True),
    )

    op.execute('INSERT INTO temp_formation ('
               'id, '
               'profile_id, '
               'title, '
               'up, '
               'down, '
               'T_top, '
               'T_bottom, '
               'dT, '
               'A_top, '
               'A_bottom, '
               'dA, '
               'A_sum, '
               'A_mean, '
               'dVt, '
               'Vt_top, '
               'Vt_sum, '
               'Vt_mean, '
               'dAt, '
               'At_top, '
               'At_sum, '
               'At_mean, '
               'dPht, '
               'Pht_top, '
               'Pht_sum, '
               'Pht_mean, '
               'Wt_top, '
               'Wt_mean, '
               'Wt_sum, '

               'width, '
               'top, '
               'land, '
               'speed, '
               'speed_cover, '

               'skew, '
               'kurt, '
               'std, '
               'k_var, '

               'A_max, '
               'Vt_max, '
               'At_max, '
               'Pht_max, '
               'Wt_max, '

               'A_T_max, '
               'Vt_T_max, '
               'At_T_max, '
               'Pht_T_max, '
               'Wt_T_max, ' 

               'A_Sn, '
               'Vt_Sn, '
               'At_Sn, '
               'Pht_Sn, '
               'Wt_Sn, '

               'A_wmf, '
               'Vt_wmf, '
               'At_wmf, '
               'Pht_wmf, '
               'Wt_wmf, '

               'A_Qf, '
               'Vt_Qf, '
               'At_Qf, '
               'Pht_Qf, '
               'Wt_Qf, '

               'A_Sn_wmf, '
               'Vt_Sn_wmf, '
               'At_Sn_wmf, '
               'Pht_Sn_wmf, '
               'Wt_Sn_wmf, '

               'k_r) '
               
               'SELECT '
               
               'id, '
               'profile_id, '
               'title, '
               'up, '
               'down, '
               'T_top, '
               'T_bottom, '
               'dT, '
               'A_top, '
               'A_bottom, '
               'dA, '
               'A_sum, '
               'A_mean, '
               'dVt, '
               'Vt_top, '
               'Vt_sum, '
               'Vt_mean, '
               'dAt, '
               'At_top, '
               'At_sum, '
               'At_mean, '
               'dPht, '
               'Pht_top, '
               'Pht_sum, '
               'Pht_mean, '
               'Wt_top, '
               'Wt_mean, '
               'Wt_sum, '

               'width, '
               'top, '
               'land, '
               'speed, '
               'speed_cover, '

               'skew, '
               'kurt, '
               'std, '
               'k_var, '

               'A_max, '
               'Vt_max, '
               'At_max, '
               'Pht_max, '
               'Wt_max, '

               'A_T_max, '
               'Vt_T_max, '
               'At_T_max, '
               'Pht_T_max, '
               'Wt_T_max, ' 

               'A_Sn, '
               'Vt_Sn, '
               'At_Sn, '
               'Pht_Sn, '
               'Wt_Sn, '

               'A_wmf, '
               'Vt_wmf, '
               'At_wmf, '
               'Pht_wmf, '
               'Wt_wmf, '

               'A_Qf, '
               'Vt_Qf, '
               'At_Qf, '
               'Pht_Qf, '
               'Wt_Qf, '

               'A_Sn_wmf, '
               'Vt_Sn_wmf, '
               'At_Sn_wmf, '
               'Pht_Sn_wmf, '
               'Wt_Sn_wmf, '

               'k_r '
               
               'FROM formation'
               )


    # Drop the original table
    op.drop_table('formation')

    # Rename the temporary table to the original table's name
    op.rename_table('temp_formation', 'formation')


def downgrade() -> None:
    op.drop_column('formation', 'CRL_top')
    op.drop_column('formation','CRL_bottom')
    op.drop_column('formation','dCRL')
    op.drop_column('formation','CRL_sum')
    op.drop_column('formation','CRL_mean')
    op.drop_column('formation','CRL_max')
    op.drop_column('formation','CRL_T_max')
    op.drop_column('formation','CRL_Sn')
    op.drop_column('formation','CRL_wmf')
    op.drop_column('formation','CRL_Qf')
    op.drop_column('formation','CRL_Sn_wmf')

    op.drop_column('formation','CRL_skew')
    op.drop_column('formation','CRL_kurt')
    op.drop_column('formation','CRL_std')
    op.drop_column('formation','CRL_k_var')