from func import *

excel_file_path = "Геохимия.xlsx"
df = pd.read_excel(excel_file_path)

list_cols = df.columns.tolist()

# new_expl = Exploration(title='new_expl', object_id=1, date_explore=datetime.datetime.now())
# session.add(new_expl)
#
#
session.query(ParameterExploration).delete()
session.query(PointExploration).delete()
session.query(ParameterPoint).delete()
session.commit()

expl = session.query(Exploration).first()
point = SetPoints(exploration_id=expl.id, title="set_point")
session.add(point)
session.commit()

for i in df.index:
    p = PointExploration(set_points_id=point.id, x_coord=df.loc[i, list_cols[1]],
                         y_coord=df.loc[i, list_cols[2]],
                         title=str(df.loc[i, list_cols[0]]))
    session.add(p)
    for j, el in enumerate(list_cols):
        flag = 0
        old_param = session.query(ParameterExploration).filter(ParameterExploration.parameter == el).first()
        if not old_param:
            old_param = ParameterExploration(exploration_id=expl.id, parameter=el)
            session.add(old_param)
            session.commit()

        par_point = ParameterPoint(point_id=p.id, param_id=old_param.id, value=df.loc[i, list_cols[j]])
        session.add(par_point)


session.commit()
