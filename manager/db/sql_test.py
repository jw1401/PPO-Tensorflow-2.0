from sqlite_engine import db_session
from sql_config_model import Config
from sqlalchemy import exc

try:
    cfg = Config('config1', './Path/to/config1')
    db_session.add(cfg)
    db_session.commit()
except exc.SQLAlchemyError as ex:
    print("Na")

try:
    db_session.rollback()
    res = Config.query.all()
    print(res[0].path)
except Exception as ex:
    db_session.rollback()
    print("nana")

db_session.remove()
