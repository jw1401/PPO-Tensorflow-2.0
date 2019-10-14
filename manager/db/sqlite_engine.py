from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base


engine = create_engine('sqlite:///./manager/db/test.db', convert_unicode=True, connect_args={'check_same_thread': False})
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()


def init_db():

    import manager.db.sql_config_model
    Base.metadata.create_all(bind=engine)

init_db()

# engine.execute('DROP TABLE IF EXISTS users')
