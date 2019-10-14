from sqlalchemy import Column, Integer, String
from .sqlite_engine import Base


class Config(Base):
    
    __tablename__ = 'configs'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    path = Column(String(120), unique=True)

    def __init__(self, name=None, path=None):
        self.name = name
        self.path = path

    def __repr__(self):
        return '<Config %r>' % (self.name)
