from sqlalchemy import Column, Integer, String, Boolean, Date, Enum, Numeric
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()

class Location(Base):
    __tablename__ = "Locations"

    key = Column(Integer, primary_key=True)
    street_name_highway = Column(String)
    intersection_1 = Column(String)
    intersection_2 = Column(String)
    longitude = Column(Numeric)
    latitude = Column(Numeric)
    neighbourhood = Column(String)
