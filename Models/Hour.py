from sqlalchemy import Column, Integer, String, Boolean, Date, Enum
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()

class Hour(Base):
    __tablename__ = 'Hours'

    key = Column(Integer, primary_key=True)
    hour_start = Column(Integer)
    hour_end = Column(Integer)
    date = Column(Date)
    day_of_week = Column(Enum(DayOfWeek))
    month = Column(Enum(Month))
    year = Column(Integer)
    weekend = Column(Boolean)
    holiday = Column(Boolean)
    holiday_name = Column(String)


class DayOfWeek(enum.Enum):
    SUNDAY = "Sunday"
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"

class Month(enum.Enum):
    JANUARY = "January"
    FEBRUARY = "February"
    MARCH = "March"
    APRIL = "April"
    MAY = "May"
    JUNE = "June"
    JULY = "July"
    AUGUST = "August"
    SEPTEMBER = "September"
    OCTOBER = "October"
    NOVEMBER = "November"
    DECEMBER = "December"

