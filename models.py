from sqlalchemy import Column, Integer, String, Boolean, Date, Enum, Numeric, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()

# Enums


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


class EnvironmentType(enum.Enum):
    UNKNOWN = "00 - Unknown"
    CLEAR = "01 - Clear"
    RAIN = "02 - Rain"
    SNOW = "03 - Snow"
    FREEZING_RAIN = "04 - Freezing rain"
    DRIFTING_SNOW = "05 - Drifter snow"
    STRONG_WIND = "06 - Strong wind"
    FOG_MIST_SMOKE_DUST = "07 - Fog, mist, smoke, dust"


class RoadSurfaceType(enum.Enum):
    DRY = "01 - Dry"
    WET = "02- Wet"
    LOOSE_SNOW = "03 - Loose snow"
    SLUSH = "04 - Slush"
    PACKED_SNOW = "05 - Packed snow"
    ICE = "06 - Ice"


class TrafficControlType(enum.Enum):
    TRAFFIC_SIGNAL = "01 - Traffic signal"
    STOP_SIGN = "02 - Stop sign"
    NO_CONTROL = "10 - No control"


# Hour Dimension

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


# Accident Dimension

class Accident(Base):
    __tablename__ = "Accidents"

    key = Column(Integer, primary_key=True)
    accident_datetime = Column(Date)
    environemnt = Column(Enum(EnvironmentType))
    road_surface = Column(Enum(RoadSurfaceType))
    traffic_control = Column(Enum(TrafficControlType))


# Event Dimension

class Event(Base):
    __tablename__ = "Events"

    key = Column(Integer, primary_key=True)
    event_name = Column(String)
    event_start_date = Column(Date)
    event_end_date = Column(Date)


# Location Dimension

class Location(Base):
    __tablename__ = "Locations"

    key = Column(Integer, primary_key=True)
    street_name_highway = Column(String)
    intersection_1 = Column(String)
    intersection_2 = Column(String)
    longitude = Column(Numeric)
    latitude = Column(Numeric)
    neighbourhood = Column(String)


# Weather Dimension

class Weather(Base):
    __tablename__ = "Weather"

    key = Column(Integer, primary_key=True)
    station_name = Column(String)
    longitude = Column(Numeric)
    latitude = Column(Numeric)
    temperature = Column(Numeric)
    visibility = Column(Numeric)
    wind_speed = Column(Numeric)
    wind_chill = Column(Numeric)
    wind_direction = Column(Integer)
    pressure = Column(Numeric)


# Accident Fact Table

class AccidentFact(Base):
    __tablename__ = "Accident Facts"

    hour_key = Column(Integer, ForeignKey('Hours.key'), primary_key=True)
    location_key = Column(Integer, ForeignKey(
        'Locations.key'), primary_key=True)
    accident_key = Column(Integer, ForeignKey(
        'Accidents.key'), primary_key=True)
    weather_key = Column(Integer, ForeignKey('Weather.key'), primary_key=True)
    event_key = Column(Integer, ForeignKey('Events.key'), primary_key=True)
    is_fatal = Column(Boolean)
    is_intersection = Column(Boolean)
