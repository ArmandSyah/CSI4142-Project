from sqlalchemy import Column, Integer, String, Boolean, Date, Enum, Numeric
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()

class Accident(Base):
    __tablename__ = "Accidents"

    key = Column(Integer, primary_key=True)
    accident_datetime = Column(Date)
    environemnt = Column(Enum(EnvironmentType))
    road_surface = Column(Enum(RoadSurfaceType))
    traffic_control = Column(Enum(TrafficControlType))

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

