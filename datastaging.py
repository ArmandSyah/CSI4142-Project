from datetime import datetime, timedelta
from models import Base, Hour, DayOfWeek, Month, Weather
from haversine import haversine
from collections import namedtuple
import math
import time
import calendar
import holidays
import pandas as pd
pd.set_option('display.max_rows', 500)

Weather_Averages = namedtuple("Weather_Averages", "high low")

# within 30 degress
acceptable_weather_ranges = {1: Weather_Averages(24.0, -44.0), 2: Weather_Averages(27.0, -43.0), 3: Weather_Averages(32.0, -37.0), 4: Weather_Averages(41.0, -29.0), 5: Weather_Averages(49.0, -22.0), 6: Weather_Averages(
    54.0, -17.0), 7: Weather_Averages(57.0, -14.0), 8: Weather_Averages(55.0, -16.0), 9: Weather_Averages(50.0, -20.0), 10: Weather_Averages(43.0, -26.0), 11: Weather_Averages(25.0, -32.0), 12: Weather_Averages(28.0, -39.0)}

chunksize = 10 ** 6
ottawa_lat_long = (45.41117, -75.69812)
weather_files = ['C:\Projects\\ontario_1_1.csv', 'C:\Projects\\ontario_1_2.csv', 'C:\Projects\\ontario_2_1.csv',
                 'C:\Projects\\ontario_2_2.csv', 'C:\Projects\\ontario_3.csv', 'C:\Projects\\ontario_4.csv']


def import_data(db):
    return


def clean_weather_data(db):
    li = []
    for batch in pd.read_csv("C:\Projects\\priority_weather.csv", chunksize=chunksize):
        batch = batch.loc[batch["temp_celcius"].notnull(), ["date_time", "year", "month", "day", "time", "temp_celcius", "dew_point_temp_celcius", "rel_hum", "wind_dir_10s_deg",
                                                            "wind_spd_km_h", "visibility_km", "stn_press_kpa", "humidex", "wind_chill", "weather", "station_name", "province"]]
        if (len(batch) > 0):
            li.append(batch)

    cleaned_weather_frame = pd.concat(li, axis=0, ignore_index=True)
    print(len(cleaned_weather_frame))
    cleaned_weather_frame["valid"] = cleaned_weather_frame.apply(lambda row:
                                                                 check_valid(row["temp_celcius"], row["month"], row["rel_hum"], row["wind_spd_km_h"], row["dew_point_temp_celcius"], row["wind_dir_10s_deg"], row["visibility_km"]), axis=1)
    cleaned_weather_frame = cleaned_weather_frame.loc[cleaned_weather_frame["valid"], ["date_time", "year", "month", "day", "time", "temp_celcius", "dew_point_temp_celcius", "rel_hum", "wind_dir_10s_deg",
                                                                                       "wind_spd_km_h", "visibility_km", "stn_press_kpa", "humidex", "wind_chill", "weather", "station_name", "province"]]
    cleaned_weather_frame["hour_id"] = cleaned_weather_frame.apply(lambda row: find_hour_id(
        row["year"], row["month"], row["day"], row["time"].split(":")[0], db), axis=1)
    cleaned_weather_frame = cleaned_weather_frame.loc[cleaned_weather_frame["hour_id"] != -1, ["date_time", "year", "month", "day", "time", "temp_celcius", "dew_point_temp_celcius", "rel_hum", "wind_dir_10s_deg",
                                                                                               "wind_spd_km_h", "visibility_km", "stn_press_kpa", "humidex", "wind_chill", "weather", "station_name", "province", "hour_id"]]
    cleaned_weather_frame.to_csv("c:\Projects\\cleaned.csv", sep=",")
    print(len(cleaned_weather_frame))


def check_valid(temp, month, humidity, wind_spd, dew_point_temp, wind_dir, visibility):
    valid_temp = temp >= acceptable_weather_ranges[month].low and temp <= acceptable_weather_ranges[month].high
    valid_humidity = math.isnan(humidity) or (
        humidity >= 0 and humidity <= 100)
    valid_wind_spd = math.isnan(wind_spd) or wind_spd >= 0
    valid_dew_point_temp = math.isnan(dew_point_temp) or (
        dew_point_temp >= acceptable_weather_ranges[month].low and dew_point_temp <= acceptable_weather_ranges[month].high)
    valid_wind_dir = math.isnan(wind_dir) or (
        wind_dir >= 0 and wind_dir <= 36)
    valid_visibility = math.isnan(visibility) or (visibility >= 0)
    return valid_temp and valid_humidity and valid_wind_spd and valid_dew_point_temp and valid_wind_dir and valid_visibility


def find_hour_id(year, month, day, hour, db):
    d = datetime(year, month, day,
                 int(hour))
    print(d)
    hour_obj = db.session.query(Hour).filter(Hour.hour_start == d.hour, Hour.month == Month(
        calendar.month_name[month]), Hour.year == year, Hour.date == d.date()).first()
    print(hour_obj)
    return hour_obj.key if hour_obj != None else -1


def find_all_ottawa_weather_stations(db):
    weather_station_inv = pd.read_csv(
        'C:\Projects\\Station Inventory EN.csv', skiprows=3, encoding="utf-8")
    ontario_weather_stations = weather_station_inv[weather_station_inv.Province == "ONTARIO"]
    ontario_weather_stations["distance_to_ottawa"] = -1
    cols = ontario_weather_stations.columns
    cols = cols.map(lambda x: x.replace(' ', '_'))
    ontario_weather_stations.columns = cols
    for index, row in ontario_weather_stations.iterrows():
        weather_station_lat_long = (
            abs(row['Latitude_(Decimal_Degrees)']), -row['Longitude_(Decimal_Degrees)'] if row['Longitude_(Decimal_Degrees)'] > 0 else row['Longitude_(Decimal_Degrees)'])
        ontario_weather_stations.at[index, 'Latitude_(Decimal_Degrees)'] = abs(
            row['Latitude_(Decimal_Degrees)'])
        ontario_weather_stations.at[index, 'Longitude_(Decimal_Degrees)'] = - \
            row['Longitude_(Decimal_Degrees)'] if row['Longitude_(Decimal_Degrees)'] > 0 else row['Longitude_(Decimal_Degrees)']
        ottawa_distance = haversine(ottawa_lat_long, weather_station_lat_long)
        ontario_weather_stations.at[index,
                                    "distance_to_ottawa"] = ottawa_distance

    ontario_weather_stations = ontario_weather_stations[
        ontario_weather_stations.distance_to_ottawa <= 100]  # Get all stations within a 100 km radius of
    ontario_weather_stations.to_csv(
        "C:\Projects\\ontario_weather_stations_8253748.csv", sep=',', columns=["Name", "Province", "Latitude_(Decimal_Degrees)", "Longitude_(Decimal_Degrees)", "Elevation_(m)", "distance_to_ottawa"])


def retrieve_priority_weather(db):
    find_all_ottawa_weather_stations(db)
    ottawa_weather_stations = pd.read_csv(
        "C:\Projects\\ontario_weather_stations_8253748.csv")
    station_names = ottawa_weather_stations["Name"].tolist()
    li = []
    batch = 1

    for weather_file in weather_files:
        print(f'Now reading {weather_file}')
        for weather_set_chunk in pd.read_csv(weather_file, chunksize=chunksize,
                                             usecols=['X.Date.Time', 'Year', 'Month', 'Day', 'Time', 'Temp...C.', 'Dew.Point.Temp...C.', 'Rel.Hum....', 'Wind.Dir..10s.deg.', 'Wind.Spd..km.h.', 'Visibility..km.', 'Stn.Press..kPa.', 'Hmdx', 'Wind.Chill', 'Weather.', 'X.U.FEFF..Station.Name.', 'X.Province.']):
            weather_set_chunk.columns = ["date_time", "year", "month", "day", "time", "temp_celcius", "dew_point_temp_celcius", "rel_hum", "wind_dir_10s_deg",
                                         "wind_spd_km_h", "visibility_km", "stn_press_kpa", "humidex", "wind_chill", "weather", "station_name", "province"]
            important_chunk = weather_set_chunk[
                weather_set_chunk.station_name.isin(station_names)]
            if (len(important_chunk) > 0):
                print(f"Writing Batch {batch}")
                li.append(important_chunk)
                batch += 1

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.to_csv("C:\Projects\\priority_weather.csv", sep=",")


def set_up_hour_table_csv(db):
    df_prio = pd.read_csv('C:\Projects\\priority_weather.csv')
    drop = df_prio.drop_duplicates(subset=['year', 'month', 'day', 'time'])
    drop.to_csv("C:\Projects\\derp.csv")


def set_up_hours_on_db(db):
    used_dates = set()
    hours = []
    canada_holidays = holidays.Canada()
    hour_df = pd.read_csv("C:\Projects\\derp.csv")
    print("Starting Hour Insert")
    for index, row in hour_df.iterrows():
        hour, minute = row['time'].split(':')

        d = datetime(row['year'], row['month'], row['day'],
                     int(hour), int(minute))
        if d in used_dates:
            continue
        else:
            used_dates.add(d)
            is_holiday = d in canada_holidays
            hours.append(
                {'hour_start': d.hour,
                    'hour_end': 0 if d.hour + 1 == 24 else d.hour + 1,
                    'date': d.date(),
                    'day_of_week': DayOfWeek(calendar.day_name[d.weekday()]),
                    'month': Month(calendar.month_name[d.month]),
                    'year': d.year,
                    'weekend': d.weekday() >= 5,
                    'holiday': is_holiday,
                    'holiday_name': canada_holidays.get(d) if is_holiday else None})

    db.engine.execute(Hour.__table__.insert(), hours)
    print("Finished inserting hours")


def bulk_insert_weather_into_table(db):
    weather_data = pd.read_csv("C:\projects\\cleaned.csv")
    stations_frame = pd.read_csv(
        "C:\Projects\\ontario_weather_stations_8253748.csv")
    weather_entries = []
    print("Beggining process of entering to weather dimension table")
    for index, row in weather_data.iterrows():
        station_lat_long = stations_frame.loc[stations_frame['Name'] == row['station_name'], [
            "Latitude_(Decimal_Degrees)", "Longitude_(Decimal_Degrees)"]].iloc[0]
        weather_entry = {
            'station_name': row['station_name'],
            'longitude': station_lat_long['Longitude_(Decimal_Degrees)'],
            'latitude': station_lat_long['Latitude_(Decimal_Degrees)'],
            'temperature': row['temp_celcius'],
            'visibility': row['visibility_km'] if not math.isnan(row['visibility_km']) else None,
            'wind_speed': row['wind_spd_km_h'] if not math.isnan(row['wind_spd_km_h']) else None,
            'wind_chill': row['wind_chill'] if not math.isnan(row['wind_chill']) else None,
            'wind_direction': row['wind_dir_10s_deg'] if not math.isnan(row['wind_dir_10s_deg']) else None,
            'pressure': row['stn_press_kpa'] if not math.isnan(row['stn_press_kpa']) else None
        }
        weather_entries.append(weather_entry)
    db.engine.execute(Weather.__table__.insert(), weather_entries)
    print("Finished processing data into weather dimension")


def hour_staging_first_attempt(db):
    collision_set = pd.read_csv(
        'C:\Projects\\2014collisionsfinal.xls.csv')

    collision_set["adjusted_hour"] = 0
    collision_set["hour_id"] = -1

    canada_holidays = holidays.Canada()

    adjust_hours_on_collision_set(collision_set)
    print(collision_set.head(20))

    used_dates = set()
    hours = []
    i = 0
    for weather_set_chunk in pd.read_csv('C:\Projects\\ontario_1_1.csv', chunksize=chunksize, usecols=[
            'Year', 'Month', 'Day', 'Time'], dtype={'Year': 'int32', 'Month': 'int8', 'Day': 'int8', 'Time': 'str'}):
        for index, row in weather_set_chunk.iterrows():
            print(f"Row {i}")
            i += 1
            hour, minute = row['Time'].split(':')

            d = datetime(row['Year'], row['Month'], row['Day'],
                         int(hour), int(minute))
            if d in used_dates:
                continue
            else:
                used_dates.add(d)
                is_holiday = d in canada_holidays
                hours.append(
                    {'hour_start': d.hour,
                        'hour_end': 0 if d.hour + 1 == 24 else d.hour + 1,
                        'date': d.date(),
                        'day_of_week': DayOfWeek(calendar.day_name[d.weekday()]),
                        'month': Month(calendar.month_name[d.month]),
                        'year': d.year,
                        'weekend': d.weekday() >= 5,
                        'holiday': is_holiday,
                        'holiday_name': canada_holidays.get(d) if is_holiday else None})

    db.engine.execute(Hour.__table__.insert(), hours)


def adjust_hours_on_collision_set(collision_set):
    for index, row in collision_set.iterrows():
        t = row['TIME']
        t = datetime.strptime(t, '%I:%M:%S %p')
        # t = time.strptime(t[:-3], "%H:%M")
        collision_set.at[index, 'adjusted_hour'] = (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
                                                    + timedelta(hours=t.minute//30)).hour
