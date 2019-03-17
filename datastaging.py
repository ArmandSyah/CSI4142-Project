from datetime import datetime, timedelta
from models import Base, Hour, DayOfWeek, Month, Weather, Accident, Location, AccidentFact
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from haversine import haversine
from collections import namedtuple
import math
import time
import calendar
import holidays
import pandas as pd
import numpy as np
import re
import os
dir = os.path.dirname(__file__)
input_path = os.path.join(dir, 'input')
output_path = os.path.join(dir, "output")
pd.set_option('display.max_rows', 500)

Weather_Averages = namedtuple("Weather_Averages", "high low")
geolocator = Nominatim(
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36")
geocoderatelimit = RateLimiter(geolocator.reverse, min_delay_seconds=1)

# within 30 degress
acceptable_weather_ranges = {1: Weather_Averages(24.0, -44.0), 2: Weather_Averages(27.0, -43.0), 3: Weather_Averages(32.0, -37.0), 4: Weather_Averages(41.0, -29.0), 5: Weather_Averages(49.0, -22.0), 6: Weather_Averages(
    54.0, -17.0), 7: Weather_Averages(57.0, -14.0), 8: Weather_Averages(55.0, -16.0), 9: Weather_Averages(50.0, -20.0), 10: Weather_Averages(43.0, -26.0), 11: Weather_Averages(25.0, -32.0), 12: Weather_Averages(28.0, -39.0)}

chunksize = 10 ** 6
ottawa_lat_long = (45.41117, -75.69812)
weather_station_file = os.path.join(input_path, "Station Inventory EN.csv")
weather_files = [os.path.join(input_path, "ontario_1_1.csv"), os.path.join(input_path, "ontario_1_2.csv"), os.path.join(input_path, "ontario_2_1.csv"),
                 os.path.join(input_path, "ontario_2_2.csv"), os.path.join(input_path, "ontario_3.csv"), os.path.join(input_path, "ontario_4.csv")]
ottawa_collision_files = [os.path.join(input_path, '2013collisionsfinal.xls.csv'), os.path.join(input_path, '2014collisionsfinal.xls.csv'),
                          os.path.join(input_path, '2015collisionsfinal.xls.csv'), os.path.join(input_path, '2016collisionsfinal.xls.csv'), os.path.join(input_path, '2017collisionsfinal.xls.csv')]
pattern_1 = re.compile(r'.* btwn .* & .*')
pattern_2 = re.compile(r'.* @ .*')
pattern_3 = re.compile(r'.*/.*/.*')

pattern_4 = re.compile(r'.* @ .*/.*')
pattern_5 = re.compile(r'.* btwn .*')


def import_data(db):
    find_all_ottawa_weather_stations(db)
    retrieve_priority_weather(db)
    set_up_hour_table_csv(db)
    set_up_hours_on_db(db)
    clean_weather_data(db)
    bulk_insert_weather_into_table(db)
    concat_traffic_data(db)
    collision_set_cleaning(db)
    set_up_collisions_frames(db)
    insert_locations_accidents_table(db)
    final_step_fact_table(db)


def import_weather(db):
    find_all_ottawa_weather_stations(db)
    retrieve_priority_weather(db)
    set_up_hour_table_csv(db)
    set_up_hours_on_db(db)
    clean_weather_data(db)
    bulk_insert_weather_into_table(db)


def import_collision(db):
    concat_traffic_data(db)
    collision_set_cleaning(db)
    set_up_collisions_frames(db)
    insert_locations_accidents_table(db)


def setup_facttable(db):
    final_step_fact_table(db)


def find_all_ottawa_weather_stations(db):
    print("Finding most important weather stations")
    weather_station_inv = pd.read_csv(
        weather_station_file, skiprows=3, encoding="utf-8")
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
        os.path.join(output_path, "priority_weather_stations.csv"), sep=',', columns=["Name", "Province", "Latitude_(Decimal_Degrees)", "Longitude_(Decimal_Degrees)", "Elevation_(m)", "distance_to_ottawa"])
    print("Finished finding most important weather stations")


def retrieve_priority_weather(db):
    print("Retrieving weather from priority stations")
    ottawa_weather_stations = pd.read_csv(
        os.path.join(output_path, "priority_weather_stations.csv"))
    station_names = ottawa_weather_stations["Name"].tolist()
    li = []

    for weather_file in weather_files:
        print(f'Now reading {weather_file}')
        for weather_set_chunk in pd.read_csv(weather_file, chunksize=chunksize,
                                             usecols=['X.Date.Time', 'Year', 'Month', 'Day', 'Time', 'Temp...C.', 'Dew.Point.Temp...C.', 'Rel.Hum....', 'Wind.Dir..10s.deg.', 'Wind.Spd..km.h.', 'Visibility..km.', 'Stn.Press..kPa.', 'Hmdx', 'Wind.Chill', 'Weather.', 'X.U.FEFF..Station.Name.', 'X.Province.']):
            weather_set_chunk.columns = ["date_time", "year", "month", "day", "time", "temp_celcius", "dew_point_temp_celcius", "rel_hum", "wind_dir_10s_deg",
                                         "wind_spd_km_h", "visibility_km", "stn_press_kpa", "humidex", "wind_chill", "weather", "station_name", "province"]
            important_chunk = weather_set_chunk[
                weather_set_chunk.station_name.isin(station_names)]
            if (len(important_chunk) > 0):
                li.append(important_chunk)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.to_csv(os.path.join(output_path, "priority_weather.csv"), sep=",")
    print("Finished Retrieving weather from priority stations")


def set_up_hour_table_csv(db):
    df_prio = pd.read_csv(os.path.join(output_path, "priority_weather.csv"))
    drop = df_prio.drop_duplicates(subset=['year', 'month', 'day', 'time'])
    drop.to_csv(os.path.join(output_path, "hours_dim.csv"))


def set_up_hours_on_db(db):
    used_dates = set()
    hours = []
    canada_holidays = holidays.Canada()
    hour_df = pd.read_csv(os.path.join(output_path, "hours_dim.csv"))
    print("Starting Hour Insert")
    for _, row in hour_df.iterrows():
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


def clean_weather_data(db):
    li = []
    batch_no = 1
    print("Cleaning Weather data")
    for batch in pd.read_csv(os.path.join(output_path, "priority_weather.csv"), chunksize=chunksize):
        print(f"Beginning processing weather batch {batch_no}")
        batch = batch.loc[batch["temp_celcius"].notnull(), ["date_time", "year", "month", "day", "time", "temp_celcius", "dew_point_temp_celcius", "rel_hum", "wind_dir_10s_deg",
                                                            "wind_spd_km_h", "visibility_km", "stn_press_kpa", "humidex", "wind_chill", "weather", "station_name", "province"]]
        batch["valid"] = batch.apply(lambda row:
                                     check_valid(row["temp_celcius"], row["month"], row["rel_hum"], row["wind_spd_km_h"], row["dew_point_temp_celcius"], row["wind_dir_10s_deg"], row["visibility_km"]), axis=1)
        batch = batch.loc[batch["valid"], ["date_time", "year", "month", "day", "time", "temp_celcius", "dew_point_temp_celcius", "rel_hum", "wind_dir_10s_deg",
                                           "wind_spd_km_h", "visibility_km", "stn_press_kpa", "humidex", "wind_chill", "weather", "station_name", "province"]]
        batch["hour_id"] = batch.apply(lambda row: find_hour_id_weather(
            row["year"], row["month"], row["day"], row["time"].split(":")[0], db), axis=1)
        batch = batch.loc[batch["hour_id"] != -1, ["date_time", "year", "month", "day", "time", "temp_celcius", "dew_point_temp_celcius", "rel_hum", "wind_dir_10s_deg",
                                                   "wind_spd_km_h", "visibility_km", "stn_press_kpa", "humidex", "wind_chill", "weather", "station_name", "province", "hour_id"]]
        if (len(batch) > 0):
            li.append(batch)
        print(f"Batch {batch_no} Processing complete")
        batch_no += 1

    cleaned_weather_frame = pd.concat(li, axis=0, ignore_index=True)
    cleaned_weather_frame.insert(
        0, "key", range(0, len(cleaned_weather_frame)))
    cleaned_weather_frame.to_csv(os.path.join(
        output_path, "cleaned_priority_weather.csv"), sep=",")
    print("Finished Cleaning Weather data")


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


def find_hour_id_weather(year, month, day, hour, db):
    d = datetime(year, month, day,
                 int(hour))
    hour_obj = db.session.query(Hour).filter(Hour.hour_start == d.hour, Hour.month == Month(
        calendar.month_name[month]), Hour.year == year, Hour.date == d.date()).first()
    return hour_obj.key if hour_obj != None else -1


def bulk_insert_weather_into_table(db):
    weather_data = pd.read_csv(os.path.join(
        output_path, "cleaned_priority_weather.csv"))
    stations_frame = pd.read_csv(
        os.path.join(output_path, "priority_weather_stations.csv"))
    weather_entries = []
    print("Beggining process of entering to weather dimension table")
    for _, row in weather_data.iterrows():
        station_lat_long = stations_frame.loc[stations_frame['Name'] == row['station_name'], [
            "Latitude_(Decimal_Degrees)", "Longitude_(Decimal_Degrees)"]].iloc[0]
        weather_entry = {
            'key': row['key'],
            'station_name': row['station_name'],
            'longitude': station_lat_long['Longitude_(Decimal_Degrees)'],
            'latitude': station_lat_long['Latitude_(Decimal_Degrees)'],
            'temperature': row['temp_celcius'],
            'visibility': row['visibility_km'] if not math.isnan(row['visibility_km']) else None,
            'wind_speed': row['wind_spd_km_h'] if not math.isnan(row['wind_spd_km_h']) else None,
            'wind_chill': row['wind_chill'] if not math.isnan(row['wind_chill']) else None,
            'wind_direction': row['wind_dir_10s_deg'] if not math.isnan(row['wind_dir_10s_deg']) else None,
            'pressure': row['stn_press_kpa'] if not math.isnan(row['stn_press_kpa']) else None,
            'relative_humidity': row['rel_hum'] if not math.isnan(row['rel_hum']) else None,
            'humidex': row['humidex'] if not math.isnan(row['humidex']) else None,
            'hour_id': row['hour_id'] if not math.isnan(row['hour_id']) else None
        }
        weather_entries.append(weather_entry)
    db.engine.execute(Weather.__table__.insert(), weather_entries)
    print("Finished processing data into weather dimension")


def concat_traffic_data(db):
    li = []
    print("Combining ottawa collision files")
    for ottawa_collision_file in ottawa_collision_files:
        print(f'Now reading {ottawa_collision_file}')
        ottawa_collision_frame = pd.read_csv(ottawa_collision_file, usecols=[
                                             'LOCATION', 'LONGITUDE', 'LATITUDE', 'DATE', 'TIME', 'ENVIRONMENT', 'LIGHT', 'SURFACE_CONDITION', 'TRAFFIC_CONTROL', 'COLLISION_CLASSIFICATION', 'IMPACT_TYPE'])
        if ottawa_collision_file == os.path.join(input_path, '2017collisionsfinal.xls.csv'):
            ottawa_collision_frame["DATE"] = ottawa_collision_frame.apply(
                lambda row: datetime.strptime(row["DATE"], '%m/%d/%Y').strftime('%Y-%m-%d'), axis=1)
        li.append(ottawa_collision_frame)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.to_csv(os.path.join(output_path, "ottawa_collision_2013_2017.csv"))
    print("Finish Combining ottawa collision files")


def collision_set_cleaning(db):
    print("Beggining cleaning of collision data")
    collision_set = pd.read_csv(os.path.join(
        output_path, "ottawa_collision_2013_2017.csv"))
    collision_set["adjusted_hour"] = 0
    collision_set["hour_id"] = -1

    for index, row in collision_set.iterrows():
        t = row['TIME']
        t = datetime.strptime(t, '%I:%M:%S %p')
        collision_set.at[index, 'adjusted_hour'] = (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
                                                    + timedelta(hours=t.minute//30)).hour

    collision_set["hour_id"] = collision_set.apply(
        lambda row: find_hour_id_collision(row['DATE'], row['adjusted_hour'], db), axis=1)
    collision_set = collision_set.loc[collision_set["hour_id"] != -1, ['LOCATION', 'LONGITUDE', 'LATITUDE', 'DATE', 'TIME', 'ENVIRONMENT',
                                                                       'LIGHT', 'SURFACE_CONDITION', 'TRAFFIC_CONTROL', 'COLLISION_CLASSIFICATION', 'IMPACT_TYPE', 'adjusted_hour', 'hour_id']]

    clean_columns = ["ENVIRONMENT", "LIGHT", "SURFACE_CONDITION",
                     "TRAFFIC_CONTROL", "COLLISION_CLASSIFICATION", "IMPACT_TYPE"]
    for attr in clean_columns:
        print(f"Cleaning {attr}")
        collision_set[attr] = collision_set.apply(lambda row: "" if row[attr] != row[attr] else (row[attr].split(
            "-")[1].strip() if len(row[attr].split("-")) == 2 else " ".join(row[attr].split("-")[1:]).strip()), axis=1)
    collision_set.to_csv(os.path.join(
        output_path, "ottawa_collision_2013_2017.csv"))
    print("Finished cleaning traffic data")


def find_hour_id_collision(date, hour, db):
    d = datetime.strptime(date, '%Y-%m-%d')
    hour_obj = db.session.query(Hour).filter(Hour.hour_start == hour, Hour.month == Month(
        calendar.month_name[d.month]), Hour.year == d.year, Hour.date == d.date()).first()
    return hour_obj.key if hour_obj != None else -1


def set_up_collisions_frames(db):
    df = pd.read_csv(os.path.join(
        output_path, "ottawa_collision_2013_2017.csv"))

    print(f"Accidents {len(df)}")

    neighbourhoods = pd.read_csv(os.path.join(input_path, "neighbourhood.csv"))
    location_frame = pd.DataFrame(
        columns=["key", "street_name_highway", "intersection_1", "intersection_2", "longitude", "latitude", "neighbourhood", "is_intersection", "original_location"])
    accident_frame = pd.DataFrame(columns=["key", "accident_time", "environment", "road_surface",
                                           "traffic_control", "visibility", "impact_type", "is_fatal", "hour_id", "loc_id"])
    accident_key, location_key = 0, 0
    used_location = []

    print("Beggining setting up dataframes for Location and Accident")
    for _, row in df.iterrows():
        # handle location
        location, latitude, longitude = row["LOCATION"], row["LATITUDE"], row["LONGITUDE"]
        if location in used_location:
            location_row = location_frame.loc[location_frame["original_location"]
                                              == location]
            location_row = location_row.iloc[0]
            accident_frame = accident_frame.append({'key': accident_key, 'accident_time': row['TIME'], 'environment': row['ENVIRONMENT'], 'visibility': row[
                'LIGHT'], 'impact_type': row['IMPACT_TYPE'], 'is_fatal': True if row['COLLISION_CLASSIFICATION'] == 'Fatal Injury' else False,
                "is_intersection": location_row["is_intersection"], "hour_id": row["hour_id"], 'loc_id': location_row["key"]}, ignore_index=True)
            accident_key += 1
            continue

        distances_between_neighbourhoods = []

        for _, neighbour_row in neighbourhoods.iterrows():
            distances_between_neighbourhoods.append((neighbour_row["name"],
                                                     haversine((latitude, longitude), (neighbour_row["latitude"], neighbour_row["longitude"]))))

        if len(distances_between_neighbourhoods) == 0:
            continue

        distances_between_neighbourhoods.sort(key=lambda tup: tup[1])
        closest_neighbourhood = distances_between_neighbourhoods[0]

        if re.match(pattern_1, location):
            location_parts = location.split("btwn")
            street_name, intersections = location_parts[0], location_parts[1].split(
                "&")
            location_dict = {"key": location_key, 'street_name_highway': street_name.strip(), 'intersection_1': intersections[0].strip(), 'intersection_2': intersections[
                1].strip(), 'longitude': row["LONGITUDE"], 'latitude': row["LATITUDE"], 'neighbourhood': closest_neighbourhood[0], "is_intersection": True, 'original_location': location}
            location_frame = location_frame.append(
                location_dict, ignore_index=True)
            accident_frame = accident_frame.append({'key': accident_key, 'accident_time': row['TIME'], 'environment': row['ENVIRONMENT'], 'visibility': row[
                'LIGHT'], 'impact_type': row['IMPACT_TYPE'], 'is_fatal': True if row['COLLISION_CLASSIFICATION'] == 'Fatal Injury' else False,
                "hour_id": row["hour_id"], 'loc_id': location_key}, ignore_index=True)
        elif re.match(pattern_2, location):
            location_parts = location.split("@")
            street_name, intersection = location_parts[0], location_parts[1]
            location_dict = {"key": location_key, 'street_name_highway': street_name.strip(), 'intersection_1': intersection.strip(), 'intersection_2': None,
                             'longitude': row["LONGITUDE"], 'latitude': row["LATITUDE"], 'neighbourhood': closest_neighbourhood[0], "is_intersection": False,  'original_location': location}
            location_frame = location_frame.append(
                location_dict, ignore_index=True)
            accident_frame = accident_frame.append({'key': accident_key, 'accident_time': row['TIME'], 'environment': row['ENVIRONMENT'], 'visibility': row[
                'LIGHT'], 'impact_type': row['IMPACT_TYPE'], 'is_fatal': True if row['COLLISION_CLASSIFICATION'] == 'Fatal Injury' else False,
                "hour_id": row["hour_id"], 'loc_id': location_key}, ignore_index=True)
        elif re.match(pattern_3, location):
            location_parts = location.split("/")
            street_name, intersection_1, intersection_2 = location_parts[
                0], location_parts[1], location_parts[2]
            location_dict = {"key": location_key, 'street_name_highway': street_name.strip(), 'intersection_1': intersection_1.strip(), 'intersection_2': intersection_2.strip(),
                             'longitude': row["LONGITUDE"], 'latitude': row["LATITUDE"], 'neighbourhood': closest_neighbourhood[0], "is_intersection": True, 'original_location': location}
            location_frame = location_frame.append(
                location_dict, ignore_index=True)
            accident_frame = accident_frame.append({'key': accident_key, 'accident_time': row['TIME'], 'environment': row['ENVIRONMENT'], 'visibility': row[
                'LIGHT'], 'impact_type': row['IMPACT_TYPE'], 'is_fatal': True if row['COLLISION_CLASSIFICATION'] == 'Fatal Injury' else False,
                "hour_id": row["hour_id"], 'loc_id': location_key}, ignore_index=True)
        elif re.match(pattern_4, location):
            location_parts = location.split("@")
            street_name, intersections = location_parts[0], location_parts[1].split(
                "/")
            location_dict = {"key": location_key, 'street_name_highway': street_name.strip(), 'intersection_1': intersections[0].strip(), 'intersection_2': intersections[
                1].strip(), 'longitude': row["LONGITUDE"], 'latitude': row["LATITUDE"], 'neighbourhood': closest_neighbourhood[0], "is_intersection": True, 'original_location': location}
            location_frame = location_frame.append(
                location_dict, ignore_index=True)
            accident_frame = accident_frame.append({'key': accident_key, 'accident_time': row['TIME'], 'environment': row['ENVIRONMENT'], 'visibility': row[
                'LIGHT'], 'impact_type': row['IMPACT_TYPE'], 'is_fatal': True if row['COLLISION_CLASSIFICATION'] == 'Fatal Injury' else False,
                "hour_id": row["hour_id"], 'loc_id': location_key}, ignore_index=True)
        elif re.match(pattern_5, location):
            location_parts = location.split("btwn")
            street_name, intersection = location_parts[0], location_parts[1]
            location_dict = {"key": location_key, 'street_name_highway': street_name.strip(), 'intersection_1': intersection.strip(), 'intersection_2': None,
                             'longitude': row["LONGITUDE"], 'latitude': row["LATITUDE"], 'neighbourhood': closest_neighbourhood[0], "is_intersection": False, 'original_location': location}
            location_frame = location_frame.append(
                location_dict, ignore_index=True)
            accident_frame = accident_frame.append({'key': accident_key, 'accident_time': row['TIME'], 'environment': row['ENVIRONMENT'], 'visibility': row[
                'LIGHT'], 'impact_type': row['IMPACT_TYPE'], 'is_fatal': True if row['COLLISION_CLASSIFICATION'] == 'Fatal Injury' else False,
                "hour_id": row["hour_id"], 'loc_id': location_key}, ignore_index=True)

        used_location.append(location)

        accident_key += 1
        location_key += 1

    location_frame.to_csv(os.path.join(
        output_path, "locationdim.csv"))
    accident_frame.to_csv(os.path.join(
        output_path, "accidentdim.csv"))
    print("Finished setting up dataframes for Location and Accident")


def insert_locations_accidents_table(db):
    print("Beggining inserting into location and accident dimension tables")
    location_frame = pd.read_csv(os.path.join(
        output_path, "locationdim.csv"))
    accident_frame = pd.read_csv(os.path.join(
        output_path, "accidentdim.csv"))

    locations = []
    accidents = []

    for _, row in location_frame.iterrows():
        location_entry = {
            "key": row["key"],
            "street_name_highway": row["street_name_highway"],
            "intersection_1": row["intersection_1"],
            "intersection_2": row["intersection_2"],
            "longitude": row["longitude"],
            "latitude": row["latitude"],
            "neighbourhood": row["neighbourhood"]
        }
        locations.append(location_entry)

    for _, row in accident_frame.iterrows():
        accident_entry = {
            "key": row["key"],
            "accident_time": row["accident_time"],
            "environment": row["environment"],
            "road_surface": row["road_surface"],
            "traffic_control": row["traffic_control"],
            "visibility": row["visibility"],
            "impact_type": row["impact_type"]
        }
        accidents.append(accident_entry)

    db.engine.execute(Location.__table__.insert(), locations)
    db.engine.execute(Accident.__table__.insert(), accidents)
    print("Finished inserting into location and accident dimension tables")


def final_step_fact_table(db):
    accident_facts = []
    stations_frame = pd.read_csv(os.path.join(
        output_path, "priority_weather_stations.csv"))
    weather_frame = pd.read_csv(os.path.join(
        output_path, "cleaned_priority_weather.csv"))
    accidents_frame = pd.read_csv(os.path.join(output_path, "accidentdim.csv"))
    locations_frame = pd.read_csv(os.path.join(output_path, "locationdim.csv"))

    print("Beggining Fact Table insertion")
    for _, row in accidents_frame.iterrows():
        key, hour_id, loc_id = row["key"], row["hour_id"], row["loc_id"]
        weather_with_matching_hour = weather_frame.loc[weather_frame['hour_id']
                                                       == hour_id]
        location_row = locations_frame.loc[locations_frame['key']
                                           == loc_id].iloc[0]
        location_latitude, location_longitude = location_row["latitude"], location_row["longitude"]
        station_names = weather_with_matching_hour["station_name"]
        collected_distances = []
        for station_name in station_names:
            station_row = stations_frame.loc[stations_frame["Name"]
                                             == station_name].iloc[0]
            station_latitude, station_longitude = station_row[
                "Latitude_(Decimal_Degrees)"], station_row["Longitude_(Decimal_Degrees)"]
            collected_distances.append((station_name, haversine(
                (location_latitude, location_longitude), (station_latitude, station_longitude))))
        collected_distances.sort(key=lambda tup: tup[1])
        shortest_weather_station_distance = collected_distances[0]
        matching_weather_hour_row = weather_frame.loc[(weather_frame["hour_id"] ==
                                                       hour_id) & (weather_frame["station_name"] == shortest_weather_station_distance[0])].iloc[0]
        accident_facts.append({"hour_key": int(hour_id), "location_key": int(loc_id), "accident_key": int(key),
                               "weather_key": int(matching_weather_hour_row["key"]), "is_fatal": row["is_fatal"], "is_intersection": location_row["is_intersection"]})

    db.engine.execute(AccidentFact.__table__.insert(), accident_facts)
    print("Finished Fact Table insertion")
