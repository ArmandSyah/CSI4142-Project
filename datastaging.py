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
import dateutil.parser
import pytz
dir = os.path.dirname(__file__)
input_path = os.path.join(dir, 'input')
output_path = os.path.join(dir, "output")
pd.set_option('display.max_rows', 500)

Weather_Averages = namedtuple("Weather_Averages", "high low")

# within 30 degress
acceptable_weather_ranges = {1: Weather_Averages(24.0, -44.0), 2: Weather_Averages(27.0, -43.0), 3: Weather_Averages(32.0, -37.0), 4: Weather_Averages(41.0, -29.0), 5: Weather_Averages(49.0, -22.0), 6: Weather_Averages(
    54.0, -17.0), 7: Weather_Averages(57.0, -14.0), 8: Weather_Averages(55.0, -16.0), 9: Weather_Averages(50.0, -20.0), 10: Weather_Averages(43.0, -26.0), 11: Weather_Averages(25.0, -32.0), 12: Weather_Averages(28.0, -39.0)}

chunksize = 10 ** 5
ottawa_lat_long = (45.41117, -75.69812)
toronto_lat_long = (43.5232, -79.3832)
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


def chunk(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


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


def import_hours(db):
    set_up_hour_table_csv(db)
    set_up_hours_on_db(db)


def import_weather(db):
    find_all_ottawa_weather_stations(db)
    retrieve_priority_weather(db)
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
        toronto_distance = haversine(
            toronto_lat_long, weather_station_lat_long)
        ontario_weather_stations.at[index,
                                    "distance_to_ottawa"] = ottawa_distance
        ontario_weather_stations.at[index,
                                    "distance_to_toronto"] = toronto_distance

    ontario_weather_stations = ontario_weather_stations[
        (ontario_weather_stations.distance_to_ottawa <= 75) | (ontario_weather_stations.distance_to_toronto <= 75)]  # Get all stations within a 75 km radius of ottawa or toronto
    ontario_weather_stations.to_csv(
        os.path.join(output_path, "priority_weather_stations.csv"), sep=',', columns=["Name", "Province", "Latitude_(Decimal_Degrees)", "Longitude_(Decimal_Degrees)", "Elevation_(m)", "distance_to_ottawa", "distance_to_toronto"])
    print("Finished finding most important weather stations")


def retrieve_priority_weather(db):
    print("Retrieving weather from priority stations")
    priority_weather_stations_frame = pd.read_csv(
        os.path.join(output_path, "priority_weather_stations.csv"))
    station_names = priority_weather_stations_frame["Name"].tolist()
    li = []

    for weather_file in weather_files:
        print(f'Now reading {weather_file}')
        for weather_set_chunk in pd.read_csv(weather_file, chunksize=chunksize,
                                             usecols=['X.Date.Time', 'Year', 'Month', 'Day', 'Time', 'Temp...C.', 'Dew.Point.Temp...C.', 'Rel.Hum....', 'Wind.Dir..10s.deg.', 'Wind.Spd..km.h.', 'Visibility..km.', 'Stn.Press..kPa.', 'Hmdx', 'Wind.Chill', 'Weather.', 'X.U.FEFF..Station.Name.', 'X.Province.']):
            weather_set_chunk.columns = ["date_time", "year", "month", "day", "time", "temp_celcius", "dew_point_temp_celcius", "rel_hum", "wind_dir_10s_deg",
                                         "wind_spd_km_h", "visibility_km", "stn_press_kpa", "humidex", "wind_chill", "weather", "station_name", "province"]
            important_chunk = weather_set_chunk[
                weather_set_chunk.station_name.isin(station_names)]
            important_chunk = important_chunk.loc[(important_chunk["temp_celcius"].notnull())
                                                  & (important_chunk["dew_point_temp_celcius"].notnull()) & (important_chunk["rel_hum"].notnull() & (important_chunk["weather"].notnull())), ["date_time", "year", "month", "day", "time", "temp_celcius", "dew_point_temp_celcius", "rel_hum", "wind_dir_10s_deg",
                                                                                                                                                                                              "wind_spd_km_h", "visibility_km", "stn_press_kpa", "humidex", "wind_chill", "weather", "station_name", "province"]]
            if (len(important_chunk) > 0):
                li.append(important_chunk)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.to_csv(os.path.join(output_path, "priority_weather.csv"), sep=",")
    print("Finished Retrieving weather from priority stations")


def set_up_hour_table_csv(db):
    print("Beggining seting up hour dimension table")
    df_prio = pd.read_csv(os.path.join(input_path, "ontario_1_1.csv"))
    drop = df_prio.drop_duplicates(subset=['Year', 'Month', 'Day', 'Time'])
    drop.insert(0, "key", range(0, len(drop)))
    drop.to_csv(os.path.join(output_path, "hours_dim.csv"))
    print("Finished seting up hour dimension table")


def set_up_hours_on_db(db):
    hours = []
    canada_holidays = holidays.Canada()
    hour_df = pd.read_csv(os.path.join(output_path, "hours_dim.csv"))
    print("Starting Hour Insert")
    for _, row in hour_df.iterrows():
        hour, minute = row['Time'].split(':')

        d = datetime(row['Year'], row['Month'], row['Day'],
                     int(hour), int(minute))
        is_holiday = d in canada_holidays
        hours.append(
            {'key': row["key"],
                'hour_start': d.hour,
                'hour_end': 0 if d.hour + 1 == 24 else d.hour + 1,
                'date': d.date(),
                'day_of_week': DayOfWeek(calendar.day_name[d.weekday()]),
                'day': d.day,
                'month': Month(calendar.month_name[d.month]),
                'year': d.year,
                'weekend': d.weekday() >= 5,
                'holiday': is_holiday,
                'holiday_name': canada_holidays.get(d) if is_holiday else None})

    print(f"# of hour dimensional entries {len(hours)}")
    hour_chunks = list(chunk(hours, 5000))
    for index, hour_chunk in enumerate(hour_chunks):
        print(f"Inserting Hour chunk #{index}")
        db.engine.execute(Hour.__table__.insert(), hour_chunk)
    print("Finished inserting hours")


def clean_weather_data(db):
    hour_dim = pd.read_csv(os.path.join(output_path, "hours_dim.csv"))
    li = []
    batch_no = 1
    print("Cleaning Weather data")
    for batch in pd.read_csv(os.path.join(output_path, "priority_weather.csv"), chunksize=chunksize):
        print(f"Beginning processing weather batch {batch_no}")

        batch["valid"] = batch.apply(lambda row:
                                     check_valid(row["temp_celcius"], row["month"], row["rel_hum"], row["wind_spd_km_h"], row["dew_point_temp_celcius"], row["wind_dir_10s_deg"], row["visibility_km"]), axis=1)
        batch = batch.loc[batch["valid"], ["date_time", "year", "month", "day", "time", "temp_celcius", "dew_point_temp_celcius", "rel_hum", "wind_dir_10s_deg",
                                                                                        "wind_spd_km_h", "visibility_km", "stn_press_kpa", "humidex", "wind_chill", "weather", "station_name", "province"]]
        batch["hour_id"] = batch.apply(lambda row: find_hour_id_weather(
            row["year"], row["month"], row["day"], row["time"], hour_dim), axis=1)
        batch = batch.loc[batch["hour_id"] != -1, ["date_time", "year", "month", "day", "time", "temp_celcius", "dew_point_temp_celcius", "rel_hum", "wind_dir_10s_deg",
                                                   "wind_spd_km_h", "visibility_km", "stn_press_kpa", "humidex", "wind_chill", "weather", "station_name", "province", "hour_id"]]
        if (len(batch) > 0):
            li.append(batch)
        print(f"Batch {batch_no} Processing complete")
        batch_no += 1

    cleaned_weather_frame = pd.concat(li, axis=0, ignore_index=True)
    print(
        f"# of weather rows remaining after taking out null temperatures: {len(cleaned_weather_frame)}")
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


def find_hour_id_weather(year, month, day, hour, hour_dim):
    hour_obj = hour_dim[(hour_dim["Year"] == year) & (hour_dim["Month"] == month) & (
        hour_dim["Day"] == day) & (hour_dim["Time"] == hour)]
    if hour_obj.empty:
        return -1
    return hour_obj.iloc[0]["key"]


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
            'weather': row['weather'] if row["weather"] == row["weather"] else None
        }
        weather_entries.append(weather_entry)

    print(f"# of weather dimensional entries {len(weather_entries)}")
    weather_chunks = list(chunk(weather_entries, 5000))
    for index, weather_chunk in enumerate(weather_chunks):
        print(f"Inserting Weather chunk #{index}")
        db.engine.execute(Weather.__table__.insert(), weather_chunk)
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
    hour_dim = pd.read_csv(os.path.join(output_path, "hours_dim.csv"))
    print("Beggining cleaning of collision data")
    toronto_collision_set = pd.read_csv(os.path.join(input_path, "toronto_collisions.csv"), usecols=[
                                        "DATE", "TIME", "Hour", "STREET1", "STREET2", "LATITUDE", "LONGITUDE", "LOCCOORD", "TRAFFCTL", "VISIBILITY", "LIGHT", "RDSFCOND", "IMPACTYPE", "ACCLASS", "Hood_Name"])
    toronto_collision_set["Hood_Name"] = toronto_collision_set.apply(
        lambda row: re.sub(r'\([0-9]+\)', '', row["Hood_Name"]), axis=1)
    toronto_collision_set["hour_id"] = toronto_collision_set.apply(
        lambda row: find_hour_id_collision_toronto(row["DATE"], row["Hour"], hour_dim), axis=1)
    toronto_collision_set = toronto_collision_set.loc[toronto_collision_set["hour_id"] != -1, [
        "DATE", "TIME", "Hour", "STREET1", "STREET2", "LATITUDE", "LONGITUDE", "LOCCOORD", "TRAFFCTL", "VISIBILITY", "LIGHT", "RDSFCOND", "IMPACTYPE", "ACCLASS", "Hood_Name", "hour_id"]]
    toronto_collision_set.to_csv(os.path.join(
        output_path, "cleaned_toronto_collisions.csv"), sep=",")

    ottawa_collision_set = pd.read_csv(os.path.join(
        output_path, "ottawa_collision_2013_2017.csv"))
    ottawa_collision_set["adjusted_hour"] = 0
    ottawa_collision_set["hour_id"] = -1

    for index, row in ottawa_collision_set.iterrows():
        t = row['TIME']
        t = datetime.strptime(t, '%I:%M:%S %p')
        ottawa_collision_set.at[index, 'adjusted_hour'] = (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
                                                           + timedelta(hours=t.minute//30)).hour

    ottawa_collision_set["hour_id"] = ottawa_collision_set.apply(
        lambda row: find_hour_id_collision_ottawa(row['DATE'], row['adjusted_hour'], hour_dim), axis=1)
    ottawa_collision_set = ottawa_collision_set.loc[ottawa_collision_set["hour_id"] != -1, ['LOCATION', 'LONGITUDE', 'LATITUDE', 'DATE', 'TIME', 'ENVIRONMENT',
                                                                                            'LIGHT', 'SURFACE_CONDITION', 'TRAFFIC_CONTROL', 'COLLISION_CLASSIFICATION', 'IMPACT_TYPE', 'adjusted_hour', 'hour_id']]

    clean_columns = ["ENVIRONMENT", "LIGHT", "SURFACE_CONDITION",
                     "TRAFFIC_CONTROL", "COLLISION_CLASSIFICATION", "IMPACT_TYPE"]
    for attr in clean_columns:
        print(f"Cleaning {attr}")
        ottawa_collision_set[attr] = ottawa_collision_set.apply(lambda row: "" if row[attr] != row[attr] else (row[attr].split(
            "-")[1].strip() if len(row[attr].split("-")) == 2 else " ".join(row[attr].split("-")[1:]).strip()), axis=1)
    ottawa_collision_set.to_csv(os.path.join(
        output_path, "ottawa_collision_2013_2017.csv"))

    print("Finished cleaning traffic data")


def find_hour_id_collision_ottawa(date, hour, hour_dim):
    d = datetime.strptime(date, '%Y-%m-%d')
    hour_obj = hour_dim.loc[(hour_dim["Time"] == f"{str(hour).zfill(2)}:00") & (hour_dim["Month"] == d.month) & (
        hour_dim["Day"] == d.day) & (hour_dim["Year"] == d.year)]
    if hour_obj.empty:
        return -1
    return hour_obj.iloc[0]["key"]


def find_hour_id_collision_toronto(date, hour, hour_dim):
    d = dateutil.parser.parse(date)
    hour_obj = hour_dim.loc[(hour_dim["Time"] == f"{str(hour).zfill(2)}:00") & (hour_dim["Month"] == d.month) & (
        hour_dim["Day"] == d.day) & (hour_dim["Year"] == d.year)]
    if hour_obj.empty:
        return -1
    return hour_obj.iloc[0]["key"]


def fill_time(time):
    t = str(time)
    return t.zfill(4)


def set_up_collisions_frames(db):
    ottawa_collision_frame = pd.read_csv(os.path.join(
        output_path, "ottawa_collision_2013_2017.csv"))
    toronto_collision_frame = pd.read_csv(
        os.path.join(output_path, "cleaned_toronto_collisions.csv"))

    print(f"# of Ottawa Accidents: {len(ottawa_collision_frame)}")
    print(f"# of Toronto Accidents: {len(toronto_collision_frame)}")

    neighbourhoods = pd.read_csv(os.path.join(input_path, "neighbourhood.csv"))
    location_frame = pd.DataFrame(
        columns=["key", "street_name_highway", "intersection_1", "intersection_2", "longitude", "latitude", "neighbourhood", "is_intersection", "original_location"])
    accident_frame = pd.DataFrame(columns=["key", "accident_time", "environment", "road_surface",
                                           "traffic_control", "visibility", "impact_type", "is_fatal", "hour_id", "loc_id"])
    accident_key, location_key = 0, 0
    used_location_ottawa = []
    used_location_toronto = []

    print("Beggining setting up dataframes for Location and Accident")

    for _, row in toronto_collision_frame.iterrows():
        street1, street2, latitude, longitude, neighbourhood = row["STREET1"], row[
            "STREET2"], row["LATITUDE"], row["LONGITUDE"], row["Hood_Name"]

        time, visibility, light, impact_type, hour_id, location_coord, road_surface, traffic_control = str(row["TIME"]).zfill(4), row[
            "VISIBILITY"], row["LIGHT"], row["IMPACTYPE"], row["hour_id"], row["LOCCOORD"], row["RDSFCOND"], row["TRAFFCTL"]
        time = f"{time[:2]}:{time[2:]}"

        if street1 in used_location_toronto:
            location_row = location_frame.loc[location_frame["original_location"]
                                              == street1]
            location_row = location_row.iloc[0]
            accident_frame = accident_frame.append({'key': accident_key, 'accident_time': datetime.strptime(time, "%H:%M").replace(tzinfo=pytz.utc).astimezone(pytz.timezone('EST')).strftime("%I:%M %p"),
                                                    'environment': visibility, 'road_surface': road_surface, 'traffic_control': traffic_control, 'visibility': light, 'impact_type': impact_type, 'is_fatal': True,
                                                    "hour_id": hour_id, 'loc_id': location_row["key"]}, ignore_index=True)
            accident_key += 1
            continue

        location_frame = location_frame.append({"key": location_key, 'street_name_highway': street1.strip(), 'intersection_1': street1.strip(), 'intersection_2': street2.strip(), 'longitude': longitude, 'latitude': latitude,
                                                'neighbourhood': neighbourhood, "is_intersection": location_coord == "Intersection", 'original_location': street1, 'city': "Toronto"}, ignore_index=True)
        accident_frame = accident_frame.append({"key": accident_key, 'accident_time': datetime.strptime(time, "%H:%M").replace(tzinfo=pytz.utc).astimezone(pytz.timezone('EST')).strftime("%I:%M %p"), 'environment': visibility,
                                                'road_surface': road_surface, 'traffic_control': traffic_control, 'visibility': light,
                                                'impact_type': impact_type, 'is_fatal': True, "hour_id": hour_id, 'loc_id': location_key}, ignore_index=True)

        used_location_toronto.append(street1)
        accident_key += 1
        location_key += 1

    for _, row in ottawa_collision_frame.iterrows():
        # handle location
        location, latitude, longitude = row["LOCATION"], row["LATITUDE"], row["LONGITUDE"]
        if location in used_location_ottawa:
            location_row = location_frame.loc[location_frame["original_location"]
                                              == location]
            location_row = location_row.iloc[0]
            accident_frame = accident_frame.append({'key': accident_key, 'accident_time': row['TIME'], 'environment': row['ENVIRONMENT'],  'road_surface': row["SURFACE_CONDITION"], 'traffic_control': row["TRAFFIC_CONTROL"], 'visibility': row[
                'LIGHT'], 'impact_type': row['IMPACT_TYPE'], 'is_fatal': True if row['COLLISION_CLASSIFICATION'] == 'Fatal injury' else False,
                "hour_id": row["hour_id"], 'loc_id': location_row["key"]}, ignore_index=True)
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
                1].strip(), 'longitude': row["LONGITUDE"], 'latitude': row["LATITUDE"], 'neighbourhood': closest_neighbourhood[0], "is_intersection": True, 'original_location': location, 'city': "Ottawa"}
            location_frame = location_frame.append(
                location_dict, ignore_index=True)
            accident_frame = accident_frame.append({'key': accident_key, 'accident_time': row['TIME'], 'environment': row['ENVIRONMENT'], 'road_surface': row["SURFACE_CONDITION"], 'traffic_control': row["TRAFFIC_CONTROL"], 'visibility': row[
                'LIGHT'], 'impact_type': row['IMPACT_TYPE'], 'is_fatal': True if row['COLLISION_CLASSIFICATION'] == 'Fatal injury' else False,
                "hour_id": row["hour_id"], 'loc_id': location_key}, ignore_index=True)
        elif re.match(pattern_2, location):
            location_parts = location.split("@")
            street_name, intersection = location_parts[0], location_parts[1]
            location_dict = {"key": location_key, 'street_name_highway': street_name.strip(), 'intersection_1': intersection.strip(), 'intersection_2': None,
                             'longitude': row["LONGITUDE"], 'latitude': row["LATITUDE"], 'neighbourhood': closest_neighbourhood[0], "is_intersection": False,  'original_location': location, 'city': "Ottawa"}
            location_frame = location_frame.append(
                location_dict, ignore_index=True)
            accident_frame = accident_frame.append({'key': accident_key, 'accident_time': row['TIME'], 'environment': row['ENVIRONMENT'], 'road_surface': row["SURFACE_CONDITION"], 'traffic_control': row["TRAFFIC_CONTROL"], 'visibility': row[
                'LIGHT'], 'impact_type': row['IMPACT_TYPE'], 'is_fatal': True if row['COLLISION_CLASSIFICATION'] == 'Fatal injury' else False,
                "hour_id": row["hour_id"], 'loc_id': location_key}, ignore_index=True)
        elif re.match(pattern_3, location):
            location_parts = location.split("/")
            street_name, intersection_1, intersection_2 = location_parts[
                0], location_parts[1], location_parts[2]
            location_dict = {"key": location_key, 'street_name_highway': street_name.strip(), 'intersection_1': intersection_1.strip(), 'intersection_2': intersection_2.strip(),
                             'longitude': row["LONGITUDE"], 'latitude': row["LATITUDE"], 'neighbourhood': closest_neighbourhood[0], "is_intersection": True, 'original_location': location, 'city': "Ottawa"}
            location_frame = location_frame.append(
                location_dict, ignore_index=True)
            accident_frame = accident_frame.append({'key': accident_key, 'accident_time': row['TIME'], 'environment': row['ENVIRONMENT'], 'road_surface': row["SURFACE_CONDITION"], 'traffic_control': row["TRAFFIC_CONTROL"], 'visibility': row[
                'LIGHT'], 'impact_type': row['IMPACT_TYPE'], 'is_fatal': True if row['COLLISION_CLASSIFICATION'] == 'Fatal injury' else False,
                "hour_id": row["hour_id"], 'loc_id': location_key}, ignore_index=True)
        elif re.match(pattern_4, location):
            location_parts = location.split("@")
            street_name, intersections = location_parts[0], location_parts[1].split(
                "/")
            location_dict = {"key": location_key, 'street_name_highway': street_name.strip(), 'intersection_1': intersections[0].strip(), 'intersection_2': intersections[
                1].strip(), 'longitude': row["LONGITUDE"], 'latitude': row["LATITUDE"], 'neighbourhood': closest_neighbourhood[0], "is_intersection": True, 'original_location': location, 'city': "Ottawa"}
            location_frame = location_frame.append(
                location_dict, ignore_index=True)
            accident_frame = accident_frame.append({'key': accident_key, 'accident_time': row['TIME'], 'environment': row['ENVIRONMENT'], 'road_surface': row["SURFACE_CONDITION"], 'traffic_control': row["TRAFFIC_CONTROL"], 'visibility': row[
                'LIGHT'], 'impact_type': row['IMPACT_TYPE'], 'is_fatal': True if row['COLLISION_CLASSIFICATION'] == 'Fatal injury' else False,
                "hour_id": row["hour_id"], 'loc_id': location_key}, ignore_index=True)
        elif re.match(pattern_5, location):
            location_parts = location.split("btwn")
            street_name, intersection = location_parts[0], location_parts[1]
            location_dict = {"key": location_key, 'street_name_highway': street_name.strip(), 'intersection_1': intersection.strip(), 'intersection_2': None,
                             'longitude': row["LONGITUDE"], 'latitude': row["LATITUDE"], 'neighbourhood': closest_neighbourhood[0], "is_intersection": False, 'original_location': location, 'city': "Ottawa"}
            location_frame = location_frame.append(
                location_dict, ignore_index=True)
            accident_frame = accident_frame.append({'key': accident_key, 'accident_time': row['TIME'], 'environment': row['ENVIRONMENT'], 'road_surface': row["SURFACE_CONDITION"], 'traffic_control': row["TRAFFIC_CONTROL"], 'visibility': row[
                'LIGHT'], 'impact_type': row['IMPACT_TYPE'], 'is_fatal': True if row['COLLISION_CLASSIFICATION'] == 'Fatal injury' else False,
                "hour_id": row["hour_id"], 'loc_id': location_key}, ignore_index=True)

        used_location_ottawa.append(location)

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
            "street_name_highway": row["street_name_highway"] if row["street_name_highway"] == row["street_name_highway"] else None,
            "intersection_1": row["intersection_1"] if row["intersection_1"] == row["intersection_1"] else None,
            "intersection_2": row["intersection_2"] if row["intersection_2"] == row["intersection_2"] else None,
            "longitude": row["longitude"] if row["longitude"] == row["longitude"] else None,
            "latitude": row["latitude"] if row["latitude"] == row["latitude"] else None,
            "neighbourhood": row["neighbourhood"] if row["neighbourhood"] == row["neighbourhood"] else None,
            "city": row["city"] if row["city"] == row["city"] else None
        }
        locations.append(location_entry)

    for _, row in accident_frame.iterrows():
        accident_entry = {
            "key": row["key"],
            "accident_time": row["accident_time"] if row["accident_time"] == row["accident_time"] else None,
            "environment": row["environment"] if row["environment"] == row["environment"] else None,
            "road_surface": row["road_surface"] if row["road_surface"] == row["road_surface"] else None,
            "traffic_control": row["traffic_control"] if row["traffic_control"] == row["traffic_control"] else None,
            "visibility": row["visibility"] if row["visibility"] == row["visibility"] else None,
            "impact_type": row["impact_type"] if row["impact_type"] == row["impact_type"] else None
        }
        accidents.append(accident_entry)

    print(f"# of location dimensional entries {len(locations)}")
    location_chunks = list(chunk(locations, 5000))
    for index, location_chunk in enumerate(location_chunks):
        print(f"Inserting Location chunk #{index}")
        db.engine.execute(Location.__table__.insert(), location_chunk)

    print(f"# of accident dimensional entries {len(accidents)}")
    accident_chunks = list(chunk(accidents, 5000))
    for index, accident_chunk in enumerate(accident_chunks):
        print(f"Inserting Accident chunk #{index}")
        db.engine.execute(Accident.__table__.insert(), accident_chunk)

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

        # in the odd chance that there is no corresponding weather rows, continue to next row
        if len(station_names) == 0:
            continue

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
                                                       hour_id) & (weather_frame["station_name"] == shortest_weather_station_distance[0])]
        if matching_weather_hour_row.empty:
            continue
        matching_weather_hour_row = matching_weather_hour_row.iloc[0]
        accident_facts.append({"hour_key": int(hour_id), "location_key": int(loc_id), "accident_key": int(key),
                               "weather_key": int(matching_weather_hour_row["key"]), "is_fatal": row["is_fatal"], "is_intersection": location_row["is_intersection"]})

    print(f"# of Accident fact entries {len(accident_facts)}")
    accident_facts_dataframe = pd.DataFrame(accident_facts)
    accident_facts_dataframe.to_csv(
        os.path.join(output_path, "accident_facts.csv"))
    db.engine.execute(AccidentFact.__table__.insert(), accident_facts)
    print("Finished Fact Table insertion")
