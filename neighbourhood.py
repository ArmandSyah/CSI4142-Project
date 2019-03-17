from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import os

dir = os.path.dirname(__file__)
input_path = os.path.join(dir, 'input')

geolocator = Nominatim(
    user_agent="CSI4142 Data Science Project")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
neighbourhoods = ["ByWard Market", "The Glebe", "Westboro", "Orleans", "Kanata", "Barrhaven", "Stittsville",
                  "Hintonburg", "New Edinburgh", "Rockcliffe Park", "Old Ottawa South", "Manotick", "Vanier", "Downtown Ottawa",
                  "Lower Town", "Old Ottawa East", "Riverside South", "Little Italy", "Overbrook", "Bayshore", "Centretown West", "Alta Vista",
                  "Osgoode", "Manor Park", "Dunrobin", "Blackburn Hamlet", "Golden Triangle", "Carleton Heights", "South Keys", "Cumberland",
                  "Rockland", "Mooney's Bay", "Navan", "Bells Corners", "Elmvale Acres", "Greely", "Centrepointe", "LeBreton Flats", "North Gower",
                  "Beaverbrook", "McKellar Park", "Katimavik-Hazeldean", "Queenswood Heights", "Morgan's Grant", "Carlington", "Bridlewood", "Island Park",
                  "Tunney's Pasture", "Kars", "Carleton Square", "Pineview", "Nepean"]


def setup_neighbourhood():
    df = pd.DataFrame({"name": neighbourhoods})
    df['location'] = df["name"].apply(geocode)
    df['latitude'] = df['location'].apply(
        lambda loc: loc.raw["lat"] if loc else None)
    df['longitude'] = df['location'].apply(
        lambda loc: loc.raw["lon"] if loc else None)
    df.drop('location', axis=1, inplace=True)
    df.to_csv(os.path.join(input_path, "neighbourhood.csv"))
