import click
import xlrd
import csv
import os
from timeit import default_timer as timer
from flask import Flask
from models import Base
from datastaging import import_data, import_hours, import_weather, import_collision, setup_facttable
from neighbourhood import setup_neighbourhood
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['SQLALCHEMY_DATABASE_URI']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


@app.cli.command()
def initdb():
    try:
        Base.metadata.create_all(bind=db.engine)
    except:
        click.echo('Something went wrong')
        raise

    click.echo('db tables successfully made')


@app.cli.command()
def dropdb():
    try:
        Base.metadata.drop_all(bind=db.engine)
    except:
        click.echo('Something went wrong')
        raise

    click.echo('db tables successfully dropped')


@app.cli.command()
def setupneighbourhoodcsv():
    setup_neighbourhood()


@app.cli.command()
def importdata():
    start = timer()
    print("Starting data import")
    current_dir = os.getcwd()
    final_directory = os.path.join(current_dir, r'output')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    import_data(db)
    end = timer()
    print(f"{end - start} seconds")


@app.cli.command()
def setupoutput():
    current_dir = os.getcwd()
    final_directory = os.path.join(current_dir, r'output')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)


@app.cli.command()
def importhour():
    start = timer()
    print("Starting Hours import")
    import_hours(db)
    end = timer()
    print(f"{end - start} seconds")


@app.cli.command()
def importweather():
    start = timer()
    print("Starting weather import")
    import_weather(db)
    end = timer()
    print(f"{end - start} seconds")


@app.cli.command()
def importcollision():
    start = timer()
    print("Starting accident and location import")
    import_collision(db)
    end = timer()
    print(f"{end - start} seconds")


@app.cli.command()
def setupfacttable():
    start = timer()
    print("Starting fact table set up")
    setup_facttable(db)
    end = timer()
    print(f"{end - start} seconds")
