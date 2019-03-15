import click
import xlrd
import csv
import os
from timeit import default_timer as timer
from flask import Flask
from models import *
from datastaging import import_data
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = r''
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
def importdata():
    start = timer()
    current_dir = os.getcwd()
    final_directory = os.path.join(current_dir, r'output')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    import_data(db)
    end = timer()
    print(f"{end - start} seconds")


@app.cli.command()
def xlsx2csv2017():
    wb = xlrd.open_workbook(
        'C:\\Users\Armand Syahtama\Downloads\h2017collisionsfinal.xlsx')
    sh = wb.sheet_by_name('AllCollisions2017LatLong')
    your_csv_file = open('C:\\projects\\2017collisionsfinal.xls.csv', 'w')
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    your_csv_file.close()


@app.cli.command()
def makenewfolder():
    current_dir = os.getcwd()
    final_directory = os.path.join(current_dir, r'output')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
