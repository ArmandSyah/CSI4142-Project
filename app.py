import click
from flask import Flask
from models import *
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = r'<Put db uri here>'
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
