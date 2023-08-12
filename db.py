from concurrent.futures import thread
from os import name
from altair import Description
from peewee import *
import datetime

db = SqliteDatabase('chatbot.db')

class BaseModel(Model):
    class Meta:
        database = db

# class User(BaseModel):
#     username = CharField(unique=True)

class Thread(BaseModel):
    user_id = TextField()
    description = TextField()
    created_date = DateTimeField(default=datetime.datetime.now)

class Message(BaseModel):
    user_id = TextField()
    role = TextField(null=True)
    name = TextField(null=True) # function name of openai API
    message = TextField(null=True) # message of bot
    message_gpt = TextField(null=True) # message of openai API
    created_date = DateTimeField(default=datetime.datetime.now)
    thread = ForeignKeyField(Thread, backref='messages', null=True)


db.connect()
db.create_tables([Thread, Message])