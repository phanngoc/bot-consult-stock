from peewee import *
import datetime

db = SqliteDatabase('chatbot.db')

class BaseModel(Model):
    class Meta:
        database = db

# class User(BaseModel):
#     username = CharField(unique=True)

class Message(BaseModel):
    user_id = TextField()
    role = TextField()
    message = TextField()
    created_date = DateTimeField(default=datetime.datetime.now)

db.connect()
db.create_tables([Message])