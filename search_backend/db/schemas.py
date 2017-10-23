from pony.orm import *

db = Database(provider='postgres', user='postgres', password='postgres', host='localhost', database='postgres')


class Index(db.Entity):
    key = Required(str, index=True, unique=True)
    documents = Set('DocumentPosition')


class Document(db.Entity):
    snippet = Optional(str)
    location = Required(str, index=True, unique=True)
    documentPositions = Set('DocumentPosition')


class DocumentPosition(db.Entity):
    document = Required('Document', index=True)
    position = Required(int)
    index = Optional('Index')


db.generate_mapping(create_tables=True, check_tables=True)

