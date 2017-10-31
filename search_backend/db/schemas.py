from pony.orm import *

db = Database(provider='postgres', user='postgres', password='postgres', host='localhost', database='postgres')


class Index(db.Entity):
    key = Required(str, index=True, unique=True)
    documents = Set('DocumentPosition', index=True)


class Document(db.Entity):
    snippet = Optional(str)
    location = Required(str, index=True, unique=True)
    documentPositions = Set('DocumentPosition', index=True)
    len = Optional(int)


class DocumentPosition(db.Entity):
    document = Required('Document', index=True)
    position = Required(int)
    index = Optional('Index', index=True)
    composite_index(document, index)


db.generate_mapping(create_tables=True)

