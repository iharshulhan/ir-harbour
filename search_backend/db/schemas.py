from pony.orm import *

db = Database(provider='postgres', user='postgres', password='postgres', host='localhost', database='postgres')


class Index(db.Entity):
    key = Required(str, index=True, unique=True)
    documents = Set('DocumentIndex', index=True)


class Document(db.Entity):
    snippet = Optional(str)
    location = Required(str, index=True, unique=True)
    documentIndexes = Set('DocumentIndex', index=True)


class DocumentIndex(db.Entity):
    document = Required('Document', index=True)
    index = Required('Index', index=True)
    positions = Set('DocumentIndexPosition', index=True)


class DocumentIndexPosition(db.Entity):
    position = Required(int)
    documentIndex = Required('DocumentIndex', index=True)


db.generate_mapping(create_tables=True, check_tables=True)

