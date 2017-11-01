from pony.orm import *
# from psycogreen.gevent import patch_psycopg
# patch_psycopg()

db = Database(provider='postgres', user='postgres', password='postgres', host='localhost', database='postgres')


class Index(db.Entity):
    key = Required(str, index=True, unique=True)
    documents = Set('DocumentPosition', index=True)


class Document(db.Entity):
    snippet = Optional(str)
    location = Required(str, index=True, unique=True)
    documentPositions = Set('DocumentPosition', index=True)
    len = Optional(int)
    vector = Optional(float)


class DocumentPosition(db.Entity):
    document = Required('Document', index=True)
    position = Required(int)
    index = Optional('Index', index=True)
    composite_index(document, index)


class IndexTitle(db.Entity):
    key = Required(str, index=True, unique=True)
    titles = Set('TitlePosition', index=True)


class Title(db.Entity):
    titlePositions = Set('TitlePosition', index=True)
    len = Optional(int)
    location = Required(str, index=True, unique=True)
    vector = Optional(float)


class TitlePosition(db.Entity):
    title = Required('Title', index=True)
    position = Required(int)
    index = Optional('IndexTitle', index=True)
    composite_index(title, index)


db.generate_mapping(create_tables=True)

