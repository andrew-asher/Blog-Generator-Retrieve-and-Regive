from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['knowledge_base']
collection = db['documents']

def insert_document(name, report_type, notes, pages, file_path):
    document = {
        "name": name,
        "type": report_type,
        "notes": notes,
        "page_count": pages,
        "location": file_path
    }
    collection.insert_one(document)