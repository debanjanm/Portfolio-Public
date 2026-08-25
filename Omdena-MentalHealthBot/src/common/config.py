import os

SQLITE_DB_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "../db.sqlite3"
)
PDF_DB_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "../sandbox/database/PDFs/faiss_index/"
)
