import sqlite3
import os
from config import BASE_DIR

PATH_DB = os.path.join(BASE_DIR, 'data', 'realec.db')


def connect(dbfile):
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    return c, conn


c, conn = connect(PATH_DB)
c.execute("SELECT TEXT, MARK, NAME, TYPE FROM MAIN")
result = c.fetchall()

