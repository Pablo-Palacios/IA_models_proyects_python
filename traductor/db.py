import sqlite3


def create_table_images():
    conn = sqlite3.connect("IA.sqlite3")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS images(id INTEGER PRIMARY KEY AUTOINCREMENT,image TEXT)")
    conn.commit()
    conn.close()


def insert_img_db(img):
    conn = sqlite3.connect("IA.sqlite3")
    cursor = conn.cursor()
    cursor.execute("""INSERT INTO images (image) VALUES (?)""",(img,))
    conn.commit()
    conn.close()



