import numpy as np
import sqlite3
import os
from Face_Recognition import *

conn = sqlite3.connect('smart_database.db')
conn.execute('''CREATE TABLE IF NOT EXISTS person (name TEXT(20) NOT NULL);''')


def load_to_dict():
    conn = sqlite3.connect('smart_database.db')
    database = {}
    data = conn.execute('''SELECT name FROM person''')
    data = data.fetchall()
    for row in data:
        print(row[0])
        database[row[0]] = np.load("data/"+row[0]+".npy")
    return database


def add_person(name, encoding):
    temp = "insert into person(name) values('"+name+"');"
    conn.execute(temp)
    np.save("data\\"+name, encoding)
    print("inserted succesfully")


def del_person(name):
    temp = "delete from person where name='"+name+"';"
    conn.execute(temp)
    database.pop(name)
    os.remove("data//"+name+".npy")



# # add_person("somraj", img_to_encoding("images/mb_2_short.jpg", FRmodel))
# database["vasanth"] = img_to_encoding("images/vasanth_test.jpg", FRmodel)
# database['frame12'] = img_to_encoding("images/frame12.jpg", FRmodel)
# # database["mb"] = img_to_encoding("images/mb_2_short.jpg", FRmodel)
# database["frame21"] = img_to_encoding("images/frame21.jpg", FRmodel)


# add_person("vasanth", img_to_encoding("images/vasanth_test.jpg", FRmodel))
# add_person("frame12", img_to_encoding("images/frame12.jpg", FRmodel))
# add_person("frame21", img_to_encoding("images/frame21.jpg", FRmodel))

load_to_dict()






# del_person("somraj")

conn.commit()
conn.close()





