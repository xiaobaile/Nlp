import pymysql


class MySqlHandler(object):
    def __init__(self, data_path):
        self.host = "localhost"
        self.port = 3306
        self.user = "root"
        self.password = "root"
        self.database = "ccpg_database"
        self.path = data_path

    def open_file(self):
        with open(self.path, mode="r", encoding="UTF-8") as fr:
            lines = fr.readlines()
            return lines

    def run(self):
        db = pymysql.connect(self.host, self.user, self.password, self.database, charset="utf-8", port=self.port)
        cursor = db.cursor()
        query = """insert into ccpg_data_update(style, class, items, title, add_time, buyer, agency, area, 
        url) values (%s, %s, %s, %s, %s, %s, %s, %s, %s) """
        for line in self.open_file():
            temp = line.split("\t")
            line = [str(ll) for ll in temp]
            cursor.execute(query, tuple(line))
        db.commit()
