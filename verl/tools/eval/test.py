from exec_eval import exec_sql


# test database correct?

if __name__ == '__main__':
    db = "music_1"
    print(exec_sql(db, "SELECT DISTINCT artist.artist_name FROM artist JOIN song ON artist.artist_name = song.artist_name WHERE artist.country = 'UK' AND song.languages = 'English'"))

    # read gold.sql
    # TODO: set to your local path to train_sql.sql
    with open('PATH/TO/train_sql.sql', 'r') as f:
        gold = f.readlines()

    cnt = 0
    for i in gold:
        if i == "\n":
            continue
        sql, db = i.strip().split("\t")

        flag, msg = exec_sql(db, sql)
        if flag == 'exception':
            cnt += 1

    print(cnt)
