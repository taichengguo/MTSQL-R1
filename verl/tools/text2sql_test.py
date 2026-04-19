from eval.exec_eval import exec_sql


if __name__ == '__main__':

    sql = "SELECT Name, Milliseconds FROM Track ORDER BY Milliseconds DESC LIMIT 1"
    db = "chinook_1"
    flag, results = exec_sql(db, sql)
    print(flag)
    print(results)