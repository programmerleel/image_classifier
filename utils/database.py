# -*- coding: utf-8 -*-
# @Time    : 2024/03/05 10:40
# @Author  : LiShiHao
# @FileName: database.py
# @Software: PyCharm

import argparse
import sys
import sqlite3


"""
测试流程：
    生成注册数据库
    生成特征
    与注册库进行比对
    选择 top one

"""
class DBHelper():
    def __init__(self,db_path):
        self.connection = sqlite3.Connection(db_path,check_same_thread=False)
        self.cursor = self.connection.cursor()
        self.create_table_register()

    def create_table_register(self):
        sql = """create table if not exists table_register(category text primary key not null,center_feature blob);"""
        self.cursor.execute(sql)
        self.connection.commit()

    def insert_table_register(self,values):
        sql = """replace into table_register (category,center_feature) values (?,?);"""
        self.cursor.executemany(sql,values)
        self.connection.commit()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument()
    return parser.parse_args(argv)

# def main(args):
#     pass
#
# if __name__ == '__main__':
#     pass
