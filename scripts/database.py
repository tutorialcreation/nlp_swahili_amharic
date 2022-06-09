from sqlalchemy import types, create_engine
import pandas as pd
from sqlite3 import Error
from scripts.logger import logger
import csv

class DBOps:
    """
    What this script does:
    - inserts data from json into sqlite (online)
    - inserts data from json into mysql 
    """

    def __init__(self,df=None,is_online=True):
        if is_online:
            try:
                self.conn = create_engine('sqlite:///data_science.db') # ensure this is the correct path for the sqlite file. 
                logger.info("SQLITE Connection Sucessfull!!!!!!!!!!!")
            except Exception as err:
                logger.error("SQLITE Connection Failed !!!!!!!!!!!")
                logger.error(err)
        else:
            try:
                self.conn = create_engine('mysql+pymysql://root:luther1996-@localhost/db')
                logger.info("MySQL Connection Sucessfull!!!!!!!!!!!")
            except Exception as err:
                logger.error("MySQL Connection Failed !!!!!!!!!!!")
                logger.error(err)

        self.df = df
        
    def get_engine(self):
        """
        - this function simply returns the connection
        """
        return self.conn
    
    def execute_from_script(self,sql_script):
        """
        - this function executes commands
        that come streaming in from sql_scripts
        """
        try:
            sql_file = open(sql_script)
            sql_ = sql_file.read()
            sql_file.close()

            sql_commands = sql_.split(";")
            for command in sql_commands:
                if command:
                    self.conn.execute(command)
            logger.info("Successfully created table")
        except Error as e:
            logger.error(e)
        return

    def insert_update_data(self,table):
        """
        - this function pushes data into the table
        """
        self.df.to_sql(table, con=self.conn, if_exists='replace')
        logger.info("Successfully pushed the data into the database")
        return 
    
if __name__ == "__main__":
    logger.info("Test DBOpsfile")