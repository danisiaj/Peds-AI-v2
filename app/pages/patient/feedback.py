import streamlit as st
import pymysql
import pandas as pd

if 'feedback' not in st.session_state:
    st.session_state.feedback = None

def setup_page():
    return st.subheader("Give us some feedback!")

def pymysql_connection():
    """
    This function starts the connection with MySQL for data analytics
    """
    try:
        connection = pymysql.connect(
        host = st.secrets.sql_host,
        user = st.secrets.sql_user,
        password = st.secrets.sql_password,   
        database = st.secrets.database_feedback,
        port = st.secrets.port 
        )
        print("Connection successful!")
        return connection
    except pymysql.MySQLError as e:
        print("Error:", e)


def store_feedback_query(df):
    """
    This function stores the feedback from the user into MySQL database
    Arguments: 
        - df: Dataframe to store in MYSQL

    """

    # Establish connection
    connection = pymysql_connection()

    # Upload DataFrames to SQL using pymysql connection
    try:
        with connection.cursor() as cursor:
            # Insert data into user_questions table
            queries_columns = ', '.join(df.columns)

            for index, row in df.iterrows():
                query_store_queries = f"INSERT INTO user_feedback ({queries_columns}) VALUES ({', '.join(['%s'] * len(df.columns))})"
                cursor.execute(query_store_queries, tuple(row))
            
            # Commit the transaction
            connection.commit()
            print("DataFrames uploaded successfully.")
    except Exception as e:
        print("Error while uploading DataFrames:", e)
    finally:
        connection.close() 

def main():
    feedback = st.text_area(label="Did you enjoy it? What can we do better?", placeholder="type your feedback here")
    if st.button("Submit feedback"):
        new_row = pd.DataFrame([{
                                'user': st.session_state.user, 
                                'feedback': feedback,
                                'role':st.session_state.role
                                }])
        
        store_feedback_query(new_row)

main()