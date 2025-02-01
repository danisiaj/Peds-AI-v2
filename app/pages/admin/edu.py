## Import necessary libraries ##
import streamlit as st
import pandas as pd
import pymysql
import plotly.express as px
from openai import OpenAI

## OpenAI API KEY ##
API_KEY = st.session_state.open_ai_api_key
MYSQL_PASSWORD = st.secrets.sql_password

## Define the functions ##

def set_up_page():
    """ This function builds the header for the page"""

    st.header("MySQL: Education Center")
    st.info("""This service provides access a database in MySQL that contains:
            \n 1. Personal, Work and Education information about the nursing team.
            \n 2. The queries/questions made by the users in the Pediatric AI RAG models.""")
 
def load_data():
    """
    This function loads a dataset with all the nurses information 
    It also builds a dataframe widget in the app for visualization purposes
    """

    nurses_data = pd.read_csv('app/data/nurses_dataset.csv')

    return nurses_data

def pymysql_connection():
    """
    This function starts the connection with MySQL for data analytics
    """

    return pymysql.connect(
        host='127.0.0.1',  
        user='remote_user',  
        password=MYSQL_PASSWORD,  
        database='nurses_data'  
    )

def load_queries_database():
    """
    This function loads the collection that contains all the questions users have made to the RAG model
    for further analysis and visualization

    Returns: User's queries in a pandas dataframe
    """

    connection = pymysql_connection()
    cursor = connection.cursor()

    cursor.execute('SELECT * FROM user_questions;')

    results = cursor.fetchall()

    # Getting column names
    columns = [desc[0] for desc in cursor.description]

    # Converting the results to a pandas DataFrame
    user_queries_df = pd.DataFrame(results, columns=columns)
    st.session_state.user_queries = user_queries_df
    st.dataframe(st.session_state.user_queries, use_container_width=True, hide_index=True)

    # Closing the connection
    cursor.close()
    connection.close()

    return user_queries_df

def setup_database_for_user_query():
    """
    This function builds the table in the dataset in MySQL to store any new question from the user
    *** QUESTIONS COME FROM THE PEDS_AI PAGE

    Arguments:
        - db_config: dict, information to connect to MySQL
    """


    # Establish connection with MySQL
    connection = pymysql_connection()
    cursor = connection.cursor()

    # Build the query
    cursor.execute("""

        CREATE TABLE IF NOT EXISTS user_questions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user VARCHAR(255),
            question TEXT,
            topic TEXT
            )
        """
    )

    # Execute the query
    connection.commit()
    connection.close()

def store_user_question(df):
    """
    This function takes the dataframe with all the questions from the users and update the database in MyQSL
    *** QUESTIONS COME FROM THE PEDS_AI PAGE

    Arguments:
        - db_config: dict, information to connect to MySQL
    """

    # Establish connection
    connection = pymysql_connection()

    # Upload DataFrames to SQL using pymysql connection
    try:
        with connection.cursor() as cursor:
            # Insert data into user_questions table
            queries_columns = ', '.join(df.columns)

            for index, row in df.iterrows():
                query_store_queries = f"INSERT INTO user_questions ({queries_columns}) VALUES ({', '.join(['%s'] * len(df.columns))})"
                cursor.execute(query_store_queries, tuple(row))
            
            # Commit the transaction
            connection.commit()
            print("DataFrames uploaded successfully.")
    except Exception as e:
        print("Error while uploading DataFrames:", e)
    finally:
        connection.close() 

def execute_sql_query(sql_query):
    """
    This function executes the sql_query generated by a filter chosen by the user in the app
    
    Arguments:
        - sql_query: str

    Returns: 
        - df: Dataframe with the filter information
    """

    connection = pymysql_connection()
    cursor = connection.cursor()
    cursor.execute(sql_query)

    # Fetching all results
    results = cursor.fetchall()

    # Getting column names
    columns = [desc[0] for desc in cursor.description]

    # Converting the results to a pandas DataFrame
    df = pd.DataFrame(results, columns=columns)



    # Closing the connection
    cursor.close()
    connection.close()
    return df

def sql_query(selection):
    """
    This function generates a dynamic SQL query based on the selected certification from user in app

    Arguments:
        - selection: str

    Returns:
        - sql_query: str
    """

    sql_query = f"""
        SELECT  
            personal_info._id AS ID,
            last_name AS 'Last Name',
            first_name AS 'First Name',
            work_info.shift AS 'Shift',
            work_info.employment_status AS 'Employment Status'
        FROM personal_info
        RIGHT JOIN (
            SELECT
                education_info._id,
                {selection.lower()} AS {selection.upper()}
            FROM education_info 
            WHERE {selection.lower()} = False
        ) AS education
        ON personal_info._id = education._id 
        LEFT JOIN work_info
        ON personal_info._id = work_info._id
        ORDER BY last_name ASC;
    """

    return sql_query

def certifications_filters():
    """
    - Build the options for the user to filter information about the nurses certifications
    - Generate the sql_query
    - Execute the query
    - build the Dataframe with the filter information
    """
    nurses_data = load_data()
    certifications = [None, "BLS", "PALS", "ACEs", "Cardiology Hours", "Transplant Hours", "Nephrology Hours", "Respiratory Hours"]
    selection = st.selectbox("Filter by Certifications", certifications)  # Select certification
    st.write('________________________')
    if selection is not None:
        col1, col2 = st.columns([2,3])

        with col1:
            query = sql_query(selection.replace(" ", "_"))  # Generate SQL query
            df = execute_sql_query(query)  # Execute and get the result DataFrame
            st.dataframe(df[['ID', 'First Name', 'Last Name']], use_container_width=False, hide_index=True)  # Display DataFrame
        with col2:
            st.metric(label=f'## _Nurses missing {selection.upper()}:_', value=f'{len(df)} ({(len(df)/len(nurses_data)*100)} %)')
            st.write('_________________________________')
            df_for_visualization = df[['Shift', 'Employment Status']]
            full_time_count_am = df[(df['Employment Status'] == 'Full Time') & (df['Shift'] == 'AM')].shape[0]
            part_time_count_am = df[(df['Employment Status'] == 'Part Time') & (df['Shift'] == 'AM')].shape[0]
            prn_time_count_am = df[(df['Employment Status'] == 'PRN') & (df['Shift'] == 'AM')].shape[0]

            full_time_count_pm = df[(df['Employment Status'] == 'Full Time') & (df['Shift'] == 'PM')].shape[0]
            part_time_count_pm = df[(df['Employment Status'] == 'Part Time') & (df['Shift'] == 'PM')].shape[0]
            prn_time_count_pm = df[(df['Employment Status'] == 'PRN') & (df['Shift'] == 'PM')].shape[0]

            st.markdown('_Day Shift Nurses:_')
            col2_1, col2_2, col2_3 = st.columns([1,1,1])
            with col2_1:
                st.metric(label="FT Employees", value=full_time_count_am)
            with col2_2:
                st.metric(label="PT Employees", value=part_time_count_am)
            with col2_3:
                st.metric(label="PRN Employees", value=prn_time_count_am)

            st.markdown('_Night Shift Nurses:_')
            col3_1, col3_2, col3_3 = st.columns([1,1,1])
            with col3_1:
                st.metric(label="FT Employees", value=full_time_count_pm)
            with col3_2:
                st.metric(label="PT Employees", value=part_time_count_pm)
            with col3_3:
                st.metric(label="PRN Employees", value=prn_time_count_pm)

        col3, col4 = st.columns([1,1])
        with col3:
            fig_1 = px.pie(df_for_visualization, names='Shift', title='Shift Distribution')
            st.plotly_chart(fig_1)
            shift_counts = df_for_visualization['Shift'].value_counts()
            st.bar_chart(shift_counts, use_container_width=False, width=300, height=150, horizontal=True)
        with col4:
            fig_2 = px.pie(df_for_visualization, names='Employment Status', title='Employment Distribution')
            st.plotly_chart(fig_2)
            status_counts = df_for_visualization['Employment Status'].value_counts()
            st.bar_chart(status_counts, use_container_width=False, width=300, height=150, horizontal=True)

        df_for_visualization['Shift & Status'] = df_for_visualization['Shift'] + ' - ' + df_for_visualization['Employment Status']

        # Calculate value counts for each combination
        value_counts = df_for_visualization['Shift & Status'].value_counts().reset_index()
        value_counts.columns = ['Shift & Status', 'Count']

        # Create a pie chart using the value counts and show counts in labels
        fig_3 = px.pie(
            value_counts, 
            names='Shift & Status', 
            values='Count', 
            title='Shift and Employment Status Distribution',
            labels={'Shift & Status': 'Shift & Status'},
            hover_data=['Count'],  # Add count info on hover
        )
        st.plotly_chart(fig_3)

            

    else:
        st.markdown('### _All Nurses:_')
        st.dataframe(nurses_data, use_container_width=True, hide_index=True)

def analyze_queries():
    """
    This functions builds a dynamic prompt based on the user's questions to create a chat with OpenAI

    Returns:
        analysis: str
    """

    queries = '\n'
    for user, question in zip(st.session_state.user_queries.user, st.session_state.user_queries.question):
        queries += r'\Queries:\n'
        queries += f'user: {str(user)} -> {str(question)}\n\n'

    prompt = f"""
    You are a data analyst specializing in user behavior analysis. 
    I will provide you with a series of queries made by users. Your task is to:

    1. Analyze the queries to extract the **key words** from each query, ignoring all stopwords.
    2. Identify the **most common words** across all the queries after removing stopwords.
    3. Generate a **brief paragraph summarizing the most common topics** that users inquire about based on the extracted key words.
    4. Propose **two actionable recommendations** for improving training and education related to the identified topics.

    Here are the user queries:
        {queries}     
    Provide your response in the following format:

    - **Key Words**: In bullet points, a list of key words extracted from the queries after filtering stopwords
    - **Most Common Words**: In bullet points, a list of the most common words after filtering stopwords, in desendent order]
    - **Summary**: A concise paragraph summarizing the common topics in user queries
    - **Recommendations**:
            1. First recommendation for training and education
            2. Second recommendation for training and education

    Remember to return the answer in markdown format.
    """

    client_openai = OpenAI(api_key = API_KEY)

    messages = [{'role':'user', 'content':prompt}]
    model_params = {'model': 'gpt-4o-mini', 'temperature': 0.4, 'max_tokens': 1000}
    completion = client_openai.chat.completions.create(messages=messages, **model_params, timeout=120)

    answer = completion.choices[0].message.content

    return answer

def query_history():
    """
    This function stores the user, question and the topic in a dataframe. 
    The dataframe is printed for visualization purposes and stored in MySQL database
    """

    st.markdown('### _Query History:_')
    # Initialize 'nurses_data' in session state if it doesn't exist

    # Check if user and query are not None before continuing
    if st.session_state.user is not None and st.session_state.query is not None and st.session_state.user_queries is not None: 
               
        # Check if the combination of 'user' and 'question' in new_row already exists in nurses_data
        if not ((st.session_state.user_queries.user == st.session_state.user) & 
                (st.session_state.user_queries.question == st.session_state.query)).any():
            # Merge only if it doesn't exist
            new_row = pd.DataFrame([{
                                'user': st.session_state.user, 
                                'question': st.session_state.query,
                                'topic':st.session_state.collection
                                }])
            user_queries = pd.concat([st.session_state.user_queries, new_row], ignore_index=True)
            st.session_state.user_queries = user_queries
            
            # Update MySQL database with the new datapoint
            setup_database_for_user_query()
            store_user_question(new_row) 
            load_queries_database()

        else:
            load_queries_database()

    else: 
        load_queries_database()

    # Option for the user to analyze the questions through an LLM
    if st.button('Analyze queries') and len(st.session_state.user_queries) > 0:
        analysis = analyze_queries()
        st.markdown(analysis)

def main():

    set_up_page()
    tab1, tab2 = st.tabs(["Nurses", "Queries"])
    with tab1:
        st.info('This tab uses data from a CSV file for data visualization.')
        certifications_filters()
    with tab2:
        st.info('This tab uses MySQL to load the dataframe containing the queries made by different users')
        query_history()



## Initialize the app   
main()