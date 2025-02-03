
## Import Necessary Libraries
    
import streamlit as st
import pandas as pd
import pymysql
import time
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from uuid import uuid4
import json


COLLECTIONS = ['Cardiology']
LANGUAGES = ['English', 'Spanish']

# API KEY from OpenAI
#API_KEY = st.secrets.openai_api_key
API_KEY = st.session_state.open_ai_api_key


# Initiliaze Open AI
def client_openai_init():
    """
    This function initializes the OpenAI API with the user's API KEY

    Returns: 
        - client_open, connection with OpenAI API
    """

    client_openai = OpenAI(api_key=API_KEY)

    return client_openai

def openai_embeddings():
    """
    This function builds the embeddings function using the OpenAI API

    Returns: 
        - embedding_model, embedding function from OpenAI
    """

    # Embeddings object from OpenAI
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=API_KEY)

    return embeddings_model

# Initialize Qdrant Client
def qdrant_client_init():
    """
    This function initializes the Qdrant object
    """
    qdrant_client = QdrantClient(
        url=st.secrets.qdrant_cloud_url,
        api_key=st.secrets.qdrant_API_KEY    
    )

    return qdrant_client

def load_cardiac_vector_store():
    """
    Loads or retrieves the Qdrant vector store for the cardiology dataset.

    Returns:
        - vector_store: Qdrant database
    """

    embeddings = openai_embeddings()

    db = QdrantVectorStore.from_existing_collection(
        collection_name="peds_cardiology",
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE,
        prefer_grpc=False,
        distance=Distance.COSINE,
        location=st.secrets.qdrant_cloud_url,
        api_key=st.secrets.qdrant_API_KEY
    )

    return db
##### 1. Set up Page #####

# Function to print the header of the page
def set_up_page():
    """
    This function sets up the header of the page Peds AI
    it also set the 'collection' selectbox to choose the vector store later

    Returns:
        - collection: str
    """

    st.header('Peds Cardiology Chatbot')
    with st.expander('Explanation'):
        st.info("""Langchain RAG Model AI Assistant:
            \n a. Data: Pediatric Cardiology The Essential Pocket Guide. Rights reserved*\n
            \n b. Embeddings: Open AI Embeddings
            \n c. Database: Qdrant Cloud Vector Store
            \n d. LLM: Open AI\n
            \n\n AI CHATBOT to answer questions about Pediatric Cardiology. 
                User's queries will be store using MySQL for further analysis.
            """, 
        icon="ℹ️")
    col1, col2, = st.columns([1,1])
    with col1:
        collection = st.selectbox("   Choose your topic", COLLECTIONS) 
    with col2:
        language = st.selectbox("   Choose your language", LANGUAGES) 

 

    return collection, language


##### 2. Functions to generate the prompt and retrieve the answer from our LLM #####

# User query function
def get_query():
    """
    This function prompts the user to type a question and store the query into the current session

    Return: query, str
    """

    # Initializes the text_input object and gets the query from the user
    query = st.text_area('type your question here', placeholder='e.g.: Tell me about HLHS')

    return query

def transcribe_audio(audio_input, client_openai):
    # Send audio data to OpenAI Whisper API for transcription
    
    with st.spinner("Transcribing audio..."):
        # Call OpenAI Whisper API to transcribe the audio
        response = client_openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_input,
            prompt='Provide an accurate transcription of the audio file. Be aware the user most likely is going to ask about cardiology or nephrology topics'
        )
        # Extract transcript from the response
        transcript = response.text
        st.write(transcript)
    
    return transcript
        
    # You can now use the transcribed text as input for your model or further processing

def pymysql_connection():
    return pymysql.connect(
        host='localhost',  # Replace with your host
        user='root',  # Replace with your username
        password=st.secrets.sql_password,  # Replace with your password
        database='nurses_data'  # Replace with your database name
    )

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
            # Insert data into personal_info table
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

def query_history():
    """
    This function stores the user, question and the topic in a dataframe. 
    """
    
    # Check if user and query are not None before continuing
    if st.session_state.user is not None and st.session_state.query is not None:        
        # Check if the combination of 'user' and 'question' in new_row already exists in nurses_data
        if not ((st.session_state.user_queries['user'] == st.session_state.user) & 
                (st.session_state.user_queries['question'] == st.session_state.query)).any():
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

        else:
            pass
      
    elif st.session_state.query is None:
        pass


# Similarity Search function
def perform_similarity_search(query, db):
    """
    This function takes the query from the user and matches it 
    with the 5 more similar embeddings from the Qdrant db

    Arguments: db: vector store, user_question: str

    Return: docs, list of str
    """

    docs = db.similarity_search(query, k=5)
    return docs  

# Build context for prompt function
def get_document_context(docs):
    """
    This function builds a context paragraph for the prompt, using the results from the similarity search function

    Arguments: str

    Return: context, str
    """

    context = '\n'
    for doc in docs:
        context += '\nContext:\n'
        context += doc.page_content + '\n\n'

    return context

# Dynamic prompt function
def generate_prompt_from_user_query(query, docs, collection, language):
    """
    This functions uses a template to generate a dynamic prompt that can be adapted to the user's query

    Arguments: user_question: str, docs :str
    """

    prompt = f"""
    INTRODUCTION
    You are an expert virtual assistant specializing in pediatric {collection}. Your role is to provide clear, step-by-step answers to questions about medical protocols, required materials, and procedural guidelines relevant to pediatric cardiology nursing practices. Your responses should be informative, precise, and formatted in Markdown.

    The user asked: "{query}"

    CONTEXT
    Pediatric {collection}:
    '''
    {get_document_context(docs)}
    '''

    RESTRICTIONS
    Always refer to specific steps, materials, and procedures exactly as described in the documentation. Provide responses based only on available context; avoid assumptions or interpretations. Inform the user if the requested information is not present in the provided context.
    Maintain a professional tone, avoid humor, and refrain from discussing topics unrelated to pediatric cardiology or nursing practices.
    Request clarification if the user’s question is vague or lacks sufficient detail for a precise response. If the query does not relate to pediatric cardiology protocols, procedures, or required materials, ask for more information.

    EXAMPLES:
        Example 1:
            User: 'How do I prepare a child for a cardiac catheterization procedure?'
            Agent: 'To prepare a child for cardiac catheterization, follow these steps:
                    1. Review the child’s medical history and confirm any allergies.
                    2. Gather necessary materials, including monitoring equipment, sterile catheters, and any required medications.
                    3. Explain the procedure to the child and family in age-appropriate language.
                    4. Ensure consent forms are signed and pre-procedure fasting protocols are followed.
                    For complete guidelines, refer to the cardiac catheterization protocol, section 4A.'

        Example 2:
            User: 'What are the monitoring requirements for a pediatric patient post-cardiac surgery?'
            Agent: 'The post-operative monitoring requirements include:
                    - Continuous cardiac monitoring
                    - Frequent assessment of vital signs and oxygen levels
                    - Checking for signs of infection at the surgical site
                    Ensure all assessments follow the pediatric post-operative care guidelines in section 5C.'

    TASK
    Respond directly and comprehensively to the user’s question using the provided context. Refer to specific sections when additional details are required.
    Format all answers in Markdown for easy readability. you must generate the response in {language}.

    CONVERSATION:
    User: {query}
    Agent:
    """

    return prompt

# Get response from LLM function
def get_response_from_client(prompt, client_openai):
    """
    This function initiliazes an OpenAI chat to generate the response to a query.

    Arguments: prompt: str; client_openai: connection with OpenAI API

    Return: answer, str
    """

    messages = [{'role':'user', 'content':prompt}]
    model_params = {'model': 'gpt-4o-mini', 'temperature': 0.4, 'max_tokens': 500}
    completion = client_openai.chat.completions.create(messages=messages, **model_params, timeout=120)
    answer = completion.choices[0].message.content
    return answer

# COMBINE THE PREVIOUS FUNCTIONS
def get_response(db, query, client_openai, collection, language):
    """
    This function takes the vector store (with embeddings) and the query from the user
    and applies the different functions to retrieve the data relevant to the query,
    build the prompt for the LLM and get the response from the LLM

    Arguments: db, vector store; query, str

    Return: answer, str
    """

    docs = perform_similarity_search(query, db)
    prompt = generate_prompt_from_user_query(query, docs, collection, language)
    answer = get_response_from_client(prompt, client_openai)

    return answer

# Stream data for better user experience
def stream_data(text):
    for char in text:
        yield char 
        time.sleep(0.001)

# Function to set up form in Streamlit and display the process
def set_up_user_form(collection, query, client_openai, language):
    """
    This function build the form element in Streamlit and build the interface for the user

    Arguments: db, vector store; query, str; client_openai, connection with OpenAI API
    """

    db = load_cardiac_vector_store()
    st.session_state.collection = "Cardiology"
    

    col1, col2, = st.columns([1,1])
    ##### Generate response #####
    with col1:
        with st.spinner('Generating response...'):
            answer = get_response(db, query, client_openai, collection, language) 
            with col2:

        ##### Evaluate response #####
                with st.spinner('Evaluating response...'):
                    prompt_for_eval = generate_prompt_for_eval(query, answer) 
                    evaluation = get_evaluation_from_LLM_as_a_judge(client_openai, prompt_for_eval)


                

        #####. Display answer
    dash_line = '--------------'
    #st.markdown(f"##### ---------- _Your Question:_ {query} ----------")

    st.write_stream(stream_data(answer))

    st.write(dash_line)

    try:
        
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            st.metric(label="Relevance Score", value=evaluation['Relevance Score'])
        with c2:
            st.metric(label="Accuracy Score", value=evaluation['Accuracy Score'])
        with c3:
            st.metric(label="Completeness Score", value=evaluation['Completeness Score'])
        with c4:
            st.metric(label="Clarity Score", value=evaluation['Clarity Score'])


    except json.JSONDecodeError as e:
        st.error(f"JSON Decode Error: {e}")
        st.write("Raw Response Debug:", repr(evaluation))

##### 3. Functions for OpenAI API #####
# Generates a new prompt based on the question from the user and the answer generated by our model, for evaluation   
def generate_prompt_for_eval(query, answer):
    """
    This function creates a dynamic prompt that will be used to ask our LLM (in this case, OpenAI)
    to evaluate our model's answer.

    Arguments: query: str, answer: str
    """
    prompt_for_eval = f"""
        Task:
        You are an expert evaluator tasked with assessing the quality of responses generated by an AI model. 
        The model takes a question and provides an answer with a maximum length of 200 tokens. 
        Please evaluate the answer according to the Evaluation Criteria provided below, and provide 4 different scores, 
        one score for each different criteria from 0 to 5,
        with 0 being completely incorrect or irrelevant and 5 being exceptionally accurate, coherent, and comprehensive.

        Evaluation Criteria:
        Relevance: Is the answer directly relevant to the question? Provide a score from 0 to 5
        Accuracy: Does the answer provide correct and factual information? Provide a score from 0 to 5
        Completeness: Does the answer sufficiently cover the main points without missing key information? Provide a score from 0 to 5
        Clarity: Is the answer clear, easy to understand, and well-structured? Provide a score from 0 to 5

        Scoring Scale:
        5: Excellent – Highly accurate, relevant, and complete answer with clear, coherent language.
        4: Good – Mostly accurate and relevant answer, with minor omissions or slight clarity issues.
        3: Adequate – Provides some relevant information but may lack accuracy, completeness, or clarity in parts.
        2: Poor – Limited relevance or accuracy, missing key points, or difficult to understand.
        1: Very Poor – Largely irrelevant or incorrect answer.
        0: No relevance – Completely off-topic or nonsensical answer.

        Format: json format, example:
        
            "Relevance Score":5,"Accuracy Score":4,"Completeness Score":4,"Clarity Score":4
        
        Here is the Question: {query}
        And here is the Answer: {answer}
        Make sure you return the answer in a dict with json format. do not add new lines or unnecessary spaces.

        Thank you."""
    
    return prompt_for_eval

# Our LLM evaluates our model's answer and generates a score with an explanation
def get_evaluation_from_LLM_as_a_judge(client_openai, prompt_for_eval):
    """
    This function calls the LLM designated to be the judge. The judge will evaluate the answer provided by our model,
    and it will return 4 different scores, evaluating the answer using the following criteria:

    Evaluation Criteria:
        Relevance: Is the answer directly relevant to the question? 
        Accuracy: Does the answer provide correct and factual information? 
        Completeness: Does the answer sufficiently cover the main points without missing key information? 
        Clarity: Is the answer clear, easy to understand, and well-structured? 

    Arguments: client: OpenAI object; prompt_for_eval: str
    """

    messages = [{'role':'user', 'content':prompt_for_eval}] 
    model_params = {'model': 'gpt-4o-mini', 'temperature': 0.4, 'max_tokens': 200}
    completion = client_openai.chat.completions.create(messages=messages, **model_params, timeout=120)

    evaluation = completion.choices[0].message.content
    evaluation_dict = json.loads(evaluation)

    return evaluation_dict

def main():
    client_openai = client_openai_init()
    db = load_cardiac_vector_store()
    st.session_state.collection = "Cardiology"
    collection, language = set_up_page()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask something..."):
        st.session_state.query = query # Stores query into current session for later use of the data

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            answer = get_response(db, query, client_openai, collection, language) 
            st.write_stream(stream_data(answer))        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    if st.button('Clear chat History'):
        st.session_state.messages = []
        st.chat_message(st.session_state.messages)

main()

