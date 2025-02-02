import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema import Document
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import time

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
def client_qdrant_init():
    """
    This function initializes the Qdrant object for local use

    Returns:
        - client_qdrant: Qdrant Object
    """

    client_qdrant = QdrantClient(":memory:") # use for computer memory use

    return client_qdrant

def set_up_page():
    st.header('Multimodal RAGP: Personal AI Assistant')
    st.info("""This page is the 'Original RAG Model', where the user can upload a document and ask questions about it.
            \nDepending on the length of the document:\n
            \n   *If the document is short, the program will add the documetn straight to the LLM prompt.\n
            \n   *If the document is long, the document will be embedded into a vector database using OpenAI Embeddings and :memory: local mode""")
    st.write('Drop your own document and ask questions about it!')

# Function to set up the file uploader and change the width
def set_up_doc_uploader():
    """
    This function creates 3 columns and initializes a file uploader 

    Returns the uploaded file to embed
    """

    col1, col2, col3 = st.columns([2, 1, 1])  # build 3 columns to control the width of the container

    with col1:
        uploaded_file = st.file_uploader("Upload your document and ask questions about it!")

    return uploaded_file

# Function to load the pdf and split it in pages
def load_and_split_pdf(uploaded_file):
    """
    This function uses pdfplumber to load a PDF document and split it into pages.
    Each page is converted into a Document object with 'page_content' set to the text.
    
    Arguments: 
        - path for pdf document

    Returns: 
        - pages, list of Document objects
    """

    # Open file with pdfplumber
    with pdfplumber.open(uploaded_file) as pdf: 
        # Extract text from each page and wrap it in a Document object
        pages = [Document(page_content=page.extract_text()) for page in pdf.pages]
    
    return pages

# Text splitter function
def text_splitter_pages(pages):
    """
    This function takes uses the RecursiveCharacterTextSplitter from langchain 
    to split the text into chunks of 700 characters with an overlap of 50

    Arguments: str

    Returns: chukns, list of strings
    """
    # Initiliaze text splitter object
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50) 
    chunks = text_splitter.split_documents(pages) # Split the pages into chunks

    return chunks

# Function to create a Qdrant collection and initialize our Qdrant vector store
def setup_qdrant_collection(embeddings_model, client_qdrant):
    """
    This function creates the collection where the embeddings of the documents.
    Initializes the Vector Store form Qdrant

    Arguments: 
        - embeddings_model, embedding function

    Returns: 
        - vector_store (empty): Qdrant databse    
    """

    client_qdrant.recreate_collection(
        collection_name="user_collection",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client_qdrant,
        collection_name="user_collection",
        embedding=embeddings_model,
        
    )

    print('vector store created successfully')
    return vector_store
    

def insert_embeddings(vector_store, chunks):
    """
    This function takes the Vector Store from Qdrant and the chunks from the text splitter function
    1. Generates ids using the uuid library.
    2. Generates the embeddings from each chunk
    3. Stores chunks and ids in the Vector Store
    4. Stores the Vector Store (with embeddings) into the current session

    Arguments: vector_store, vector store object; chunks, list of str

    Return: user_db: Qdrant Vector Store (with embeddings)
    """

    uuids = [str(uuid4()) for _ in range(len(chunks))] # Create unique ids for each chunk
    user_db = vector_store.add_documents(documents=chunks, ids=uuids) # Generate and store embeddings 
    st.session_state.user_db = user_db # Store Vector Store in the current session to keep available for the user

    return user_db

# Main function to set up collection and insert text chunks
def process_and_store_text_chunks(chunks, embeddings_model, client_qdrant):
    """
    This function takes the chunks generated by the text splitter function,
    Initializes the Qdrant vector store and applies the embeddings function

    Arguments: chunks, list of str; embeddings_model; client_qdrant; collection_name, str

    Return: db, Vector Store (with embeddings)
    """

    if not chunks:
        print("No text chunks provided.")
        return
    
    # Set up collection with the appropriate embedding size
    db = setup_qdrant_collection(embeddings_model, client_qdrant)
    
    # Insert embeddings with unique UUIDs
    insert_embeddings(db, chunks)
    print(f"Successfully inserted {len(chunks)} chunks into collection new collection'.")

    return db

#PDF loader and process function
def upload_document(pages):
    """
    This function takes the uploaded file and applies all the necessary functions to:
    1. Load the pdf
    2. Split in chunks
    3. Get the embeddings 
    4. Build the vector store

    Arguments: uploaded_file, pdf document

    Returns: db, Qdrant vector store
    """
    embeddings_model = openai_embeddings()
    client_qdrant = client_qdrant_init()
    
    with st.spinner("Processing document..."):
        chunks = text_splitter_pages(pages) # Apply text splitter
        db = process_and_store_text_chunks(chunks, embeddings_model, client_qdrant) # Embeddgins and vector store
        print('Document uploaded successfully')
                    
    return db

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
def get_context_from_docs(docs):
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

def get_context_from_pages(pages):
    """
    This function builds a context paragraph for the prompt, using the results from the similarity search function

    Arguments: str

    Return: context, str
    """

    context_from_pages = '\n'
    for page in pages:
        context_from_pages += '\nContext:\n'
        context_from_pages += page.page_content + '\n\n'

    return context_from_pages

def generate_prompt_from_context(query, context):

# Dynamic prompt function
    """
    This functions uses a template to generate a dynamic prompt that can be adapted to the user's query

    Arguments: user_question: str, docs :str
    """

    prompt = f"""
        INTRODUCTION
        You are an expert virtual assistant designed to analyze and provide insights from user-uploaded documents. Your role is to respond to questions based on the content provided, offering clear, concise, and step-by-step answers. Your responses should always be grounded in the context extracted from the user's uploaded document and formatted in Markdown for clarity.

        The user asked: "{query}"

        CONTEXT
        Document Content:
        '''
        {context}
        '''

        RESTRICTIONS
        - Use only the provided context to answer the user's question. Avoid assumptions or external references.
        - If the user's query is outside the scope of the document or lacks sufficient detail, request clarification politely.
        - Maintain a professional and informative tone. Avoid humor, personal opinions, or unrelated topics.
        - Ensure responses reference specific sections or details within the document whenever possible.

        EXAMPLES:
            Example 1:
                User: 'What are the main steps outlined for project management in this document?'
                Agent: 'The document highlights the following main steps for project management:
                        1. Initial project planning, including defining scope and objectives (Section 2).
                        2. Resource allocation and timeline setting (Section 3).
                        3. Regular progress monitoring and risk assessment (Section 4).
                        Refer to sections 2–4 for more detailed instructions.'

            Example 2:
                User: 'Does the document mention safety protocols for chemical handling?'
                Agent: 'Yes, the document outlines safety protocols for chemical handling:
                        - Always wear appropriate personal protective equipment (PPE), such as gloves and goggles.
                        - Follow the proper storage guidelines detailed in Section 8.
                        - Handle chemicals in a well-ventilated area as specified in Section 9B.
                        For further details, refer to sections 8 and 9B.'

        TASK
        Your task is to answer the user's question directly and comprehensively, using only the provided context. Reference specific sections or details whenever applicable, and format your response in Markdown for readability.

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
def get_response(query, context, client_openai):
    """
    This function takes the vector store (with embeddings) and the query from the user
    and applies the different functions to retrieve the data relevant to the query,
    build the prompt for the LLM and get the response from the LLM

    Arguments: db, vector store; query, str

    Return: answer, str
    """

    prompt = generate_prompt_from_context(query, context)
    answer = get_response_from_client(prompt, client_openai)

    return answer

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)

# Function to set up form in Streamlit and display the process
def set_up_user_form(context, query):
    """
    This function build the form element in Streamlit and build the interface for the user

    Arguments: db, vector store; query, str; client_openai, connection with OpenAI API
    """
    client_openai = client_openai_init()

    # Simulate response retrieval
    with st.spinner('Generating response...'):
        answer = get_response(query, context, client_openai) 
    
    # Clear the progress bar
    

    ##### Evaluate response #####
    with st.spinner('Evaluating response...'):
        prompt_for_eval = generate_prompt_for_eval(query, answer) 
        evaluation = get_evaluation_from_LLM_as_a_judge(client_openai, prompt_for_eval)
                

        #####. Display answer
    dash_line = '--------------'
    st.markdown(f"### ---------- _Your Question:_ {query} ----------")

    st.write_stream(stream_data(answer))

    st.write(dash_line)
    st.write("### ---------- _Evaluation from LLM_ ----------")
    st.write_stream(stream_data(evaluation))


##### 4. Functions for OpenAI API #####
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

        Format: Please provide the following:
        Relevance Score: (0-5)
        Accuracy Score: (0-5)
        Completeness Score: (0-5)
        Clarity Score: (0-5)
        Brief Justification: Describe why you assigned these scores based on relevance, accuracy, completeness, and clarity.
        Here is the Question: {query}
        And here is the Answer: {answer}

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

    return evaluation

def main():
    set_up_page()
    uploaded_file = set_up_doc_uploader()
    if uploaded_file is not None:
        query = get_query()
        pages = load_and_split_pdf(uploaded_file)
        if st.button('Submit') and query is not None and pages != 0:
            if len(pages) > 4:
                db = upload_document(pages)
                docs = perform_similarity_search(query, db)
                context = get_context_from_docs(docs)
                set_up_user_form(context, query)
            elif len(pages) <= 4:
                context = get_context_from_pages(pages)
                set_up_user_form(context, query)
        
    else:
        pass
    

main()