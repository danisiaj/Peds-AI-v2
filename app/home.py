## Import necessary librries ##
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from openai import OpenAI


## Define the differnt roles for the user ##
ROLES = [None, "Patient | Family", "Provider", "Educator | Admin"]
# Initiate all the session variables
if "role" not in st.session_state:
    st.session_state.role = None
if "user" not in st.session_state:
    st.session_state.user = None
if "mrn" not in st.session_state:
    st.session_state.mrn = None
if "last_name" not in st.session_state:
    st.session_state.last_name = None
if "query" not in st.session_state:
    st.session_state.query = None
if 'user_queries' not in st.session_state:
    st.session_state.user_queries = None
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'db' not in st.session_state:
    st.session_state.db = None
if 'cardiac_defect' not in st.session_state:
    st.session_state.cardiac_defect = None





## Define all the functions ##
# API KEY from OpenAI
API_KEY = st.secrets.openai_api_key


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

def login():
    """
    This function controls the log in process of the user and allows access to certain pages, based on user's role.
    it also sets st.session_state for user/last_name, role, mrn(for patients).
    After credentials are verified with st.secrets, the function loads the cardiology and nephrology databases, 
    storing them in st.session_state for efficiency and speed.

    ST.SECRETS NEEDED!!
    """
    with st.expander(label='Learn more about this app'):
                
        st.info("""Peds AI App was created by a nurse who works in a Pediatric Cardiology Floor in the hospital.
                The goal of the app is to enhance education for patients, family members and providers.\n
            \nAI CHATBOT to answer questions about Pediatric Cardiology.
                \nLangchain RAG Model AI Assistant:\n
            \n \tData: Pediatric Cardiology The Essential Pocket Guide. Rights reserved*
            \n \tEmbeddings: Open AI Embeddings
            \n \tDatabase: Qdrant Cloud Vector Store  
            \n \tLLM: Open AI\n
              
            \nUser's queries will be stored using MySQL for further analysis.
            \n Peds AI adds different features depending what ROLE the user chooses, giving special priviledges to Educators and Administrators.

            \n Enjoy!
        """)       
    status = False
    st.subheader("Start")
    role = st.selectbox("   Choose your role", ROLES)


    if role == None:
        pass

    elif role != None:
        user = st.text_input(" Your name", placeholder="Your name")
        col1, col2, col3 = st.columns([1,2,6])
        with col1:
            if st.button("Start"):
                st.session_state.role = role
                st.session_state.user = user
                with col2:
                    with st.spinner('Loading Qdrant Vectorstore...'):
                        database = load_cardiac_vector_store()
                        st.session_state.db = database
                
                st.rerun()   

### THIS CODE IS TO APPLY USER AND PASSWROD SECURITY TO THE APP ###

    # if role == None:
    #     pass

    # elif role == "Patient | Family": 
    #     last_name = st.text_input(" Patient's last name and MRN: ", placeholder="Last Name")
    #     mrn = st.text_input("MRN: ", label_visibility='collapsed', placeholder="Medical Record Number")

    #     if mrn == st.secrets.MRN and last_name.lower() == st.secrets.patient_last_name:
    #         status = True
    #     else:
    #         pass
                    
    # elif role == "Provider":
    #     username = st.text_input("  Credentials: ", placeholder="Username")
    #     password = st.text_input("Password", label_visibility='collapsed', placeholder="Password")
    #     if username.lower() == st.secrets.user_provider and password == st.secrets.password_provider:
    #         status = True
    #     else:
    #         pass

    # elif role == "Educator | Admin":
    #     username = st.text_input("  Admin Creadentials: ", placeholder="Username")
    #     password = st.text_input("Admin password: ", label_visibility='collapsed', placeholder="Password", type='password')
    #     if username.lower() == st.secrets.user_admin and password == st.secrets.password_admin:
    #         status = True
    #     else:
    #         pass

    
    # col1, col2, col3 = st.columns([1,2,6])
    # with col1:
    #     if st.button("Log in"):
    #         with col3:
    #             with st.spinner("Checking credentials..."):
    #                 if status == True:
    #                     st.success("Access granted!")
    #                     st.session_state.role = role
    #                     if role == 'Patient | Family':
    #                         st.session_state.user = last_name
    #                         st.session_state.mrn = mrn

                                
    #                     elif role in ['Educator | Admin', 'Provider']:
    #                         st.session_state.user = username

    #                     st.rerun()   
                    
    #                 elif status == False:
    #                     with col3:
    #                         st.error("Access denied!")
    # with col2:
    #     create =  st.button("Create Account")

def logout():
    """
    This function erases all the session information and returns to the log in page
    """

    st.session_state.role = None
    st.session_state.user = None
    st.session_state.mrn = None
    st.session_state.last_name = None
    st.session_state.query = None
    st.session_state.query_id = None

    st.rerun()

def create_account():
    """
    This function allows the user to create a new account.
    After username and password meet requirements, credentials are stored in st.secrets file
    """

    new_username = st.text_input("Username", 
                                placeholder="Username")
    new_password = st.text_input("Password", 
                                label_visibility='collapsed', 
                                placeholder="Password", 
                                type='password')
    new_password_confirm = st.text_input("Confirm Password", 
                                        label_visibility='collapsed', 
                                        placeholder="confirm Password",
                                        type='password')
    
    if submitted:
        if len(new_username) > 0 and new_password == new_password_confirm:
            st.success("Account created!")
            new_username = st.secrets["test_user"]["username"]
            new_password = st.secrets["test_user"]["password"]

        elif len(new_username) == 0:
            st.error("username cannot be empty")
        elif new_password != new_password_confirm:
            st.warning("passwords do not match")
        submitted = st.button("Submit")

    else:
        pass

def set_up_home_pages():
    """
    This function builds all the different pages and permissions, based on the user's role.
    It also builds the st.navigation sidebar after log in.
    """

    logout_page = st.Page(logout, 
        title="Log out", 
        icon=":material/logout:"
        )
    
    settings = st.Page("pages/settings.py", 
        title="Settings", 
        icon=":material/settings:"
        )
    
    request_1 = st.Page(
        "pages/patient/chatbot.py",
        title="Pediatric AI RAG Chatbot",
        icon=":material/help:",
        )
    
    request_2 = st.Page(
        "pages/patient/peds_ai.py",
        title="Pediatric AI RAG Single Question",
        icon=":material/help:",
        )

    provider_1 = st.Page(
        "pages/provider/xray_ai.py",
        title="X-Ray AI Diagnostics",
        icon=":material/healing:",
    )

    provider_2 = st.Page(
        "pages/provider/assistant_ai.py",
        title="Personal AI RAG Model",
        icon=":material/info:",
    )

    admin_1 = st.Page(
        "pages/admin/edu.py",
        title="Education Center",
        icon=":material/book:",
    )

    account_pages = [logout_page ]
    patient_pages = [request_1]
    provider_pages = [provider_1, provider_2]
    admin_pages = [admin_1]

    st.logo(image="app/images/chat-logo.png", icon_image="app/images/chat-logo.png")

    page_dict = {}
    if st.session_state.role == 'Patient | Family':
        page_dict["Patient"] = patient_pages
    if st.session_state.role == 'Provider':
        page_dict["Patient"] = patient_pages
        #page_dict["Provider"] = provider_pages

    if st.session_state.role == "Educator | Admin":
        page_dict["Patient"] = patient_pages
        #page_dict["Provider"] = provider_pages
        page_dict["Admin"] = admin_pages


    if len(page_dict) > 0:
        with st.sidebar:
            #API_KEY = st.text_input('type you OpenAI API KEY', placeholder='OpenAI API KEY', type='password')
            st.session_state.open_ai_api_key = API_KEY

            pg = st.navigation(page_dict | {"Account": account_pages})
    else:
        pg = st.navigation([st.Page(login)])



    pg.run()

def main():
    """
    This function calls the CSS style file and sets all the instantiates all the session variables
    It also sets up the different pages of the app
    """
    set_up_home_pages()


## Initialize the app ##
if __name__ == "__main__":
    main()





