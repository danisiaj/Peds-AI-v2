## Import necessary librries ##
import streamlit as st
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_openai import OpenAIEmbeddings
from qdrant_client.http.models import Distance

## Define the differnt roles for the user ##
ROLES = [None, "Patient | Family", "Provider", "Educator | Admin"]

## Define all the functions ##
def load_css(file_name):
    """
    This function calls the CSS file to set up the design of the app

    Arguments:
        - file_name: str, path to the styles.css file
    """

    with open(file_name) as f:
        return f"<style>{f.read()}</style>"

def login():
    """
    This function controls the log in process of the user and allows access to certain pages, based on user's role.
    it also sets st.session_state for user/last_name, role, mrn(for patients).
    After credentials are verified with st.secrets, the function loads the cardiology and nephrology databases, 
    storing them in st.session_state for efficiency and speed.

    ST.SECRETS NEEDED!!
    """
    
    status = False
    st.subheader("Log in")
    role = st.selectbox("   Choose your role", ROLES)

    if role == None:
        pass

    elif role == "Patient | Family": 
        last_name = st.text_input(" Patient's last name and MRN: ", placeholder="Last Name")
        mrn = st.text_input("MRN: ", label_visibility='collapsed', placeholder="Medical Record Number")

        if mrn == st.secrets.MRN and last_name == st.secrets.patient_last_name:
            status = True
        else:
            pass
                    
    elif role == "Provider":
        username = st.text_input("  Credentials: ", placeholder="Username")
        password = st.text_input("Password", label_visibility='collapsed', placeholder="Password")
        if username == st.secrets.user_provider and password == st.secrets.password_provider:
            status = True
        else:
            pass

    elif role == "Educator | Admin":
        username = st.text_input("  Admin Creadentials: ", placeholder="Username")
        password = st.text_input("Admin password: ", label_visibility='collapsed', placeholder="Password", type='password')
        if username == st.secrets.user_admin and password == st.secrets.password_admin:
            status = True
        else:
            pass

    
    col1, col2, col3 = st.columns([1,2,6])
    with col1:
        if st.button("Log in"):
            with col3:
                with st.spinner("Checking credentials..."):
                    if status == True:
                        with st.spinner("Loading database..."):
                            db_cardiology = load_cardiac_vector_store()
                            st.session_state['vector_store_cards'] = db_cardiology
                            db_nephrology = load_nephro_vector_store()
                            st.session_state['vector_store_nephro'] = db_nephrology
                            st.success("Access granted!")
                            st.session_state.role = role
                            if role == 'Patient | Family':
                                st.session_state.user = last_name
                                st.session_state.mrn = mrn
                                with st.sidebar:
                                    st.write(f"Welcome, {st.session_state.user}, MRN:{st.session_state.mrn}")
                            elif role in ['Educator | Admin', 'Provider']:
                                st.session_state.user = username

                        st.rerun()   
                    
                    elif status == False:
                        with col3:
                            st.error("Access denied!")
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
    st.session_state.collection = None
    #st.session_state.vector_store_cards = None
    #st.session_state.vector_store_nephro = None
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

def openai_embeddings():
    """
    This function builds the embeddings function using the OpenAI API
    API_KEY from OpenAI needed!

    Returns: 
        -embedding_model, embedding function from OpenAI
    """

    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=st.secrets.openai_api_key)

    return embeddings_model

def load_cardiac_vector_store():
    """
    This function loads the "CARDIOLOGY" vector store from Qdrant server.
    DOCKER DESKTOP needed to connect to Qdrant server!
    Embeddings model needed to retrieve vectors in DENSE mode!

    Returns:
        - vector_store: Qdrant database
    """

    embeddings = openai_embeddings()

    vector_store = QdrantVectorStore.from_existing_collection(
            collection_name='cardiology',
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            prefer_grpc=False,
            url="localhost:6333",
            distance=Distance.COSINE,
    )

    return vector_store

def load_nephro_vector_store():
    """
    This function loads the "NEPHROLOGY" vector store from Qdrant server.
    DOCKER DESKTOP needed to connect to Qdrant server!
    Embeddings model needed to retrieve vectors in DENSE mode!

    Returns:
        - vector_store: Qdrant database
    """    
  
    embeddings = openai_embeddings()

    vector_store = QdrantVectorStore.from_existing_collection(
            collection_name='nephrology',
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            prefer_grpc=False,
            url="localhost:6333",
            distance=Distance.COSINE,
    )

    return vector_store

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
        "pages/patient/peds_ai.py",
        title="Pediatric AI",
        icon=":material/help:",
        )

    provider_1 = st.Page(
        "pages/provider/xray_ai.py",
        title="X-Ray AI",
        icon=":material/healing:",
    )

    provider_2 = st.Page(
        "pages/provider/assistant_ai.py",
        title="Personal AI",
        icon=":material/info:",
    )

    admin_1 = st.Page(
        "pages/admin/edu.py",
        title="Education Center",
        icon=":material/book:",
    )

    account_pages = [settings, logout_page ]
    patient_pages = [request_1]
    provider_pages = [provider_1, provider_2]
    admin_pages = [admin_1]

    st.logo("images/logo_3 copy.png", icon_image="images/logo.png", size='large')

    page_dict = {}
    if st.session_state.role == ['Patient | Family']:
        page_dict["Patient"] = patient_pages
    if st.session_state.role == 'Provider':
        page_dict["Patient"] = patient_pages
        page_dict["Provider"] = provider_pages

    if st.session_state.role == "Educator | Admin":
        page_dict["Patient"] = patient_pages
        page_dict["Provider"] = provider_pages
        page_dict["Admin"] = admin_pages


    if len(page_dict) > 0:
        pg = st.navigation(page_dict | {"Account": account_pages})
    else:
        pg = st.navigation([st.Page(login)])

    pg.run()

def main():
    """
    This function calls the CSS style file and sets all the instantiates all the session variables
    It also sets up the different pages of the app
    """

    # Initiate CSS style file
    css = load_css("./styles/style.css")
    st.markdown(css, unsafe_allow_html=True)

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
    if "query_id" not in st.session_state:
        st.session_state.query_id = None
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'vector_store_cards' not in st.session_state:
        st.session_state.vector_store_cards = None
    if 'vector_store_nephro' not in st.session_state:
        st.session_state.vector_store_nephro = None

    # Build the pages for the app
    set_up_home_pages()

## Initialize the app ##
if __name__ == "__main__":
    main()





