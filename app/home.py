## Import necessary librries ##
import streamlit as st

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
if 'vector_store_cards' not in st.session_state:
    st.session_state.vector_store_cards = None
if 'cardiac_defect' not in st.session_state:
    st.session_state.cardiac_defect = None



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
    with st.expander(label='Log in instructions'):
                
        st.info("""
                Login info:\n 
    username/last name = user\n 
    password/MRN = 12345\n
                This APP integrates almost all the concepts learned in my Data Science & Machine Learning Bootcamp through IronHack.
            \n Peds AI App was created by a nurse who works in a Pediatric Cardiology Floor in the hospital. The goal of the app is to enhance education for patients, family members and providers. 
            \n Peds AI adds different features depending WHO is logging in, giving special priviledges to Educators and Administrators.

            \n Enjoy!       
            \n YOU WILL NEED YOUR OPENAI API KEY!""")
    status = False
    st.subheader("Log in")
    role = st.selectbox("   Choose your role", ROLES)

    if role == None:
        pass

    elif role == "Patient | Family": 
        last_name = st.text_input(" Patient's last name and MRN: ", placeholder="Last Name")
        mrn = st.text_input("MRN: ", label_visibility='collapsed', placeholder="Medical Record Number")

        if mrn == st.secrets.MRN and last_name.lower() == st.secrets.patient_last_name:
            status = True
        else:
            pass
                    
    elif role == "Provider":
        username = st.text_input("  Credentials: ", placeholder="Username")
        password = st.text_input("Password", label_visibility='collapsed', placeholder="Password")
        if username.lower() == st.secrets.user_provider and password == st.secrets.password_provider:
            status = True
        else:
            pass

    elif role == "Educator | Admin":
        username = st.text_input("  Admin Creadentials: ", placeholder="Username")
        password = st.text_input("Admin password: ", label_visibility='collapsed', placeholder="Password", type='password')
        if username.lower() == st.secrets.user_admin and password == st.secrets.password_admin:
            status = True
        else:
            pass

    
    col1, col2, col3 = st.columns([1,2,6])
    with col1:
        if st.button("Log in"):
            with col3:
                with st.spinner("Checking credentials..."):
                    if status == True:
                        st.success("Access granted!")
                        st.session_state.role = role
                        if role == 'Patient | Family':
                            st.session_state.user = last_name
                            st.session_state.mrn = mrn

                                
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
    #st.session_state.collection = None
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
        title="Pediatric AI RAG Single Question",
        icon=":material/help:",
        )
    
    request_2 = st.Page(
        "pages/patient/chatbot.py",
        title="Pediatric AI RAG Chatbot",
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

    account_pages = [settings, logout_page ]
    patient_pages = [request_1, request_2]
    provider_pages = [provider_1, provider_2]
    admin_pages = [admin_1]

    #st.logo(image="images/logo_3_copy.png", icon_image="images/Logo.png")

    page_dict = {}
    if st.session_state.role == 'Patient | Family':
        page_dict["Patient"] = patient_pages
    if st.session_state.role == 'Provider':
        page_dict["Patient"] = patient_pages
        page_dict["Provider"] = provider_pages

    if st.session_state.role == "Educator | Admin":
        page_dict["Patient"] = patient_pages
        page_dict["Provider"] = provider_pages
        page_dict["Admin"] = admin_pages


    if len(page_dict) > 0:
        with st.sidebar:
            API_KEY = st.text_input('type you OpenAI API KEY', placeholder='OpenAI API KEY', type='password')
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





