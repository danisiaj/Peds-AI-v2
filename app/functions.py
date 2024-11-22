########################################################################

""" Functions for home.py (Inititializes Streamlit App)"""

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
    st.session_state.vector_store_cards = None
    st.session_state.vector_store_nephro = None
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
        title="Peds Cardiology AI",
        icon=":material/help:",
        )

    respond_1 = st.Page(
        "pages/provider/wound_classifier.py",
        title="X-ray AI",
        icon=":material/healing:",
    )

    admin_1 = st.Page(
        "pages/admin/Education.py",
        title="Education Center",
        icon=":material/person_add:",
    )

    account_pages = [settings, logout_page ]
    request_pages = [request_1]
    respond_pages = [respond_1]
    admin_pages = [admin_1]

    st.logo("images/logo.png", icon_image="images/logo.png", size='large')

    page_dict = {}
    if st.session_state.role == ['Patient | Family']:
        page_dict["Patient"] = request_pages
    if st.session_state.role == 'Provider':
        page_dict["Patient"] = request_pages
        page_dict["Provider"] = respond_pages

    if st.session_state.role == "Educator | Admin":
        page_dict["Patient"] = request_pages
        page_dict["Provider"] = respond_pages
        page_dict["Admin"] = admin_pages


    if len(page_dict) > 0:
        pg = st.navigation(page_dict | {"Account": account_pages})
    else:
        pg = st.navigation([st.Page(login)])

    pg.run()

########################################################################

""" Functions for model_accuracy.py (Calculates the accuracy of the X-Ray prediction model) """

def load_model():
    """
    This function loads the keras model that will be used to predicts the fractures

    Returns:
        - model: keras CNN model
    """
    
    model = tf.keras.models.load_model('./best_model-0.95.keras')

    return model

def load_image(file_path):
    """
    This function decodes a given .png or .jpeg image and returns a tensorflow vector
 
    Arguments:
        - file_path: str, path to image

    Returns:
        - image: tf vector
    """

    image = tf.io.read_file(file_path)

    return image

def preprocess_image(image_path):
    """
    Preprocesses the image to make it compatible with the model.
    - Reads the image from the given path.
    - Resizes the image to (100, 100).
    - Normalizes pixel values to the range [0, 1].
    - Adds a dimension for the CNN

    Argumentss:
        - image_path (str): The path to the image file.

    Returns:
        - tf.Tensor: Preprocessed image tensor.
    """
    image = load_image(image_path)

    # Decode as JPEG or PNG depending on file format
    try:
        image = tf.image.decode_jpeg(image, channels=3)
    except tf.errors.InvalidArgumentError:
        image = tf.image.decode_png(image, channels=3)

    # Resize and normalize the image
    image = tf.image.resize(image, [100, 100])

    # Normalize 
    image = tf.cast(image, tf.float32) / 255.0  

    # 3D per image for CNN
    image = tf.ensure_shape(image, [100, 100, 3])  

    return image

def predict_fracture(image_path, model):
    """
    Predicts whether the image at `image_path` represents a fractured or non-fractured bone.

    Args:
        image_path (str): The path to the image file.
        model: keras CNN model

    Returns:
        str: 'Fractured' or 'Not Fractured' based on the model's prediction.
    """

    # Preprocess the image
    image = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(image)

    # Convert prediction probability to label
    if prediction[0] > 0.5:
        return "Fractured"
    else:
        return "Not Fractured"

def display_prediction(image_path, model):
    """
    Displays the image and model's prediction.

    Args:
        image_path (str): The path to the image file.
        model: keras CNN model
    """
    result = predict_fracture(image_path, model)

    # Display the image
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"Prediction: {result}")
    plt.axis("off")
    plt.show()

def remove_corrupted_images(directory):
    """
    This functino loops through the entire collection of images and removes the corrupted file

    Arguments:
        - directory: str, path to the folder containing the images

    Returns:
        - num_removed: number of corrupted images removed from collection
    """

    num_removed = 0
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # try to read the image file
                img = load_image(file_path)
                # Try decoding it
                try:
                    _ = tf.image.decode_jpeg(img)
                except tf.errors.InvalidArgumentError:
                    _ = tf.image.decode_png(img)
            except:
                print(f"Removing corrupted image: {file_path}")
                num_removed += 1

    print(f"Total corrupted images removed: {num_removed}")

def calculate_fractured_accuracy(model):
    """
    This function loops through all the images in the folder 'fractured' 
    and calculates the percentage of correct predictions

    Arguments:
        - model: keras CNN model

    Returns:
        - fractured_accuracy: float
    """

    # Path to the images with label 'Fractured'
    fractured_images_path = './Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/test/fractured/*'

    i=0 # total number of FRACTURED images
    j=0 # total number of Images classified as FRACTURED

    # Loop through all the images and get the prediction
    for image_path in glob.glob(fractured_images_path):
        result = predict_fracture(image_path, model)
        i+=1
        if result == "Fractured":
            j+=1 

    fractured_accuracy = (j/i)*100 

    return fractured_accuracy

def calculate_not_fractured_accuracy(model):
    """
    This function loops through all the images in the folder 'not fractured' 
    and calculates the percentage of correct predictions

    Arguments:
        - model: keras CNN model
        
    Returns:
        - fractured_accuracy: float
    """

    # Path to the images with label 'Fractured'
    fractured_images_path = './Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/test/not fractured/*'

    i=0 # total number of NOT FRACTURED images
    j=0 # total number of Images classified as  NOT FRACTURED

    # Loop through all the images and get the prediction
    for image_path in glob.glob(fractured_images_path):
        result = predict_fracture(image_path, model)
        i+=1
        if result == "Fractured":
            j+=1 

    not_fractured_accuracy = (j/i)*100 

    return not_fractured_accuracy

########################################################################

""" Functions for edu.py (Education Center Page in Streamlit App)"""

def set_up_page():
    """ This function builds the header for the page"""

    st.header("Education Center")
 
def load_data():
    """
    This function loads a dataset with all the nurses information 
    It also builds a dataframe widget in the app for visualization purposes
    """

    nurses_data = pd.read_csv('./data/nurses_dataset.csv')
    st.markdown('### _My Nurses:_')
    st.dataframe(nurses_data.head(), use_container_width=True, hide_index=True)

def define_db_config():
    """
    Set up MyQSL connection configuration

    Returns:
        - db_config: dict
    """

    db_config = {
        "host": "localhost",
        "user": "root",
        "password": st.secrets.sql_password,
        "database": "nurses"
    }
    
    return db_config

def setup_database_for_user_query(db_config):
    """
    This function builds the table in the dataset in MySQL to store any new question from the user
    *** QUESTIONS COME FROM THE PEDS_AI PAGE

    Arguments:
        - db_config: dict, information to connect to MySQL
    """


    # Establish connection with MySQL
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    # Build the query
    cursor.execute("""
    DROP TABLE IF EXISTS user_questions;

    CREATE TABLE IF NOT EXISTS user_questions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user VARCHAR(255),
        question TEXT,
        topic TEXT
    )
    """)

    # Execute the query
    connection.close()

def store_user_question(db_config, df):
    """
    This function takes the dataframe with all the questions from the users and update the database in MyQSL
    *** QUESTIONS COME FROM THE PEDS_AI PAGE

    Arguments:
        - db_config: dict, information to connect to MySQL
    """

    # Establish connection
    connection = mysql.connector.connect(**db_config)

    # Upload DataFrames to SQL using pymysql connection
    try:
        with connection.cursor() as cursor:
            # Insert data into personal_info table
            queries_columns = ', '.join(df.columns)

            for index, row in df.iterrows():
                query_store_queries = f"INSERT INTO personal_info ({queries_columns}) VALUES ({', '.join(['%s'] * len(df.columns))})"
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

    # Connect to the SQLite database
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Load and execute SQL script
    with open('./data/nurses_1.sql') as file:
        sql_script = file.read()
    cursor.executescript(sql_script)
    conn.commit()
    
    # Execute the specific query and fetch results into a DataFrame
    df = pd.read_sql_query(sql_query, conn)
    
    # Close the connection after retrieving data
    conn.close()

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
            personal_info._id as ID,
            last_name as 'Last Name',
            first_name as 'First Name'
        FROM personal_info
        RIGHT JOIN (
            SELECT
                education_info._id,
                {selection.lower()} AS {selection.upper()}
            FROM education_info 
            WHERE {selection.lower()} = 0) as education
        ON personal_info._id = education._id 
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

    certifications = ["BLS", "PALS", "ACEs", "Cardiology Hours", "Transplant Hours", "Nephrology Hours", "Respiratory Hours"]
    selection = st.selectbox("Filter by Certifications", certifications)  # Select certification

    if selection:
        st.markdown(f'### _Nurses missing {selection.upper()}:_')
        query = sql_query(selection.replace(" ", "_"))  # Generate SQL query
        df = execute_sql_query(query)  # Execute and get the result DataFrame
        st.write("Total nurses:", len(df))  # Display count
        st.dataframe(df, use_container_width=True, hide_index=True)  # Display DataFrame

def analyze_queries():
    """
    This functions builds a dynamic prompt based on the user's questions to create a chat with OpenAI

    Returns:
        analysis: str
    """

    queries = '\n'
    for user, question in zip(st.session_state.nurses_data.user, st.session_state.nurses_data.question):
        queries += '\Queries:\n'
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

    - **Key Words**: [List of key words extracted from the queries]
    - **Most Common Words**: [List of the most common words after filtering stopwords]
    - **Summary**: [A concise paragraph summarizing the common topics in user queries]
    - **Recommendations**:
    1. [First recommendation for training and education]
    2. [Second recommendation for training and education]

    Remember to return the answer in markdown format.
    """

    client_openai = OpenAI(api_key = API_KEY)

    messages = [{'role':'user', 'content':prompt}]
    model_params = {'model': 'gpt-4o-mini', 'temperature': 0.4, 'max_tokens': 500}
    completion = client_openai.chat.completions.create(messages=messages, **model_params, timeout=120)

    answer = completion.choices[0].message.content

    return answer

def query_history():
    """
    This function stores the user, question and the topic in a dataframe. 
    The dataframe is printed for visualizatino purposes and stored in MySQL database
    """

    st.markdown('### _Query History:_')
    # Initialize 'nurses_data' in session state if it doesn't exist
    if 'nurses_data' not in st.session_state:
        st.session_state.nurses_data = pd.DataFrame(columns=['user', 'question'])
        if len(st.session_state.nurses_data) == 0:
            st.markdown("** Query history empty **")
        
    # Check if user and query are not None before continuing
    if st.session_state.user is not None and st.session_state.query is not None:        
        # Check if the combination of 'user' and 'question' in new_row already exists in nurses_data
        if not ((st.session_state.nurses_data.user == st.session_state.user) & 
                (st.session_state.nurses_data.question == st.session_state.query)).any():
            # Merge only if it doesn't exist
            new_row = pd.DataFrame([{
                                'user': st.session_state.user, 
                                'question': st.session_state.query,
                                'topic':st.session_state.collection
                                }])

            st.session_state.nurses_data = pd.concat([st.session_state.nurses_data, new_row], ignore_index=True)
            
            # Update MySQL database with the new datapoint
            setup_database_for_user_query()
            store_user_question(new_row) 

        else:
            pass      


        # Display the updated dataframe
        st.dataframe(st.session_state.nurses_data, use_container_width=True)

        # Option for the user to analyze the questions through an LLM
        if st.button('Analyze queries') and len(st.session_state.nurses_data) > 0:
            analysis = analyze_queries()
            st.markdown(analysis)
   
    elif st.session_state.query is None:
        pass

########################################################################

""" Functions for xray_ai.py (computer Vision: X-Ray AI Page in Streamlit App)"""

def set_up_page():
    """ Set up the header for the page"""
    st.header("Computer Vision: X-Ray Classifier")

def load_model():
    """
    This function loads the keras model that will be used to predicts the fractures

    Returns:
        - model: keras CNN model
    """
    
    model = tf.keras.models.load_model('./best_model-0.95.keras')

    return model

def preprocess_image(uploaded_img):
    """
    Preprocesses the image to make it compatible with the model.
    - Reads the image directly from uploaded bytes.
    - Resizes the image to (100, 100).
    - Normalizes pixel values to the range [0, 1].

    Args:
        uploaded_img (UploadedFile): The uploaded image file from Streamlit.

    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    # Decode the uploaded image
    image = Image.open(uploaded_img).convert("RGB")

    # Resize to the model's input shape
    image = image.resize((100, 100))

    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image) / 255.0

    # Expand dimensions to match the model's expected input shape
    image_tensor = tf.expand_dims(image_array, axis=0)

    return image_tensor

def predict_fracture(preprocessed_image, model):
    """
    Predicts whether the image represents a fractured or non-fractured bone.

    Args:
        preprocessed_image (tf.Tensor): The preprocessed image tensor.

    Returns:
        str: 'Fractured' or 'Not Fractured' based on the model's prediction.
    """
    # Make prediction
    prediction = model.predict(preprocessed_image)

    # Interpret the result
    if prediction[0] > 0.75:
        return f"Fractured, {prediction[0]*100} %" 
    else:
        return f"Not Fractured, {prediction[0]*100} %"

########################################################################

""" Functions for peds_ai.py (Peds Cardiology AI Page in Streamlit App)"""