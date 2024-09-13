import streamlit as st
import os
from dotenv import load_dotenv
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase

# Load environment variables from .env file
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(page_title="SQL Query AI", layout="centered")

st.title("Optimized SQL Query AI with LangChain")

# Sidebar inputs for DB credentials
st.sidebar.header("Database Configuration")
user = st.sidebar.text_input("PostgreSQL User")
password = st.sidebar.text_input("PostgreSQL Password", type="password")
host = st.sidebar.text_input("PostgreSQL Host")
port = st.sidebar.text_input("PostgreSQL Port")
dbname = st.sidebar.text_input("PostgreSQL Database")

# Cache the database connection using st.cache_resource to avoid pickling issues
@st.cache_resource
def get_db_connection(user, password, host, port, dbname):
    db_uri = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return SQLDatabase.from_uri(db_uri)

# Cache the agent executor to avoid re-instantiating it, using _db to avoid hashing errors
@st.cache_resource
def get_agent_executor(_db):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    toolkit = SQLDatabaseToolkit(db=_db, llm=llm)
    return create_sql_agent(llm=llm, toolkit=toolkit, verbose=False, agent_type=AgentType.OPENAI_FUNCTIONS)

# Initialize session state for spinner
if 'loading' not in st.session_state:
    st.session_state.loading = False

# Button to establish connection and cache the agent
if st.sidebar.button("Connect to Database"):
    if user and password and host and port and dbname:  # Ensure all fields are filled
        db = get_db_connection(user, password, host, port, dbname)
        if db:
            agent_executor = get_agent_executor(db)  # Pass db to _db in cached function
            if agent_executor:
                st.success("Connected to the database and agent successfully!")
                
                # Query input area
                query_input = st.text_area("Enter your SQL query in natural language")
                
                # Execute the query when the button is clicked
                if st.button("Run Query"):
                    st.session_state.loading = True
                    with st.spinner("Running query..."):
                        try:
                            # Run the query with the cached agent executor
                            response = agent_executor.run(query_input)
                            # Optionally, you can handle the response here or do something with it
                        except Exception as e:
                            st.error(f"Query execution error: {str(e)}")
                        finally:
                            st.session_state.loading = False
        else:
            st.error("Database connection failed. Please check your credentials.")
    else:
        st.error("Please fill in all fields.")

st.sidebar.markdown("**Powered by LangChain & OpenAI**")

