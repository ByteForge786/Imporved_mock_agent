import streamlit as st
import json
from typing import Dict, Any
import pandas as pd
import yaml
from sql_agent import SQLReActAgent
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="SQL Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .sql-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 4px;
        font-family: 'Courier New', Courier, monospace;
    }
    .thought-box {
        background-color: #e8f0fe;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        border-left: 3px solid #4285f4;
    }
    .metadata-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .conversation-message {
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #f0f2f6;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #e8f0fe;
        margin-right: 20%;
    }
    .table-info {
        margin: 0.5rem 0;
        padding: 1rem;
        border-radius: 4px;
        background-color: #fff;
        border: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

def display_sql_analysis(result: Dict[str, Any]):
    """Display SQL analysis results in a structured format"""
    if result.get("status") == "success":
        for action in result.get("actions", []):
            with st.expander(f"🔍 {action['tool'].replace('_', ' ').title()}", expanded=True):
                # Display thought process
                if action.get("thought"):
                    st.markdown("**Thought Process:**")
                    st.markdown(f"""<div class="thought-box">{action['thought']}</div>""", 
                              unsafe_allow_html=True)

                # Display tool-specific results
                if action["tool"] == "schema_lookup":
                    if action["result"].get("relevant_tables"):
                        st.markdown("**Relevant Tables:**")
                        cols = st.columns(len(action["result"]["relevant_tables"]))
                        for idx, table in enumerate(action["result"]["relevant_tables"]):
                            with cols[idx]:
                                st.markdown(f"""
                                <div class="table-info">
                                    <h4>{table['name']}</h4>
                                    <p><strong>Relevance:</strong> {table['relevance']}</p>
                                    <p><strong>Description:</strong> {table['description']}</p>
                                    <p><strong>Columns:</strong> {', '.join(table['columns'])}</p>
                                </div>
                                """, unsafe_allow_html=True)

                elif action["tool"] == "sql_generation":
                    if action["result"].get("sql"):
                        st.markdown("**Generated SQL:**")
                        st.markdown(f"""<div class="sql-box">{action["result"]["sql"]}</div>""",
                                  unsafe_allow_html=True)
                        st.markdown("**Explanation:**")
                        st.write(action["result"].get("explanation", ""))

                elif action["tool"] == "sql_validation":
                    if action["result"].get("is_safe") is not None:
                        status_color = "green" if action["result"]["is_safe"] else "red"
                        st.markdown(f"""
                        <div class="metadata-box">
                            <h4>Validation Results:</h4>
                            <p style='color: {status_color}'>
                                {'✅ Query is safe' if action["result"]["is_safe"] else '❌ Issues Found'}
                            </p>
                            {f"<p><strong>Issues:</strong></p><ul>" + ''.join([f"<li>{issue}</li>" for issue in action["result"].get("issues", [])]) + "</ul>" if action["result"].get("issues") else ""}
                            {f"<p><strong>Feedback:</strong></p><p>{action['result'].get('feedback', '')}</p>" if action["result"].get("feedback") else ""}
                        </div>
                        """, unsafe_allow_html=True)

                elif action["tool"] == "db_execution":
                    if action["result"].get("data") is not None:
                        st.markdown("**Query Results:**")
                        df = pd.DataFrame(action["result"]["data"])
                        st.dataframe(df, use_container_width=True)
                        
                        metadata = action["result"]["metadata"]
                        st.markdown(f"""
                        <div class="metadata-box">
                            <h4>Query Metadata:</h4>
                            <ul>
                                <li>Rows: {metadata["row_count"]}</li>
                                <li>Columns: {', '.join(metadata["columns"])}</li>
                                <li>Execution Time: {metadata.get("execution_time", "N/A")}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

def display_conversation_message(message: Dict[str, Any]):
    """Display a conversation message"""
    st.markdown(f"""
    <div class="conversation-message {'user-message' if message['role'] == 'user' else 'assistant-message'}">
        {message['content']}
    </div>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = SQLReActAgent("schema.yaml")
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            st.error("Failed to initialize SQL Assistant. Please check the logs.")
            st.stop()

def load_schema_info():
    """Load and display schema information in the sidebar"""
    try:
        with open("schema.yaml", "r") as f:
            schema = yaml.safe_load(f)
            st.sidebar.markdown("### 📚 Available Data Tables")
            for table_name, info in schema["tables"].items():
                with st.sidebar.expander(f"📋 {table_name.title()}"):
                    st.markdown(f"**Description:** {info['description']}")
                    if info.get("sample_questions"):
                        st.markdown("**Sample Questions:**")
                        for q in info["sample_questions"]:
                            st.markdown(f"- {q}")
    except Exception as e:
        logger.error(f"Error loading schema: {str(e)}")
        st.sidebar.error("Failed to load schema information")

def main():
    st.title("🤖 SQL Chat Assistant")
    st.markdown("""
    Welcome to the SQL Chat Assistant! I can help you analyze data using natural language queries.
    Ask me about sales, products, customers, or suppliers - or type 'help' to learn more.
    """)
    
    # Initialize session state
    initialize_session_state()
    
    # Load schema information in sidebar
    load_schema_info()
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            display_conversation_message(message)
        else:
            if isinstance(message.get("content"), str):
                display_conversation_message(message)
            elif isinstance(message.get("content"), dict):
                result = message["content"]
                if result["type"] == "sql_analysis":
                    display_sql_analysis(result)
                elif result["type"] == "conversation":
                    display_conversation_message({
                        "role": "assistant",
                        "content": result["response"]
                    })
                elif result["type"] == "error":
                    st.error(f"Error: {result['message']}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        display_conversation_message({
            "role": "user",
            "content": prompt
        })
        
        # Get agent response
        try:
            with st.spinner("Processing your query..."):
                result = st.session_state.agent.process_query(prompt)
                
                # Add assistant message
                
# Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result
                })
                
                # Display result
                if result["type"] == "sql_analysis":
                    display_sql_analysis(result)
                elif result["type"] == "conversation":
                    display_conversation_message({
                        "role": "assistant",
                        "content": result["response"]
                    })
                elif result["type"] == "error":
                    st.error(f"Error: {result['message']}")
                
                logger.info(f"Successfully processed query: {prompt}")
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            st.error("An error occurred while processing your query. Please try again.")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.agent.clear_memory()
            logger.info("Chat history cleared")
            st.rerun()
        
        # Export chat history
        if st.session_state.messages and st.button("📥 Export Chat History"):
            try:
                chat_history = pd.DataFrame(st.session_state.messages)
                st.download_button(
                    label="Download Chat History",
                    data=chat_history.to_csv(index=False),
                    file_name="sql_assistant_chat_history.csv",
                    mime="text/csv"
                )
                logger.info("Chat history exported")
            except Exception as e:
                logger.error(f"Error exporting chat history: {str(e)}")
                st.error("Failed to export chat history")
        
        # Add feedback section
        st.markdown("### 📝 Feedback")
        feedback = st.text_area("Share your feedback or report issues:")
        if feedback and st.button("Submit Feedback"):
            try:
                # In a real application, you would send this to a feedback collection system
                logger.info(f"Feedback received: {feedback}")
                st.success("Thank you for your feedback!")
            except Exception as e:
                logger.error(f"Error saving feedback: {str(e)}")
                st.error("Failed to submit feedback")
        
        # Add help information
        with st.expander("ℹ️ Help & Tips"):
            st.markdown("""
            **Tips for better results:**
            1. Be specific about what data you want to analyze
            2. Mention the time period if relevant
            3. Specify any grouping or filtering criteria
            4. Ask for specific metrics or calculations
            
            **Example queries:**
            - "Show me total sales by region for the last month"
            - "Which products have the highest profit margin?"
            - "What is the customer distribution by segment?"
            - "List top 10 suppliers by reliability score"
            """)

def handle_error(error: Exception, error_type: str):
    """Handle errors with appropriate logging and user feedback"""
    error_message = str(error)
    logger.error(f"{error_type} error: {error_message}\n{traceback.format_exc()}")
    
    user_friendly_messages = {
        "initialization": "Failed to start the SQL Assistant. Please try refreshing the page.",
        "query_processing": "Had trouble processing your query. Please try rephrasing or simplifying it.",
        "schema_loading": "Could not load database information. Please check if schema.yaml exists.",
        "state_management": "Session state error. Please clear your chat history and try again."
    }
    
    st.error(user_friendly_messages.get(error_type, "An unexpected error occurred. Please try again."))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_error(e, "initialization")                  
