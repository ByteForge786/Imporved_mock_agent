import streamlit as st
import json
from typing import Dict, Any
import pandas as pd
import yaml
import traceback
from queue import Queue
from threading import Event
from sql_agent import SQLReActAgent
import logging
import plotly.express as px
import plotly.graph_objects as go

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
    page_title="SQL Analytics Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main {
        padding: 1rem 2rem;
    }
    .stAlert {
        padding: 12px;
        margin: 12px 0;
        border-radius: 8px;
    }
    .sql-box {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        font-family: 'Monaco', 'Courier New', monospace;
        border: 1px solid #e9ecef;
    }
    .thought-box {
        background-color: #fff;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #3b71ca;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 8px;
        margin: 12px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        max-width: 80%;
    }
    .user-message {
        background-color: #e7f3fe;
        margin-left: 20%;
        border: 1px solid #cce5ff;
    }
    .assistant-message {
        background-color: #ffffff;
        margin-right: 20%;
        border: 1px solid #e9ecef;
    }
    .processing-status {
        padding: 0.8rem;
        border-radius: 8px;
        margin: 8px 0;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
    }
    .result-box {
        margin: 10px 0;
        padding: 1.2rem;
        border-radius: 8px;
        background-color: white;
        border: 1px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

class ChatUI:
    def __init__(self):
        self.initialize_session_state()
        self.clarification_queue = Queue()
        self.clarification_event = Event()
        
    def initialize_session_state(self):
        """Initialize session state with error handling"""
        try:
            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "agent" not in st.session_state:
                st.session_state.agent = SQLReActAgent(
                    "schema.yaml",
                    clarification_queue=self.clarification_queue,
                    clarification_event=self.clarification_event
                )
            if "processing" not in st.session_state:
                st.session_state.processing = False
                
        except Exception as e:
            logger.error(f"Session state initialization error: {traceback.format_exc()}")
            raise InitializationError("Failed to initialize chat session")

    def display_message(self, message: Dict[str, Any], realtime: bool = False):
        """Display chat messages"""
        try:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            
            if isinstance(content, str):
                st.markdown(
                    f"""<div class="chat-message {'user-message' if role == 'user' else 'assistant-message'}">
                        {content}</div>""",
                    unsafe_allow_html=True
                )
            elif isinstance(content, dict):
                self.display_analysis_result(content, realtime)
                
        except Exception as e:
            logger.error(f"Message display error: {traceback.format_exc()}")
            st.error("Error displaying message")

    def display_analysis_result(self, result: Dict[str, Any], realtime: bool = False):
        """Display analysis results with real-time updates"""
        try:
            if result.get("status") != "success":
                st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                return

            for action in result.get("actions", []):
                if realtime:
                    # For real-time updates, use empty placeholders
                    status_placeholder = st.empty()
                    result_placeholder = st.empty()
                    
                    with status_placeholder:
                        st.markdown(f"""<div class="processing-status">
                            🔄 {action['tool'].replace('_', ' ').title()} in progress...</div>""",
                            unsafe_allow_html=True)
                    
                    with result_placeholder:
                        self.display_tool_result(action)
                        
                else:
                    # For final results, use normal display
                    with st.expander(f"📊 {action['tool'].replace('_', ' ').title()}", expanded=True):
                        self.display_tool_result(action)

        except Exception as e:
            logger.error(f"Analysis display error: {traceback.format_exc()}")
            st.error("Error displaying analysis results")

    def display_tool_result(self, action: Dict[str, Any]):
        """Display individual tool results"""
        # Display thought process
        if action.get("thought"):
            st.markdown(f"""<div class="thought-box">
                <strong>Analysis:</strong><br>{action['thought']}</div>""",
                unsafe_allow_html=True)

        tool_result = action.get("result", {})
        
        # Display based on tool type
        if action["tool"] == "schema_lookup":
            self.display_schema_info(tool_result)
        elif action["tool"] == "sql_generation":
            self.display_sql_info(tool_result)
        elif action["tool"] == "sql_validation":
            self.display_validation_info(tool_result)
        elif action["tool"] == "db_execution":
            self.display_execution_results(tool_result)

    def display_schema_info(self, result: Dict[str, Any]):
        """Display schema information"""
        if relevant_tables := result.get("relevant_tables"):
            cols = st.columns(len(relevant_tables))
            for idx, table in enumerate(relevant_tables):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="result-box">
                        <h4>{table['name']}</h4>
                        <p><strong>Relevance:</strong> {table['relevance']}</p>
                        <p><strong>Description:</strong> {table['description']}</p>
                        <p><strong>Columns:</strong> {', '.join(table['columns'])}</p>
                    </div>
                    """, unsafe_allow_html=True)

    def display_sql_info(self, result: Dict[str, Any]):
        """Display SQL query information"""
        if sql := result.get("sql"):
            st.code(sql, language="sql")
            if explanation := result.get("explanation"):
                st.markdown(f"""<div class="thought-box">
                    <strong>Query Explanation:</strong><br>{explanation}</div>""",
                    unsafe_allow_html=True)

    def display_validation_info(self, result: Dict[str, Any]):
        """Display validation results"""
        is_safe = result.get("is_safe")
        if is_safe is not None:
            st.markdown(f"""
            <div class="result-box">
                <h4>Validation Results</h4>
                <p>Status: {"✅ Safe to execute" if is_safe else "❌ Issues found"}</p>
                {f"<strong>Issues:</strong><ul>{''.join([f'<li>{issue}</li>' for issue in result.get('issues', [])])}</ul>" if result.get('issues') else ""}
                {f"<strong>Feedback:</strong><br>{result.get('feedback', '')}" if result.get('feedback') else ""}
            </div>
            """, unsafe_allow_html=True)

    def display_execution_results(self, result: Dict[str, Any]):
        """Display query results"""
        if data := result.get("data"):
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            
            if metadata := result.get("metadata"):
                st.markdown(f"""
                <div class="result-box">
                    <h4>Query Statistics</h4>
                    <ul>
                        <li>Rows: {metadata['row_count']}</li>
                        <li>Columns: {', '.join(metadata['columns'])}</li>
                        <li>Execution Time: {metadata.get('execution_time', 'N/A')}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    def handle_user_input(self, user_input: str):
        """Process user input with real-time updates"""
        try:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Create placeholders for real-time updates
            status_placeholder = st.empty()
            result_placeholder = st.empty()
            
            def update_ui_callback(state: Dict[str, Any]):
                """Callback for real-time UI updates"""
                status = state.get("status")
                
                if status == "waiting_feedback":
                    # Show clarification request
                    with status_placeholder:
                        st.info(state.get("question"))
                        clarification = st.text_input(
                            "Your response:",
                            key=f"clarification_{len(st.session_state.messages)}"
                        )
                        if st.button("Submit", key=f"submit_{len(st.session_state.messages)}"):
                            self.clarification_queue.put(clarification)
                            self.clarification_event.set()
                
                elif status == "processing":
                    # Show current action
                    with status_placeholder:
                        if thought := state.get("current_action", {}).get("thought"):
                            st.markdown(f"""<div class="processing-status">
                                🔄 {thought}</div>""", unsafe_allow_html=True)
                    
                    # Show partial results
                    with result_placeholder:
                        if result := state.get("current_action", {}).get("result"):
                            self.display_analysis_result({"status": "success", "actions": [state["current_action"]]}, realtime=True)
                
                elif status == "completed":
                    status_placeholder.empty()
                    result_placeholder.empty()

            # Process query with real-time updates
            st.session_state.processing = True
            result = st.session_state.agent.process_query(
                user_input,
                callback=update_ui_callback
            )
            st.session_state.processing = False
            
            # Add final response
            st.session_state.messages.append({
                "role": "assistant",
                "content": result
            })
            
        except Exception as e:
            logger.error(f"Query processing error: {traceback.format_exc()}")
            st.error("Error processing your request. Please try again.")

    def show_chat_interface(self):
        """Display main chat interface"""
        st.title("📊 SQL Analytics Assistant")
        
        # Display chat history
        for message in st.session_state.messages:
            self.display_message(message)
        
        # Chat input
        if not st.session_state.processing:
            if user_input := st.chat_input("Ask about your data..."):
                self.handle_user_input(user_input)
        
        # Sidebar
        self.show_sidebar()

    def show_sidebar(self):
        """Display sidebar with controls and schema info"""
        with st.sidebar:
            st.markdown("### Controls")
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
            
            st.markdown("### Available Data")
            try:
                with open("schema.yaml", "r") as f:
                    schema = yaml.safe_load(f)
                    for table, info in schema["tables"].items():
                        with st.expander(f"📋 {table}"):
                            st.write(f"**Description:** {info['description']}")
                            if samples := info.get("sample_questions"):
                                st.write("**Example Questions:**")
                                for q in samples:
                                    st.write(f"- {q}")
            except Exception as e:
                logger.error(f"Schema loading error: {traceback.format_exc()}")
                st.error("Schema information unavailable")

class InitializationError(Exception):
    """Custom exception for initialization errors"""
    pass

def main():
    try:
        chat_ui = ChatUI()
        chat_ui.show_chat_interface()
    except Exception as e:
        logger.error(f"Application error: {traceback.format_exc()}")
        st.error("An error occurred. Please refresh the page.")

if __name__ == "__main__":
    main()
