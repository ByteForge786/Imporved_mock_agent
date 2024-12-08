import streamlit as st
import json
from typing import Dict, Any
import pandas as pd
import yaml
import traceback
from queue import Queue
from threading import Event
from sql_agent import SQLReActAgent, UICallback
import logging
import plotly.express as px

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ui.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="SQL Assistant", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .agent-state { 
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    .tool-execution {
        background-color: #e8f0fe;
        border-left: 3px solid #1976d2;
        padding: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 15px;
        margin: 5px 0 5px auto;
        max-width: 80%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-radius: 15px;
        padding: 15px;
        margin: 5px auto 5px 0;
        max-width: 80%;
    }
    .status-indicator {
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 14px;
        font-weight: 500;
    }
    .sql-code {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
    .viz-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class ChatInterface:
    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize the session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "feedback_queue" not in st.session_state:
            st.session_state.feedback_queue = Queue()
        if "feedback_event" not in st.session_state:
            st.session_state.feedback_event = Event()
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "current_state" not in st.session_state:
            st.session_state.current_state = {}
        if "agent" not in st.session_state:
            # Initialize agent with UI callback
            st.session_state.agent = SQLReActAgent(
                schema_path="schema.yaml",
                ui_callback=UICallback(self.update_ui_state),
                feedback_queue=st.session_state.feedback_queue,
                feedback_event=st.session_state.feedback_event
            )

    def update_ui_state(self, state: Dict[str, Any]):
        """Callback to update UI state"""
        st.session_state.current_state = state
        self.display_current_state(state)

    def display_current_state(self, state: Dict[str, Any]):
        """Display the current state of the agent"""
        status = state.get("status")
        
        if status == "starting":
            st.info("🚀 Starting to process your query...")
            
        elif status == "processing":
            current_action = state.get("current_action", {})
            if thought := current_action.get("thought"):
                st.markdown(f"""
                <div class="tool-execution">
                    🤔 <b>Thinking:</b> {thought}
                </div>
                """, unsafe_allow_html=True)
                
            if result := current_action.get("result"):
                self.display_tool_result(current_action["tool"], result)
                
        elif status == "waiting_feedback":
            question = state.get("question")
            st.warning(f"🤔 {question}")
            response = st.text_input("Your response:", key=f"feedback_{len(st.session_state.messages)}")
            if st.button("Submit", key=f"submit_{len(st.session_state.messages)}"):
                st.session_state.feedback_queue.put(response)
                st.session_state.feedback_event.set()

    def display_tool_result(self, tool_name: str, result: Dict[str, Any]):
        """Display results from different tools"""
        if tool_name == "schema_lookup":
            if tables := result.get("relevant_tables"):
                cols = st.columns(len(tables))
                for idx, table in enumerate(tables):
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="agent-state">
                            <h4>{table['name']}</h4>
                            <p><b>Relevance:</b> {table['relevance']}</p>
                            <p><b>Columns:</b> {', '.join(table['columns'])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
        elif tool_name == "sql_generation":
            if sql := result.get("sql"):
                st.markdown(f"""
                <div class="sql-code">
                    {sql}
                </div>
                """, unsafe_allow_html=True)
                if explanation := result.get("explanation"):
                    st.markdown(f"**Explanation:**\n{explanation}")
                    
        elif tool_name == "db_execution":
            if data := result.get("data"):
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                if metadata := result.get("metadata"):
                    st.markdown(f"""
                    <div class="agent-state">
                        <p><b>Rows:</b> {metadata['row_count']}</p>
                        <p><b>Columns:</b> {', '.join(metadata['columns'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
        elif tool_name == "python_executor":
            if fig_data := result.get("figure"):
                with st.container():
                    st.markdown("""<div class="viz-container">""", unsafe_allow_html=True)
                    st.plotly_chart(px.Figure().from_json(fig_data))
                    st.markdown("</div>", unsafe_allow_html=True)

    def process_message(self, message: str):
        """Process a new message"""
        try:
            st.session_state.processing = True
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": message})
            
            # Process with agent
            result = st.session_state.agent.process_query(message)
            
            # Add response
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result.get("response") if result.get("type") == "conversation" else result
            })
            
        except Exception as e:
            logger.error(f"Error processing message: {traceback.format_exc()}")
            st.error(f"Error processing message: {str(e)}")
        finally:
            st.session_state.processing = False

    def display_chat_history(self):
        """Display chat history"""
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            
            if isinstance(content, str):
                st.markdown(f"""
                <div class="{role}-message">
                    {content}
                </div>
                """, unsafe_allow_html=True)
            elif isinstance(content, dict):
                st.markdown(f"""
                <div class="{role}-message">
                    {content.get('response', json.dumps(content, indent=2))}
                </div>
                """, unsafe_allow_html=True)

    def show_interface(self):
        """Display the main interface"""
        st.title("🤖 SQL Analytics Assistant")
        
        # Main chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            self.display_chat_history()
        
        # Input area
        if not st.session_state.processing:
            user_input = st.chat_input("Ask me anything about your data...")
            if user_input:
                self.process_message(user_input)
                st.experimental_rerun()
        
        # Sidebar with schema info and controls
        with st.sidebar:
            st.subheader("📊 Database Schema")
            try:
                with open("schema.yaml") as f:
                    schema = yaml.safe_load(f)
                    for table in schema["tables"]:
                        st.markdown(f"**{table}**")
            except Exception as e:
                st.error("Failed to load schema")
            
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.experimental_rerun()

def main():
    try:
        chat_interface = ChatInterface()
        chat_interface.show_interface()
    except Exception as e:
        logger.error(f"Application error: {traceback.format_exc()}")
        st.error("An error occurred. Please refresh the page.")

if __name__ == "__main__":
    main()
