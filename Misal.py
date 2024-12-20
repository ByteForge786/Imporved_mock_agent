def handle_user_input(self, user_input: str):
    try:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Create placeholders for real-time updates
        status_placeholder = st.empty()
        result_placeholder = st.empty()
        
        # Process query with real-time updates
        st.session_state.processing = True
        result = st.session_state.agent.process_query(
            user_input
            # Removed callback parameter since we pass ui_callback during initialization
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


class ChatUI:
    def __init__(self):
        self.feedback_queue = Queue()
        self.feedback_event = Event()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state with error handling"""
        try:
            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "agent" not in st.session_state:
                # Create a standalone callback function
                def ui_callback(state: Dict[str, Any]):
                    status = state.get("status")
                    
                    if status == "waiting_feedback":
                        # Show clarification request or feedback request
                        question = state.get("question")
                        if question:
                            st.info(question)
                            response = st.text_input(
                                "Your response:",
                                key=f"feedback_{len(st.session_state.messages)}"
                            )
                            if st.button("Submit", key=f"submit_{len(st.session_state.messages)}"):
                                self.feedback_queue.put(response)
                                self.feedback_event.set()
                    
                    elif status == "processing":
                        # Show current action
                        current_action = state.get("current_action", {})
                        if thought := current_action.get("thought"):
                            st.markdown(f"""<div class="processing-status">
                                🔄 {thought}</div>""", unsafe_allow_html=True)
                        
                        # Show partial results if available
                        if result := current_action.get("result"):
                            self.display_analysis_result(
                                {"status": "success", "actions": [current_action]},
                                realtime=True
                            )

                # Create UICallback instance with the standalone function
                callback = UICallback(ui_callback)
                st.session_state.agent = SQLReActAgent(
                    "schema.yaml",
                    ui_callback=callback,
                    feedback_queue=self.feedback_queue,
                    feedback_event=self.feedback_event
                )
            if "processing" not in st.session_state:
                st.session_state.processing = False
                
        except Exception as e:
            logger.error(f"Session state initialization error: {traceback.format_exc()}")
            raise InitializationError("Failed to initialize chat session")


class ChatInterface:
    def __init__(self):
        self.initialize_session_state()
        
    def process_message(self, message: str):
        """Process a new message"""
        try:
            st.session_state.processing = True
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": message})

            # First check if this is a conversational query
            conv_result = st.session_state.agent.tools["conversation"].execute(query=message)
            
            if conv_result["status"] == "success" and conv_result.get("type") == "conversation":
                # It's a conversational response (like greeting)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": conv_result["response"]
                })
                return
            
            # If not conversation, process with full agent pipeline
            result = st.session_state.agent.process_query(message)
            
            if result["status"] == "success":
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result
                })
            elif result["status"] == "error":
                st.error(result.get("error", "An error occurred"))
            
        except Exception as e:
            logger.error(f"Error processing message: {traceback.format_exc()}")
            st.error(f"Error processing message: {str(e)}")
        finally:
            st.session_state.processing = False

    def display_current_state(self, state: Dict[str, Any]):
        """Display the current state of the agent"""
        if not isinstance(state, dict):
            return
            
        status = state.get("status", "")
        
        if status == "starting":
            st.info("🚀 Starting to process your query...")
            
        elif status == "processing":
            current_action = state.get("current_action", {})
            if isinstance(current_action, dict):
                thought = current_action.get("thought", "")
                if thought:
                    st.markdown(f"""
                    <div class="tool-execution">
                        🤔 <b>Thinking:</b> {thought}
                    </div>
                    """, unsafe_allow_html=True)
