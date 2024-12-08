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
