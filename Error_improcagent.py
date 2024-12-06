class ConversationTool(Tool):
    def __init__(self, schema_path: str):
        super().__init__(
            name="conversation",
            description="Handle conversations and determine if they require SQL analysis"
        )
        with open(schema_path) as f:
            self.schema = yaml.safe_load(f)
            
        # We'll now use this for context rather than intent matching
        self.tables_context = self._build_tables_context()
        logger.info("Conversation tool initialized successfully")
    
    def _build_tables_context(self) -> str:
        """Build a context string about available tables and their purposes"""
        context = []
        for table_name, info in self.schema['tables'].items():
            context.append(f"Table '{table_name}': {info['description']}")
            if 'columns' in info:
                context.append(f"Columns: {', '.join(info['columns'])}")
        return "\n".join(context)
    
    @validate_input
    def execute(self, query: str) -> Dict[str, Any]:
        try:
            # Let LLM determine intent and appropriate response
            prompt = f"""
            Analyze this user query and determine if it's a conversation or a data query.
            Available database context:
            {self.tables_context}

            User query: {query}

            Rules:
            1. If the query is a greeting, help request, or general question about capabilities,
               provide a response using the database context
            2. If the query is asking about specific data or analysis, classify it as a data query
            3. Only respond about topics related to the database and its capabilities
            4. Don't make assumptions about data not mentioned in the context

            Return in this exact format:
            {{
                "is_conversation": boolean,
                "reasoning": "explain why this is conversation or data query",
                "response": "if conversation, provide the response using context. if data query, leave empty"
            }}
            """
            
            result = get_llm_response(prompt)
            
            if result["is_conversation"]:
                return {
                    "status": "success",
                    "type": "conversation",
                    "response": result["response"]
                }
            else:
                return {
                    "status": "not_conversation",
                    "message": "This appears to be a data query.",
                    "reasoning": result["reasoning"]
                }
            
        except Exception as e:
            logger.error(f"Conversation tool error: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": f"Conversation handling failed: {str(e)}"
            }

class SQLReActAgent:
    # ... (previous init code remains the same)

    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            self.cleanup_state()
            
            # First, let's try conversation handling
            conv_result = self.tools["conversation"].execute(query=query)
            
            # If it's a conversation, return the response
            if conv_result["status"] == "success" and conv_result["type"] == "conversation":
                return conv_result
                
            # If it's not a conversation, start the ReAct loop for SQL analysis
            self.current_context = {
                "query": query,
                "conversation_reasoning": conv_result.get("reasoning", "")  # Store why it was classified as a data query
            }
            
            # Rest of the ReAct loop remains the same...
