import yaml
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sql_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Tool:
    name: str
    description: str
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class ConversationTool(Tool):
    def __init__(self, schema_path: str):
        super().__init__(
            name="conversation",
            description="Handle greetings, help requests, and general queries about capabilities"
        )
        with open(schema_path) as f:
            self.schema = yaml.safe_load(f)
            
        # Get embeddings for conversation intents
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.intent_embeddings = self._init_intent_embeddings()
    
    def _init_intent_embeddings(self) -> Dict[str, np.ndarray]:
        intents = {
            "greeting": "Hello hi hey greetings good morning good afternoon good evening",
            "help": "help what can you do show capabilities abilities",
            "tables": "what tables data available show schema database structure",
            "examples": "give examples sample queries what can i ask",
        }
        return {intent: self.embed_model.encode(text) for intent, text in intents.items()}
    
    def _get_intent(self, query: str) -> str:
        query_embedding = self.embed_model.encode(query)
        similarities = {
            intent: float(np.dot(query_embedding, embedding))
            for intent, embedding in self.intent_embeddings.items()
        }
        return max(similarities.items(), key=lambda x: x[1])[0]
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute conversation handling with required query parameter"""
        try:
            intent = self._get_intent(query.lower())
            
            # Prepare table information for all responses
            tables_info = [
                {
                    "name": table_name,
                    "description": info['description'],
                    "sample_questions": info.get('sample_questions', [])
                }
                for table_name, info in self.schema['tables'].items()
            ]
            
            if intent == "greeting":
                response = (
                    f"Hello! I'm your SQL Assistant. I can help you analyze data from our "
                    f"database which includes {', '.join(t['name'] for t in tables_info)}. "
                    f"How can I help you today?"
                )
            elif intent == "help":
                response = (
                    "I can help you analyze data by generating and executing SQL queries. Here's what I can work with:\n\n"
                    + "\n".join([f"📊 {info['name'].title()}: {info['description']}" for info in tables_info])
                    + "\n\nYou can ask me questions like:\n"
                    + "\n".join([f"- {q}" for table in tables_info for q in table['sample_questions'][:2]])
                    + "\n\nFeel free to ask about specific data points or trends!"
                )
            elif intent == "tables":
                response = "Here are the available tables and their purposes:\n\n" + \
                          "\n".join([f"📊 {info['name'].title()}: {info['description']}" for info in tables_info])
            elif intent == "examples":
                response = "Here are some example questions you can ask:\n\n" + \
                          "\n".join([f"- {q}" for table in tables_info for q in table['sample_questions']])
            else:
                return {
                    "status": "not_conversation",
                    "message": "This appears to be a data query."
                }
                
            return {
                "status": "success",
                "type": "conversation",
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Conversation tool error: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": f"Conversation handling failed: {str(e)}"
            }

class SchemaLookupTool(Tool):
    def __init__(self, schema_path: str):
        super().__init__(
            name="schema_lookup",
            description="Find relevant database tables and their relationships based on the query"
        )
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        with open(schema_path) as f:
            self.schema_config = yaml.safe_load(f)
        self._init_embeddings()
        logger.info("Schema lookup tool initialized successfully")
        
    def _init_embeddings(self):
        """Initialize embeddings for all tables"""
        self.table_embeddings = {}
        for table_name, info in self.schema_config['tables'].items():
            table_text = f"""
            Table: {table_name}
            Description: {info['description']}
            Columns: {', '.join(self._extract_columns(info['create_statement']))}
            Sample Questions: {' '.join(info.get('sample_questions', []))}
            """
            self.table_embeddings[table_name] = {
                'embedding': self.embed_model.encode(table_text),
                'info': info
            }
            
    def _extract_columns(self, create_statement: str) -> List[str]:
        """Extract column definitions from CREATE statement"""
        lines = [line.strip() for line in create_statement.split('\n') 
                if line.strip() and not line.strip().startswith(('CREATE', 'PRIMARY', 'FOREIGN'))]
        columns = []
        for line in lines:
            if ',' in line:
                line = line.rstrip(',')
            parts = line.split()
            if len(parts) >= 2:
                columns.append(f"{parts[0]} ({parts[1]})")
        return columns

    def _get_related_tables(self, table_name: str) -> List[str]:
        """Find tables related through foreign keys"""
        related = []
        create_stmt = self.schema_config['tables'][table_name]['create_statement']
        for other_table in self.schema_config['tables']:
            if other_table != table_name:
                if f"REFERENCES {other_table}" in create_stmt or \
                   f"REFERENCES {table_name}" in self.schema_config['tables'][other_table]['create_statement']:
                    related.append(other_table)
        return related

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute schema lookup with required query parameter"""
        try:
            # Get query embedding and find similar tables
            query_embedding = self.embed_model.encode(query)
            table_similarities = {}
            
            for table_name, table_data in self.table_embeddings.items():
                similarity = float(np.dot(query_embedding, table_data['embedding']))
                table_similarities[table_name] = similarity
            
            # Categorize tables by relevance
            primary_threshold = 0.6
            secondary_threshold = 0.4
            
            primary_tables = [table for table, sim in table_similarities.items() 
                            if sim > primary_threshold]
            secondary_tables = [table for table, sim in table_similarities.items() 
                             if primary_threshold > sim > secondary_threshold]
            
            # Add related tables through foreign keys
            for primary_table in primary_tables[:]:
                for related_table in self._get_related_tables(primary_table):
                    if related_table not in primary_tables + secondary_tables:
                        secondary_tables.append(related_table)
            
            # Build final table list with details
            relevant_tables = []
            for table_name in primary_tables + secondary_tables:
                table_info = self.schema_config['tables'][table_name]
                relevant_tables.append({
                    "name": table_name,
                    "description": table_info['description'],
                    "create_statement": table_info['create_statement'],
                    "columns": self._extract_columns(table_info['create_statement']),
                    "relevance": "primary" if table_name in primary_tables else "secondary",
                    "similarity_score": table_similarities[table_name]
                })
            
            logger.info(f"Schema lookup completed. Primary tables: {primary_tables}, "
                       f"Secondary tables: {secondary_tables}")
            
            return {
                "status": "success",
                "relevant_tables": relevant_tables,
                "schema_context": "\n".join(t["create_statement"] for t in relevant_tables),
                "table_relevance": {
                    "primary": primary_tables,
                    "secondary": secondary_tables
                }
            }
            
        except Exception as e:
            logger.error(f"Schema lookup error: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": f"Schema lookup failed: {str(e)}"
            }

class SQLGenerationTool(Tool):
    def __init__(self):
        super().__init__(
            name="sql_generation",
            description="Generate SQL query based on the question and schema"
        )
    
    def execute(self, question: str, schema_context: str, relevant_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute SQL generation with required parameters"""
        try:
            # For demonstration, generating a mock SQL query
            primary_table = next((t for t in relevant_tables if t['relevance'] == 'primary'), None)
            if not primary_table:
                raise ValueError("No primary table identified")
            
            # Generate a simple SELECT query
            query = f"""SELECT * FROM {primary_table['name']} LIMIT 10;"""
            
            logger.info(f"Generated SQL query: {query}")
            
            return {
                "status": "success",
                "sql": query,
                "explanation": f"Generated a simple query to select from {primary_table['name']}",
                "tables_used": [primary_table['name']],
                "columns_used": ["*"]
            }
            
        except Exception as e:
            logger.error(f"SQL generation error: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": f"SQL generation failed: {str(e)}"
            }

class SQLValidationTool(Tool):
    def __init__(self):
        super().__init__(
            name="sql_validation",
            description="Validate SQL query for safety and correctness"
        )
    
    def execute(self, sql: str, schema_context: str, tables_used: List[str]) -> Dict[str, Any]:
        """Execute SQL validation with required parameters"""
        try:
            # Basic validation checks
            sql_lower = sql.lower()
            
            # Check for dangerous operations
            dangerous_keywords = ['drop', 'truncate', 'delete', 'update', 'insert']
            issues = []
            
            for keyword in dangerous_keywords:
                if keyword in sql_lower:
                    issues.append(f"Query contains potentially dangerous operation: {keyword}")
            
            # Check if all referenced tables exist
            for table in tables_used:
                if table not in schema_context:
                    issues.append(f"Referenced table not found in schema: {table}")
            
            is_safe = len(issues) == 0
            
            logger.info(f"SQL validation completed. Safe: {is_safe}, Issues: {issues}")
            
            return {
                "status": "success",
                "is_safe": is_safe,
                "issues": issues,
                "feedback": "Query appears safe" if is_safe else "Query needs revision"
            }
            
        except Exception as e:
            logger.error(f"SQL validation error: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": f"SQL validation failed: {str(e)}"
            }

class MockDBTool(Tool):
    def __init__(self):
        super().__init__(
            name="db_execution",
            description="Execute SQL query and return results"
        )
    
    def execute(self, sql: str, tables_used: List[str]) -> Dict[str, Any]:
        """Execute query with required parameters"""
        try:
            # Generate mock data
            n_rows = 100
            mock_data = {
                'date': pd.date_range('2024-01-01', periods=n_rows),
                'sales': np.random.uniform(1000, 5000, n_rows),
                'product': np.random.choice(['A', 'B', 'C'], n_rows),
                'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
                'quantity': np.random.randint(1, 100, n_rows)
            }
            
            df = pd.DataFrame(mock_data)
            
            logger.info(f"Generated mock data with {len(df)} rows")
            
            return {
                "status": "success",
                "data": df.to_dict('records'),
                "metadata": {
                    "row_count": len(df),
                    "columns": list(df.columns),
                    "tables_used": tables_used,
                    "execution_time": "0.1s"  # Mock execution time
                }
            }
            
        except Exception as e:
            logger.error(f"DB execution error: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": f"Query execution failed: {str(e)}"
            }

class SQLReActAgent:
    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        self.tools = {
            "conversation": ConversationTool(schema_path),
            "schema_lookup": SchemaLookupTool(schema_path),
            "sql_generation": SQLGenerationTool(),
            "sql_validation": SQLValidationTool(),
            "db_execution": MockDBTool()
        }
        self.conversation_history = []
        logger.info("SQL ReAct Agent initialized successfully")
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """Main entry point for processing queries"""
        try:
            # Record query in conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "role": "user",
                "content": query
            })
            
            # First try conversation tool
            conv_result = self.tools["conversation"].execute(query=query)
            if conv_result["status"] == "success" and conv_result["type"] == "conversation":
                self.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "role": "assistant",
                    "content": conv_result["response"]
                })
                return conv_result
            
            # If not conversation, process as data query
            context = {"query": query}
            executed_actions = []
            
            # 1. Schema Lookup
            schema_result = self.tools["schema_lookup"].execute(query=query)
            if schema_result["status"] != "success":
                raise Exception("Schema lookup failed")
            
            executed_actions.append({
                "tool": "schema_lookup",
                "thought": "Identifying relevant tables and their relationships",
                "result": schema_result
})
            context.update(schema_result)
            
            # 2. SQL Generation
            if not context.get("relevant_tables"):
                raise Exception("No relevant tables found for the query")
            
            sql_result = self.tools["sql_generation"].execute(
                question=query,
                schema_context=context["schema_context"],
                relevant_tables=context["relevant_tables"]
            )
            if sql_result["status"] != "success":
                raise Exception("SQL generation failed")
            
            executed_actions.append({
                "tool": "sql_generation",
                "thought": "Generating SQL query based on identified tables",
                "result": sql_result
            })
            context.update(sql_result)
            
            # 3. SQL Validation
            validation_result = self.tools["sql_validation"].execute(
                sql=sql_result["sql"],
                schema_context=context["schema_context"],
                tables_used=sql_result["tables_used"]
            )
            if validation_result["status"] != "success":
                raise Exception("SQL validation failed")
            
            executed_actions.append({
                "tool": "sql_validation",
                "thought": "Validating SQL query for safety and correctness",
                "result": validation_result
            })
            
            if not validation_result["is_safe"]:
                raise Exception("Generated SQL query failed validation")
            
            # 4. Query Execution
            if validation_result["is_safe"]:
                execution_result = self.tools["db_execution"].execute(
                    sql=sql_result["sql"],
                    tables_used=sql_result["tables_used"]
                )
                if execution_result["status"] != "success":
                    raise Exception("Query execution failed")
                
                executed_actions.append({
                    "tool": "db_execution",
                    "thought": "Executing validated SQL query",
                    "result": execution_result
                })
                context.update(execution_result)
            
            # Record response in conversation history
            result = {
                "type": "sql_analysis",
                "status": "success",
                "context": context,
                "actions": executed_actions
            }
            
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "role": "assistant",
                "content": result
            })
            
            return result
            
        except Exception as e:
            error_result = {
                "type": "error",
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "role": "assistant",
                "content": error_result
            })
            
            logger.error(f"Error in main query processing: {traceback.format_exc()}")
            return error_result
    
    def clear_memory(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Return the conversation history"""
        return self.conversation_history

# Example usage and testing
if __name__ == "__main__":
    # Initialize agent
    agent = SQLReActAgent("schema.yaml")
    
    # Test conversation query
    conv_queries = [
        "Hello! How can you help me?",
        "What data do you have access to?",
        "Can you give me some example questions?",
        "Show me the available tables"
    ]
    
    print("\nTesting conversation handling:")
    for query in conv_queries:
        print(f"\nQuery: {query}")
        result = agent.process_query(query)
        print(json.dumps(result, indent=2))
    
    # Test data queries
    data_queries = [
        "Show me total sales by region",
        "What are the top selling products?",
        "List all customers who made purchases last month"
    ]
    
    print("\nTesting data queries:")
    for query in data_queries:
        print(f"\nQuery: {query}")
        result = agent.process_query(query)
        print(json.dumps(result, indent=2))          
