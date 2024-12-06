import yaml
import json
import logging
from typing import Dict, Any, List, Optional, Type
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
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_input(func):
    """Decorator to validate tool inputs"""
    def wrapper(self, **kwargs):
        # Get function's type hints
        hints = func.__annotations__
        
        # Remove return hint if present
        hints.pop('return', None)
        
        # Validate each parameter
        for param, expected_type in hints.items():
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
                
            value = kwargs[param]
            if not isinstance(value, eval(str(expected_type).replace('typing.', ''))):
                raise TypeError(f"Parameter {param} must be of type {expected_type}, got {type(value)}")
                
        return func(self, **kwargs)
    return wrapper

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
        
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.intent_embeddings = self._init_intent_embeddings()
        logger.info("Conversation tool initialized successfully")
    
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
    
    @validate_input
    def execute(self, query: str) -> Dict[str, Any]:
        try:
            intent = self._get_intent(query.lower())
            
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
                    + "\n".join([f"📊 {info['name']}: {info['description']}" for info in tables_info])
                    + "\n\nYou can ask me questions like:\n"
                    + "\n".join([f"- {q}" for table in tables_info for q in table['sample_questions'][:2]])
                )
            elif intent == "tables":
                response = "Here are the available tables and their purposes:\n\n" + \
                          "\n".join([f"📊 {info['name']}: {info['description']}" for info in tables_info])
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
        related = []
        create_stmt = self.schema_config['tables'][table_name]['create_statement']
        for other_table in self.schema_config['tables']:
            if other_table != table_name:
                if f"REFERENCES {other_table}" in create_stmt or \
                   f"REFERENCES {table_name}" in self.schema_config['tables'][other_table]['create_statement']:
                    related.append(other_table)
        return related

    def _check_schema_sufficiency(self, query: str, initial_tables: List[Dict]) -> Dict[str, Any]:
        if not initial_tables:
            return {"is_sufficient": False, "missing_context": "No relevant tables found initially"}
        
        schema_context = "\n".join(t["create_statement"] for t in initial_tables)
        
        prompt = f"""
        Analyze if these tables are sufficient to answer the query.
        
        Query: {query}
        
        Available Tables Schema:
        {schema_context}
        
        Return in this exact format:
        {{
            "is_sufficient": boolean,
            "missing_context": "if not sufficient, describe what kind of tables/data would be needed"
        }}
        """
        return get_llm_response(prompt)

    @validate_input
    def execute(self, query: str) -> Dict[str, Any]:
        try:
            # Initial schema lookup
            query_embedding = self.embed_model.encode(query)
            table_similarities = {}
            
            for table_name, table_data in self.table_embeddings.items():
                similarity = float(np.dot(query_embedding, table_data['embedding']))
                table_similarities[table_name] = similarity
            
            # Initial categorization
            primary_threshold = 0.6
            secondary_threshold = 0.4
            
            primary_tables = [table for table, sim in table_similarities.items() 
                            if sim > primary_threshold]
            secondary_tables = [table for table, sim in table_similarities.items() 
                             if primary_threshold > sim > secondary_threshold]
            
            # Build initial table list
            initial_tables = []
            for table_name in primary_tables + secondary_tables:
                table_info = self.schema_config['tables'][table_name]
                initial_tables.append({
                    "name": table_name,
                    "description": table_info['description'],
                    "create_statement": table_info['create_statement'],
                    "columns": self._extract_columns(table_info['create_statement']),
                    "relevance": "primary" if table_name in primary_tables else "secondary",
                    "similarity_score": table_similarities[table_name]
                })
# Check sufficiency
            sufficiency_check = self._check_schema_sufficiency(query, initial_tables)
            
            if not sufficiency_check["is_sufficient"]:
                # Enhance query with missing context
                enhanced_query = f"{query} {sufficiency_check['missing_context']}"
                enhanced_embedding = self.embed_model.encode(enhanced_query)
                
                # Recompute similarities
                table_similarities = {
                    table_name: float(np.dot(enhanced_embedding, table_data['embedding']))
                    for table_name, table_data in self.table_embeddings.items()
                }
                
                # Recategorize tables
                primary_tables = [table for table, sim in table_similarities.items() 
                                if sim > primary_threshold]
                secondary_tables = [table for table, sim in table_similarities.items() 
                                 if primary_threshold > sim > secondary_threshold]

            # Add related tables through foreign keys
            for primary_table in primary_tables[:]:
                for related_table in self._get_related_tables(primary_table):
                    if related_table not in primary_tables + secondary_tables:
                        secondary_tables.append(related_table)
            
            # Build final table list
            relevant_tables = []
            seen_tables = set()
            for table_name in primary_tables + secondary_tables:
                if table_name not in seen_tables:
                    table_info = self.schema_config['tables'][table_name]
                    relevant_tables.append({
                        "name": table_name,
                        "description": table_info['description'],
                        "create_statement": table_info['create_statement'],
                        "columns": self._extract_columns(table_info['create_statement']),
                        "relevance": "primary" if table_name in primary_tables else "secondary",
                        "similarity_score": table_similarities[table_name]
                    })
                    seen_tables.add(table_name)
            
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
    
    @validate_input
    def execute(self, question: str, schema_context: str, relevant_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            prompt = f"""
            Generate a SQL query for this question: {question}
            
            Available Schema:
            {schema_context}
            
            Return in this exact format:
            {{
                "sql": "the complete SQL query",
                "explanation": "step-by-step explanation of the query logic",
                "tables_used": ["list of tables used in query"],
                "columns_used": ["list of columns used in query"]
            }}
            """
            
            result = get_llm_response(prompt)
            return {
                "status": "success",
                **result
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
    
    @validate_input
    def execute(self, sql: str, schema_context: str, tables_used: List[str]) -> Dict[str, Any]:
        try:
            prompt = f"""
            Validate this SQL query:
            {sql}

            Against schema:
            {schema_context}

            Check for:
            1. SQL injection risks
            2. Proper column references
            3. Join conditions
            4. Where clause correctness
            5. Group by completeness
            6. Overall syntax

            Return in this exact format:
            {{
                "is_safe": boolean,
                "issues": ["list of specific issues found"],
                "feedback": "detailed suggestions for improvement"
            }}
            """
            
            result = get_llm_response(prompt)
            return {
                "status": "success",
                **result
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
    
    @validate_input
    def execute(self, sql: str, tables_used: List[str]) -> Dict[str, Any]:
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
                    "execution_time": "0.1s"
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
        self.current_context = {}
        self.executed_actions = []
        logger.info("SQL ReAct Agent initialized successfully")

    def get_next_action(self, query: str) -> Dict[str, Any]:
        prompt = f"""
        Determine the next action for this query: {query}

        Current Context: {json.dumps(self.current_context, indent=2)}
        
        Previous Actions: {json.dumps(self.executed_actions, indent=2)}

        Available Tools and Required Parameters:
        1. conversation: 
           - query (str): user's question
        
        2. schema_lookup:
           - query (str): user's question
        
        3. sql_generation:
           - question (str): user's question
           - schema_context (str): schema CREATE statements
           - relevant_tables (List[Dict]): table information
        
        4. sql_validation:
           - sql (str): generated SQL query
           - schema_context (str): schema CREATE statements
           - tables_used (List[str]): tables in query
        
        5. db_execution:
           - sql (str): validated SQL query
           - tables_used (List[str]): tables in query

        Return in this exact format:
        {{
            "thought": "your reasoning for the next action",
            "action": "tool_name to use",
            "action_input": {{
                "required parameters matching the tool's needs"
            }},
            "should_continue": boolean
        }}
        """
        
        return get_llm_response(prompt)

    def cleanup_state(self):
        """Clean up state after query processing"""
        self.current_context = {}
        self.executed_actions = []
        logger.info("Agent state cleaned up")

    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            # Clean up state from any previous query
            self.cleanup_state()
            
            # First try conversation handling
            conv_result = self.tools["conversation"].execute(query=query)
            if conv_result["status"] == "success" and conv_result["type"] == "conversation":
                return conv_result
            
            # Initialize for ReAct loop
            self.current_context = {"query": query}
            max_iterations = 5
            current_iteration = 0
            
            while current_iteration < max_iterations:
                # Get next action
                action_plan = self.get_next_action(query)
                logger.info(f"Next action plan: {action_plan}")
                
                if not action_plan.get("should_continue", False):
                    break
                    
                tool_name = action_plan.get("action")
                if tool_name not in self.tools:
                    raise Exception(f"Unknown tool: {tool_name}")
                
                # Execute tool
                tool = self.tools[tool_name]
                result = tool.execute(**action_plan.get("action_input", {}))
                
                # Record action
                executed_action = {
                    "tool": tool_name,
                    "thought": action_plan.get("thought"),
                    "result": result
                }
                self.executed_actions.append(executed_action)
                
                # Update context and check for errors
                if result["status"] == "success":
                    self.current_context.update(result)
                else:
                    raise Exception(f"Tool execution failed: {result.get('error', 'Unknown error')}")
                
                current_iteration += 1
            
            # Prepare final response
            final_response = {
                "type": "sql_analysis",
                "status": "success",
                "context": self.current_context,
                "actions": self.executed_actions
            }
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in query processing: {traceback.format_exc()}")
            return {
                "type": "error",
                "status": "error",
                "message": str(e)
            }
        finally:
            # Always cleanup state after processing
            self.cleanup_state()

if __name__ == "__main__":
    # Example usage
    agent = SQLReActAgent("schema.yaml")
    
    # Test queries
    test_queries = [
        "Hello! What can you do?",
        "Show me sales by region",
        "What are the top selling products?"
    ]
    
    for query in test_queries:
        print(f"\nProcessing query: {query}")
        result = agent.process_query(query)
        print(json.dumps(result, indent=2))
