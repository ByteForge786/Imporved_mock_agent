import yaml
import json
import logging
from typing import Dict, Any, List, Optional, Type, Callable
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import traceback
from datetime import datetime
from queue import Queue
from threading import Event
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('agent.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class UICallback:
    def __init__(self, callback_fn: Callable[[Dict[str, Any]], None]):
        self.callback_fn = callback_fn
        
    def update_ui(self, state: Dict[str, Any]):
        if self.callback_fn:
            self.callback_fn(state)

def validate_input(func):
    def wrapper(self, **kwargs):
        hints = func.__annotations__
        hints.pop('return', None)
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
        super().__init__(name="conversation", description="Handle conversations and analyze queries")
        with open(schema_path) as f:
            self.schema = yaml.safe_load(f)
        self.tables_context = self._build_tables_context()
        
    def _build_tables_context(self) -> str:
        context = []
        for table_name, info in self.schema['tables'].items():
            context.append(f"Table '{table_name}': {info['description']}")
            if 'columns' in info:
                context.append(f"Columns: {', '.join(info['columns'])}")
        return "\n".join(context)
    
    @validate_input
    def execute(self, query: str, previous_context: Dict = None) -> Dict[str, Any]:
        try:
            prompt = f"""
            Analyze this query:
            User query: {query}
            
            Context:
            {self.tables_context}
            
            Previous state: {json.dumps({'last_tool': previous_context.get('last_tool'), 'last_status': previous_context.get('last_status')} if previous_context else {})}
            
            Return ONLY a JSON object in this exact format:
            {{
                "needs_clarification": boolean,
                "clarification_question": "if needs_clarification is true, provide ONE specific question",
                "is_conversation": boolean,
                "intent": "greeting/help/data_query/other",
                "reasoning": "explain classification",
                "response": "if conversation, provide response"
            }}
            """
            
            result = get_llm_response(prompt)
            
            if result["needs_clarification"]:
                return {
                    "status": "needs_clarification",
                    "question": result["clarification_question"],
                    "previous_context": previous_context or {},
                    "intent": result["intent"]
                }
            
            if result["is_conversation"]:
                return {
                    "status": "success",
                    "type": "conversation",
                    "intent": result["intent"],
                    "response": result["response"]
                }
            
            return {
                "status": "not_conversation",
                "intent": result["intent"],
                "reasoning": result["reasoning"]
            }
            
        except Exception as e:
            logger.error(f"Conversation error: {traceback.format_exc()}")
            return {"status": "error", "error": str(e)}

class SchemaLookupTool(Tool):
    def __init__(self, schema_path: str):
        super().__init__(name="schema_lookup", description="Find relevant tables and relationships")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        with open(schema_path) as f:
            self.schema_config = yaml.safe_load(f)
        self._init_embeddings()
    
    def _init_embeddings(self):
        self.table_embeddings = {}
        for table_name, info in self.schema_config['tables'].items():
            table_text = f"""
            Table: {table_name}
            Description: {info['description']}
            Columns: {', '.join(self._extract_columns(info['create_statement']))}
            """
            self.table_embeddings[table_name] = {
                'embedding': self.embed_model.encode(table_text),
                'info': info
            }
    
    def _extract_columns(self, create_statement: str) -> List[str]:
        columns = []
        for line in create_statement.strip().split('\n'):
            if any(keyword in line.upper() for keyword in ['CREATE', 'PRIMARY', 'FOREIGN']):
                continue
            if ',' in line:
                line = line.rstrip(',')
            parts = line.split()
            if len(parts) >= 2:
                columns.append(f"{parts[0]} ({parts[1]})")
        return columns

    @validate_input
    def execute(self, query: str) -> Dict[str, Any]:
        try:
            query_embedding = self.embed_model.encode(query)
            similarities = {
                name: float(np.dot(query_embedding, data['embedding']))
                for name, data in self.table_embeddings.items()
            }
            
            primary_tables = [t for t, s in similarities.items() if s > 0.6]
            secondary_tables = [t for t, s in similarities.items() if 0.4 < s <= 0.6]
            
            relevant_tables = []
            for table in primary_tables + secondary_tables:
                info = self.schema_config['tables'][table]
                relevant_tables.append({
                    "name": table,
                    "description": info['description'],
                    "create_statement": info['create_statement'],
                    "columns": self._extract_columns(info['create_statement']),
                    "relevance": "primary" if table in primary_tables else "secondary"
                })
            
            return {
                "status": "success",
                "relevant_tables": relevant_tables,
                "schema_context": "\n".join(t["create_statement"] for t in relevant_tables)
            }
            
        except Exception as e:
            logger.error(f"Schema lookup error: {traceback.format_exc()}")
            return {"status": "error", "error": str(e)}

class UserFeedbackTool(Tool):
    def __init__(self, feedback_queue: Queue, feedback_event: Event):
        super().__init__(name="user_feedback", description="Get user feedback")
        self.feedback_queue = feedback_queue
        self.feedback_event = feedback_event
    
    @validate_input
    def execute(self, question: str) -> Dict[str, Any]:
        try:
            while not self.feedback_queue.empty():
                self.feedback_queue.get()
            self.feedback_event.clear()
            
            return {
                "status": "waiting_feedback",
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Feedback error: {traceback.format_exc()}")
            return {"status": "error", "error": str(e)}

class PythonCodeGeneratorTool(Tool):
    def __init__(self):
        super().__init__(name="python_generator", description="Generate analysis code")
    
    @validate_input
    def execute(self, question: str, df_metadata: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = f"""
            Generate Python code for: {question}
            DataFrame info: {json.dumps(df_metadata)}
            
            Return ONLY a JSON object in this exact format:
            {{
                "code": "complete python code using pandas, numpy, and plotly",
                "explanation": "brief explanation of what code does",
                "expected_output": "description of expected output (plot/analysis)"
            }}
            """
            
            return {
                "status": "success",
                **get_llm_response(prompt)
            }
            
        except Exception as e:
            logger.error(f"Code generation error: {traceback.format_exc()}")
            return {"status": "error", "error": str(e)}

class PythonExecutorTool(Tool):
    def __init__(self):
        super().__init__(name="python_executor", description="Execute analysis code")
    
    @validate_input
    def execute(self, code: str, df_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            df = pd.DataFrame(df_data)
            namespace = {'pd': pd, 'np': np, 'px': px, 'go': go, 'df': df}
            exec(code, namespace)
            
            return {
                "status": "success",
                "figure": namespace.get('fig').to_json() if 'fig' in namespace else None,
                "results": namespace.get('results')
            }
            
        except Exception as e:
            logger.error(f"Code execution error: {traceback.format_exc()}")
            return {"status": "error", "error": str(e)}

class SQLGenerationTool(Tool):
    def __init__(self):
        super().__init__(name="sql_generation", description="Generate SQL queries")
    
    @validate_input
    def execute(self, question: str, schema_context: str) -> Dict[str, Any]:
        try:
            prompt = f"""
            Generate SQL query for: {question}
            Schema:
            {schema_context}
            
            Return ONLY a JSON object in this exact format:
            {{
                "sql": "complete SQL query",
                "explanation": "step-by-step explanation of query logic",
                "tables_used": ["list of tables used"],
                "columns_used": ["list of columns used"]
            }}
            """
            
            return {
                "status": "success",
                **get_llm_response(prompt)
            }
            
        except Exception as e:
            logger.error(f"SQL generation error: {traceback.format_exc()}")
            return {"status": "error", "error": str(e)}

class SQLValidationTool(Tool):
    def __init__(self):
        super().__init__(name="sql_validation", description="Validate SQL queries")
    
    @validate_input
    def execute(self, sql: str, schema_context: str) -> Dict[str, Any]:
        try:
            prompt = f"""
            Validate SQL query:
            {sql}
            
            Schema:
            {schema_context}
            
            Return ONLY a JSON object in this exact format:
            {{
                "is_safe": boolean,
                "issues": ["list of specific issues found"],
                "feedback": "detailed suggestions for improvement"
            }}
            """
            
            return {
                "status": "success",
                **get_llm_response(prompt)
            }
            
        except Exception as e:
            logger.error(f"SQL validation error: {traceback.format_exc()}")
            return {"status": "error", "error": str(e)}

class MockDBTool(Tool):
    def __init__(self):
        super().__init__(name="db_execution", description="Execute SQL queries")
    
    @validate_input
    def execute(self, sql: str) -> Dict[str, Any]:
        try:
            df = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=100),
                'sales': np.random.uniform(1000, 5000, 100),
                'product': np.random.choice(['A', 'B', 'C'], 100),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
            })
            
            return {
                "status": "success",
                "data": df.to_dict('records'),
                "metadata": {
                    "columns": list(df.columns),
                    "row_count": len(df)
                }
            }
            
        except Exception as e:
            logger.error(f"DB execution error: {traceback.format_exc()}")
            return {"status": "error", "error": str(e)}

class SQLReActAgent:
    def __init__(self, schema_path: str, ui_callback: Optional[UICallback] = None,
                 feedback_queue: Optional[Queue] = None, feedback_event: Optional[Event] = None):
        self.schema_path = schema_path
        self.ui_callback = ui_callback
        self.feedback_queue = feedback_queue or Queue()
        self.feedback_event = feedback_event or Event()
        
        self.tools = {
            "conversation": ConversationTool(schema_path),
            "schema_lookup": SchemaLookupTool(schema_path),
            "sql_generation": SQLGenerationTool(),
            "sql_validation": SQLValidationTool(),
            "db_execution": MockDBTool(),
            "python_generator": PythonCodeGeneratorTool(),
            "python_executor": PythonExecutorTool(),
            "user_feedback": UserFeedbackTool(self.feedback_queue, self.feedback_event)
        }
        
        self.current_context = {}
        self.executed_actions = []
    
    def update_ui_state(self, state_update: Dict[str, Any]):
        if self.ui_callback:
            self.ui_callback.update_ui({
                "context": self.current_context,
                "actions": self.executed_actions,
                **state_update
            })

    def get_next_action(self, query: str) -> Dict[str, Any]:
        prompt = f"""
        Determine next action for: {query}
        Current state: {{'tool': self.current_context.get('last_tool'), 'status': self.current_context.get('last_status')}}
        
        Return ONLY a JSON object in this exact format:
        {{
            "thought": "reasoning for next action",
            "action": "tool name to use",
            "action_input": {{
                "required parameters for the tool"
            }},
            "should_continue": boolean
        }}
        """
        return get_llm_response(prompt)

    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            self.cleanup_state()
            self.update_ui_state({"status": "starting", "query": query})
            
            while True:
                action_plan = self.get_next_action(query)
                self.update_ui_state({
                    "status": "processing",
                    "current_action": action_plan
                })
                
                if not action_plan["should_continue"]:
                    break
                
                tool_name = action_plan["action"]
                if tool_name not in self.tools:
                    raise Exception(f"Unknown tool: {tool_name}")
                
                tool = self.tools[tool_name]
                result = tool.execute(**action_plan["action_input"])
                
                if result["status"] == "waiting_feedback":
                    self.update_ui_state({
                        "status": "waiting_feedback",
                        "question": result["question"]
                    })
                    self.feedback_event.wait()
                    result = {
                        "status": "success",
                        "feedback": self.feedback_queue.get()
                    }
                
                self.executed_actions.append({
                    "tool": tool_name,
                    "thought": action_plan["thought"],
                    "result": result
                })
                
                if result["status"] == "success":
                    self.current_context.update({
                        "last_tool": tool_name,
                        "last_status": "success",
                        **result
                    })
                    
                    # Handle visualization after DB execution
                    if tool_name == "db_execution":
                        feedback_result = self.tools["user_feedback"].execute(
                            question="Would you like to see any specific visualization or analysis of this data?"
                        )
                        if feedback_result["status"] == "success" and feedback_result["feedback"]:
                            code_result = self.tools["python_generator"].execute(
                                question=feedback_result["feedback"],
                                df_metadata=result["metadata"]
                            )
                            if code_result["status"] == "success":
                                viz_result = self.tools["python_executor"].execute(
                                    code=code_result["code"],
                                    df_data=result["data"]
                                )
                                self.current_context["visualization"] = viz_result
                else:
                    raise Exception(f"Tool failed: {result.get('error')}")
            
            final_response = {
                "status": "success",
                "context": self.current_context,
                "actions": self.executed_actions,
                "query": query
            }
            
            self.update_ui_state({
                "status": "completed",
                "result": final_response
            })
            
            return final_response
            
        except Exception as e:
            logger.error(f"Query processing error: {traceback.format_exc()}")
            error_response = {
                "status": "error",
                "error": str(e),
                "query": query
            }
            self.update_ui_state({
                "status": "error",
                "error": error_response
            })
            return error_response
        finally:
            self.cleanup_state()

    def cleanup_state(self):
        """Reset agent state"""
        self.current_context = {}
        self.executed_actions = []
        logger.info("Agent state cleaned up")

def get_llm_response(prompt: str) -> Dict[str, Any]:
    """
    Gets response from LLM model.
    Replace with your actual LLM API call.
    """
    pass

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
