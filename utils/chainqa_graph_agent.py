"""
ChainQA Graph Agent - Multi-step reasoning for Neo4j graph queries
Uses step-by-step discovery and traversal instead of guessing entity types.
"""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
import json
import re
from typing import List, Dict, Any, Generator

load_dotenv()

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://host.docker.internal:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Initialize Neo4j graph
neo4j_graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

# LLM for reasoning
llm = ChatOpenAI(temperature=0, model="gpt-4o")
llm_stream = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)

def extract_keywords(user_query: str) -> str:
    """Extract the main entity/keyword from the user query."""
    # Remove common question words
    question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 'was', 'were', 'will', 'can', 'could', 'should', 'would']
    words = user_query.lower().split()
    keywords = [word for word in words if word not in question_words and len(word) > 2]
    return ' '.join(keywords)

def step1_entity_discovery(user_query: str) -> Dict[str, Any]:
    """Step 1: Discover what type of entity the user is asking about."""
    
    keywords = extract_keywords(user_query)
    
    discovery_prompt = PromptTemplate(
        input_variables=["question", "keywords"],
        template="""
        Analyze this question and generate a Cypher query to discover what entities exist in the database.
        
        Question: {question}
        Keywords: {keywords}
        
        Available node types in the database:
        - Topic: Project topics/categories
        - Task: Individual tasks
        - Person: People (owners, collaborators)
        - Date: Start/due dates
        - Department: Organizational departments
        - Organization: Companies/organizations
        - Role: Job roles/positions
        - Email_Index: Email references
        - Summary: Task summaries
        
        IMPORTANT: Use 'name' property for Date nodes, not 'date' property.
        Date nodes have properties: name, node_type, date_type
        
        Generate a Cypher query that searches across ALL node types to find entities matching the keywords.
        The query should return the node type, name, and any relevant properties.
        
        Return ONLY the Cypher query, nothing else.
        """
    )
    
    # Use new RunnableSequence syntax instead of deprecated LLMChain
    discovery_chain = discovery_prompt | llm
    cypher_query = discovery_chain.invoke({"question": user_query, "keywords": keywords})
    
    # Clean the query (remove markdown if present)
    cypher_query = cypher_query.content.strip()
    if cypher_query.startswith('```cypher'):
        cypher_query = cypher_query[9:]
    if cypher_query.startswith('```'):
        cypher_query = cypher_query[3:]
    if cypher_query.endswith('```'):
        cypher_query = cypher_query[:-3]
    cypher_query = cypher_query.strip()
    
    # Fix deprecated Neo4j syntax for alternative relationship types
    # Replace [:RELATIONSHIP1|:RELATIONSHIP2] with [:RELATIONSHIP1|RELATIONSHIP2]
    cypher_query = re.sub(r'\[:([A-Z_]+)\|:([A-Z_]+)\]', r'[:\1|\2]', cypher_query)
    
    try:
        discovered_entities = neo4j_graph.query(cypher_query)
        return {
            "entities": discovered_entities,
            "query": cypher_query,
            "keywords": keywords
        }
    except Exception as e:
        return {
            "entities": [],
            "query": cypher_query,
            "keywords": keywords,
            "error": str(e)
        }

def step2_relationship_exploration(user_query: str, discovered_entities: List[Dict]) -> Dict[str, Any]:
    """Step 2: Explore relationships based on discovered entities."""
    
    if not discovered_entities:
        return {"related_data": [], "query": None, "reasoning": "No entities found"}
    
    # Build context from discovered entities
    entity_context = []
    for entity in discovered_entities:
        if 'labels' in entity and 'n.name' in entity:
            node_type = entity['labels'][0] if entity['labels'] else 'Unknown'
            node_name = entity['n.name']
            entity_context.append(f"{node_type}: {node_name}")
    
    entity_summary = "\n".join(entity_context)
    
    exploration_prompt = PromptTemplate(
        input_variables=["question", "entities", "entity_summary"],
        template="""
        Based on these discovered entities in the Neo4j database:
        {entity_summary}
        
        For the question: {question}
        
        Generate a Cypher query to explore relationships and find relevant information.
        
        Available relationships:
        - (:Topic)-[:HAS_TASK]->(:Task)
        - (:Task)-[:START_ON]->(:Date)
        - (:Task)-[:DUE_ON]->(:Date)
        - (:Task)-[:RESPONSIBLE_TO]->(:Person)
        - (:Task)-[:COLLABORATED_BY]->(:Person)
        - (:Person)-[:HAS_ROLE]->(:Role)-[:BELONGS_TO]->(:Department)
        - (:Department)-[:IS_IN]->(:Organization)
        
        IMPORTANT: When using multiple relationship types in a single pattern, use the new syntax:
        - Use: (node)-[:RELATIONSHIP1|RELATIONSHIP2]->(other_node)
        - NOT: (node)-[:RELATIONSHIP1|:RELATIONSHIP2]->(other_node)
        - Example: (task)-[:RESPONSIBLE_TO|COLLABORATED_BY]->(person)
        
        IMPORTANT: Use 'name' property for Date nodes, not 'date' property.
        Date nodes have properties: name, node_type, date_type
        
        The query should:
        1. Start from the discovered entities
        2. Follow relevant relationships
        3. Return comprehensive information to answer the question
        
        Return ONLY the Cypher query, nothing else.
        """
    )
    
    # Use new RunnableSequence syntax instead of deprecated LLMChain
    exploration_chain = exploration_prompt | llm
    cypher_query = exploration_chain.invoke({
        "question": user_query,
        "entities": discovered_entities,
        "entity_summary": entity_summary
    })
    
    # Clean the query
    cypher_query = cypher_query.content.strip()
    if cypher_query.startswith('```cypher'):
        cypher_query = cypher_query[9:]
    if cypher_query.startswith('```'):
        cypher_query = cypher_query[3:]
    if cypher_query.endswith('```'):
        cypher_query = cypher_query[:-3]
    cypher_query = cypher_query.strip()
    
    # Fix deprecated Neo4j syntax for alternative relationship types
    # Replace [:RELATIONSHIP1|:RELATIONSHIP2] with [:RELATIONSHIP1|RELATIONSHIP2]
    cypher_query = re.sub(r'\[:([A-Z_]+)\|:([A-Z_]+)\]', r'[:\1|\2]', cypher_query)
    
    try:
        related_data = neo4j_graph.query(cypher_query)
        return {
            "related_data": related_data,
            "query": cypher_query,
            "reasoning": f"Explored relationships from {len(discovered_entities)} discovered entities"
        }
    except Exception as e:
        return {
            "related_data": [],
            "query": cypher_query,
            "reasoning": f"Error exploring relationships: {str(e)}"
        }

def step3_answer_generation(user_query: str, discovery_result: Dict, exploration_result: Dict) -> str:
    """Step 3: Generate natural language answer based on discovered data."""
    
    answer_prompt = PromptTemplate(
        input_variables=["question", "discovery_data", "exploration_data", "reasoning"],
        template="""
        Based on this graph data, provide a natural and helpful answer to the user's question.
        
        Question: {question}
        
        Discovery Step Results:
        - Keywords searched: {discovery_data}
        - Entities found: {discovery_data}
        
        Exploration Step Results:
        - Related data: {exploration_data}
        - Reasoning: {reasoning}
        
        Instructions:
        1. If no relevant data was found, explain that clearly
        2. If data was found, provide a comprehensive but concise answer
        3. Use natural language, not technical jargon
        4. Include relevant dates, people, and relationships
        5. If the data is incomplete, mention what information is missing
        
        Provide a helpful, conversational response.
        """
    )
    
    # Use new RunnableSequence syntax instead of deprecated LLMChain
    answer_chain = answer_prompt | llm
    
    # Format data for the prompt
    discovery_summary = json.dumps(discovery_result.get("entities", []), indent=2)
    exploration_summary = json.dumps(exploration_result.get("related_data", []), indent=2)
    reasoning = exploration_result.get("reasoning", "No reasoning provided")
    
    answer = answer_chain.invoke({
        "question": user_query,
        "discovery_data": discovery_summary,
        "exploration_data": exploration_summary,
        "reasoning": reasoning
    })
    
    return answer.content

def chainqa_graph_search(user_query: str) -> Dict[str, Any]:
    """Main ChainQA function that orchestrates the multi-step reasoning."""
    
    print(f"🔍 ChainQA: Starting analysis for '{user_query}'")
    
    # Step 1: Entity Discovery
    print("📋 Step 1: Discovering entities...")
    discovery_result = step1_entity_discovery(user_query)
    
    if discovery_result.get("error"):
        return {
            "answer": f"I encountered an error during entity discovery: {discovery_result['error']}",
            "steps": {"discovery": discovery_result},
            "success": False
        }
    
    print(f"✅ Found {len(discovery_result['entities'])} entities")
    
    # Step 2: Relationship Exploration
    print("🔗 Step 2: Exploring relationships...")
    exploration_result = step2_relationship_exploration(user_query, discovery_result["entities"])
    
    print(f"✅ Explored relationships: {exploration_result['reasoning']}")
    
    # Step 3: Answer Generation
    print("💬 Step 3: Generating answer...")
    answer = step3_answer_generation(user_query, discovery_result, exploration_result)
    
    print("✅ Answer generated")
    
    return {
        "answer": answer,
        "steps": {
            "discovery": discovery_result,
            "exploration": exploration_result
        },
        "success": True
    }

def stream_chainqa_graph_search(user_query: str) -> Generator[str, None, Dict[str, Any]]:
    """Streaming version of ChainQA for real-time responses."""
    
    # Step 1: Entity Discovery
    yield "🔍 Discovering entities in the database...\n"
    discovery_result = step1_entity_discovery(user_query)
    
    if discovery_result.get("error"):
        error_msg = f"I encountered an error during entity discovery: {discovery_result['error']}"
        yield error_msg
        return {
            "answer": error_msg,
            "steps": {"discovery": discovery_result},
            "success": False
        }
    
    entity_count = len(discovery_result["entities"])
    yield f"✅ Found {entity_count} entities\n"
    
    # Step 2: Relationship Exploration
    yield "🔗 Exploring relationships...\n"
    exploration_result = step2_relationship_exploration(user_query, discovery_result["entities"])
    
    yield f"✅ {exploration_result['reasoning']}\n"
    
    # Step 3: Answer Generation
    yield "💬 Generating answer...\n"
    answer = step3_answer_generation(user_query, discovery_result, exploration_result)
    
    yield f"\n{answer}"
    
    return {
        "answer": answer,
        "steps": {
            "discovery": discovery_result,
            "exploration": exploration_result
        },
        "success": True
    }

def demonstrate_chainqa():
    """Demonstrate ChainQA with example queries."""
    
    test_queries = [
        "Who is Rachel Wang?",
        "When is RAP enrollment?",
        "What projects are in the Engineering department?",
        "What's due this week?",
        "Who is working on the Capstone Project?"
    ]
    
    print("🧪 ChainQA Graph Agent Demonstration\n")
    
    for query in test_queries:
        print(f"❓ Query: {query}")
        print("-" * 50)
        
        result = chainqa_graph_search(query)
        
        if result["success"]:
            print(f"✅ Answer: {result['answer']}")
            
            # Show discovery details
            discovery = result["steps"]["discovery"]
            if discovery["entities"]:
                print(f"📋 Discovered entities: {len(discovery['entities'])}")
                for entity in discovery["entities"][:3]:  # Show first 3
                    if 'labels' in entity and 'n.name' in entity:
                        node_type = entity['labels'][0] if entity['labels'] else 'Unknown'
                        print(f"   - {node_type}: {entity['n.name']}")
        else:
            print(f"❌ Error: {result['answer']}")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    demonstrate_chainqa() 