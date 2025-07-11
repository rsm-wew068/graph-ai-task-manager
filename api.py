from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import os
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Neo4j connection setup
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://host.docker.internal:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

class ChatTurnCreate(BaseModel):
    conversation_id: str
    user_message: str
    assistant_message: str
    state: Optional[dict] = None

class ChatTurnRead(ChatTurnCreate):
    id: int
    timestamp: datetime

app = FastAPI()

@app.post("/chat_turns/", response_model=ChatTurnRead)
async def create_chat_turn(turn: ChatTurnCreate):
    with driver.session() as session:
        result = session.run(
            """
            CREATE (ct:ChatTurn {
                conversation_id: $conversation_id,
                user_message: $user_message,
                assistant_message: $assistant_message,
                state: $state,
                timestamp: datetime()
            })
            RETURN id(ct) AS id, ct
            """,
            conversation_id=turn.conversation_id,
            user_message=turn.user_message,
            assistant_message=turn.assistant_message,
            state=turn.state
        )
        record = result.single()
        ct = record["ct"]
        return {
            "id": record["id"],
            "conversation_id": ct["conversation_id"],
            "user_message": ct["user_message"],
            "assistant_message": ct["assistant_message"],
            "state": ct.get("state"),
            "timestamp": ct["timestamp"]
        }

@app.get("/chat_turns/{conversation_id}", response_model=List[ChatTurnRead])
async def get_chat_turns(conversation_id: str):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (ct:ChatTurn {conversation_id: $conversation_id})
            RETURN id(ct) AS id, ct
            ORDER BY ct.timestamp
            """,
            conversation_id=conversation_id
        )
        turns = []
        for record in result:
            ct = record["ct"]
            turns.append({
                "id": record["id"],
                "conversation_id": ct["conversation_id"],
                "user_message": ct["user_message"],
                "assistant_message": ct["assistant_message"],
                "state": ct.get("state"),
                "timestamp": ct["timestamp"]
            })
        return turns

@app.get("/")
async def root():
    return {"message": "FastAPI backend (Neo4j) is running!"}
