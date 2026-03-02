import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import tiktoken
from neo4j import AsyncDriver, RoutingControl
from pydantic import BaseModel, Field


# Set up logging
logger = logging.getLogger('mcp_neo4j_memory')
logger.setLevel(logging.INFO)

# Models for our knowledge graph
class Entity(BaseModel):
    """Represents a memory entity in the knowledge graph.
    
    Example:
    {
        "name": "John Smith",
        "type": "person", 
        "observations": ["Works at Neo4j", "Lives in San Francisco", "Expert in graph databases"]
    }
    """
    name: str = Field(
        description="Unique identifier/name for the entity. Should be descriptive and specific.",
        min_length=1,
        examples=["John Smith", "Neo4j Inc", "San Francisco"]
    )
    type: str = Field(
        description="Category or classification of the entity. Common types: 'person', 'company', 'location', 'concept', 'event'",
        min_length=1,
        examples=["person", "company", "location", "concept", "event"],
        pattern=r'^[A-Za-z_][A-Za-z0-9_]*$'
    )
    observations: List[str] = Field(
        description="List of facts, observations, or notes about this entity. Each observation should be a complete, standalone fact.",
        examples=[["Works at Neo4j", "Lives in San Francisco"], ["Headquartered in Sweden", "Graph database company"]]
    )

class Relation(BaseModel):
    """Represents a relationship between two entities in the knowledge graph.
    
    Example:
    {
        "source": "John Smith",
        "target": "Neo4j Inc", 
        "relationType": "WORKS_AT"
    }
    """
    source: str = Field(
        description="Name of the source entity (must match an existing entity name exactly)",
        min_length=1,
        examples=["John Smith", "Neo4j Inc"]
    )
    target: str = Field(
        description="Name of the target entity (must match an existing entity name exactly)",
        min_length=1, 
        examples=["Neo4j Inc", "San Francisco"]
    )
    relationType: str = Field(
        description="Type of relationship between source and target. Use descriptive, uppercase names with underscores.",
        min_length=1,
        examples=["WORKS_AT", "LIVES_IN", "MANAGES", "COLLABORATES_WITH", "LOCATED_IN"],
        pattern=r'^[A-Za-z_][A-Za-z0-9_]*$'
    )

class KnowledgeGraph(BaseModel):
    """Complete knowledge graph containing entities and their relationships."""
    entities: List[Entity] = Field(
        description="List of all entities in the knowledge graph",
        default=[]
    )
    relations: List[Relation] = Field(
        description="List of all relationships between entities",
        default=[]
    )

class ObservationAddition(BaseModel):
    """Request to add new observations to an existing entity.
    
    Example:
    {
        "entityName": "John Smith",
        "observations": ["Recently promoted to Senior Engineer", "Speaks fluent German"]
    }
    """
    entityName: str = Field(
        description="Exact name of the existing entity to add observations to",
        min_length=1,
        examples=["John Smith", "Neo4j Inc"]
    )
    observations: List[str] = Field(
        description="New observations/facts to add to the entity. Each should be unique and informative.",
        min_length=1
    )

class ObservationDeletion(BaseModel):
    """Request to delete specific observations from an existing entity.
    
    Example:
    {
        "entityName": "John Smith", 
        "observations": ["Old job title", "Outdated contact info"]
    }
    """
    entityName: str = Field(
        description="Exact name of the existing entity to remove observations from",
        min_length=1,
        examples=["John Smith", "Neo4j Inc"]
    )
    observations: List[str] = Field(
        description="Exact observation texts to delete from the entity (must match existing observations exactly)",
        min_length=1
    )

class ConversationChunk(BaseModel):
    """Represents a single chunk/message in a conversation stored in Neo4j.
    
    Example:
    {
        "conv_id": "session-2024-01-15",
        "chunk_number": 3,
        "content": "User asked about graph databases",
        "timestamp": "2024-01-15T10:30:00+00:00",
        "role": "user"
    }
    """
    conv_id: str = Field(
        description="Conversation identifier — a free-form string chosen by the LLM (e.g. UUID, date, session name)",
        min_length=1,
        examples=["session-2024-01-15", "conv-abc123", "project-alpha"]
    )
    chunk_number: int = Field(
        description="Sequential order of this chunk within the conversation (0-based)",
        ge=0,
        examples=[0, 1, 2, 10]
    )
    content: str = Field(
        description="Text content of this chunk/message",
        examples=["User asked about graph databases", "Assistant explained Cypher query language"]
    )
    timestamp: str = Field(
        description="ISO 8601 timestamp when this chunk was created (UTC recommended)",
        examples=["2024-01-15T10:30:00+00:00", "2024-01-15T10:31:00Z"]
    )
    role: str = Field(
        description="Role of the speaker for this chunk",
        default="user",
        examples=["user", "assistant", "system"]
    )

class ConversationExport(BaseModel):
    """Full export of a conversation including chunks and the knowledge graph snapshot."""
    conv_id: str = Field(description="The conversation ID that was exported")
    chunks: List[ConversationChunk] = Field(
        description="All conversation chunks in chronological order",
        default=[]
    )
    entities: List[Entity] = Field(
        description="All knowledge graph entities at time of export",
        default=[]
    )
    relations: List[Relation] = Field(
        description="All knowledge graph relations at time of export",
        default=[]
    )


def calculate_tokens(text: str, encoding: str = "cl100k_base") -> Dict[str, Any]:
    """Count the number of tokens in a text string using tiktoken.
    
    This is a pure utility function — it does not interact with Neo4j.
    Use it to check how many tokens a piece of text consumes before sending
    it to an LLM, or to decide when to summarize/branch a conversation.
    
    Args:
        text: The text to count tokens for.
        encoding: The tiktoken encoding to use. Default 'cl100k_base' is
                  compatible with GPT-4, GPT-3.5-turbo, and most modern models.
    
    Returns:
        dict with token_count, encoding, character_count, and a cost note.
    """
    enc = tiktoken.get_encoding(encoding)
    tokens = enc.encode(text)
    return {
        "token_count": len(tokens),
        "encoding": encoding,
        "character_count": len(text),
        "approx_cost_note": "Token count only; cost depends on your model/provider."
    }


class Neo4jMemory:
    def __init__(self, neo4j_driver: AsyncDriver):
        self.driver = neo4j_driver

    async def create_fulltext_index(self):
        """Create search indexes for Memory and ConversationChunk nodes if they don't exist."""
        try:
            query = "CREATE FULLTEXT INDEX search IF NOT EXISTS FOR (m:Memory) ON EACH [m.name, m.type, m.observations];"
            await self.driver.execute_query(query, routing_control=RoutingControl.WRITE)
            logger.info("Created fulltext search index")
        except Exception as e:
            logger.debug(f"Fulltext index creation: {e}")

        try:
            await self.driver.execute_query(
                "CREATE INDEX chunk_conv_id IF NOT EXISTS FOR (c:ConversationChunk) ON (c.conv_id);",
                routing_control=RoutingControl.WRITE
            )
            logger.info("Created ConversationChunk conv_id index")
        except Exception as e:
            logger.debug(f"chunk_conv_id index creation: {e}")

        try:
            await self.driver.execute_query(
                "CREATE INDEX chunk_timestamp IF NOT EXISTS FOR (c:ConversationChunk) ON (c.timestamp);",
                routing_control=RoutingControl.WRITE
            )
            logger.info("Created ConversationChunk timestamp index")
        except Exception as e:
            logger.debug(f"chunk_timestamp index creation: {e}")

    async def load_graph(self, filter_query: str = "*"):
        """Load the entire knowledge graph from Neo4j."""
        logger.info("Loading knowledge graph from Neo4j")
        query = """
            CALL db.index.fulltext.queryNodes('search', $filter) yield node as entity, score
            OPTIONAL MATCH (entity)-[r]-(other)
            RETURN collect(distinct {
                name: entity.name, 
                type: entity.type, 
                observations: entity.observations
            }) as nodes,
            collect(distinct {
                source: startNode(r).name, 
                target: endNode(r).name, 
                relationType: type(r)
            }) as relations
        """
        
        result = await self.driver.execute_query(query, {"filter": filter_query}, routing_control=RoutingControl.READ)
        
        if not result.records:
            return KnowledgeGraph(entities=[], relations=[])
        
        record = result.records[0]
        nodes = record.get('nodes', list())
        rels = record.get('relations', list())
        
        entities = [
            Entity(
                name=node['name'],
                type=node['type'],
                observations=node.get('observations', list())
            )
            for node in nodes if node.get('name')
        ]
        
        relations = [
            Relation(
                source=rel['source'],
                target=rel['target'],
                relationType=rel['relationType']
            )
            for rel in rels if rel.get('relationType')
        ]
        
        logger.debug(f"Loaded entities: {entities}")
        logger.debug(f"Loaded relations: {relations}")
        
        return KnowledgeGraph(entities=entities, relations=relations)

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple new entities in the knowledge graph."""
        logger.info(f"Creating {len(entities)} entities")
        for entity in entities:
            query = f"""
            WITH $entity as entity
            MERGE (e:Memory {{ name: entity.name }})
            SET e += entity {{ .type, .observations }}
            SET e:`{entity.type}`
            """
            await self.driver.execute_query(query, {"entity": entity.model_dump()}, routing_control=RoutingControl.WRITE)

        return entities

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple new relations between entities."""
        logger.info(f"Creating {len(relations)} relations")
        for relation in relations:
            query = f"""
            WITH $relation as relation
            MATCH (from:Memory),(to:Memory)
            WHERE from.name = relation.source
            AND  to.name = relation.target
            MERGE (from)-[r:`{relation.relationType}`]->(to)
            """
            
            await self.driver.execute_query(
                query, 
                {"relation": relation.model_dump()},
                routing_control=RoutingControl.WRITE
            )

        return relations

    async def add_observations(self, observations: List[ObservationAddition]) -> List[Dict[str, Any]]:
        """Add new observations to existing entities."""
        logger.info(f"Adding observations to {len(observations)} entities")
        query = """
        UNWIND $observations as obs  
        MATCH (e:Memory { name: obs.entityName })
        WITH e, [o in obs.observations WHERE NOT o IN e.observations] as new
        SET e.observations = coalesce(e.observations,[]) + new
        RETURN e.name as name, new
        """
            
        result = await self.driver.execute_query(
            query, 
            {"observations": [obs.model_dump() for obs in observations]},
            routing_control=RoutingControl.WRITE
        )

        results = [{"entityName": record.get("name"), "addedObservations": record.get("new")} for record in result.records]
        return results

    async def delete_entities(self, entity_names: List[str]) -> None:
        """Delete multiple entities and their associated relations."""
        logger.info(f"Deleting {len(entity_names)} entities")
        query = """
        UNWIND $entities as name
        MATCH (e:Memory { name: name })
        DETACH DELETE e
        """
        
        await self.driver.execute_query(query, {"entities": entity_names}, routing_control=RoutingControl.WRITE)
        logger.info(f"Successfully deleted {len(entity_names)} entities")

    async def delete_observations(self, deletions: List[ObservationDeletion]) -> None:
        """Delete specific observations from entities."""
        logger.info(f"Deleting observations from {len(deletions)} entities")
        query = """
        UNWIND $deletions as d  
        MATCH (e:Memory { name: d.entityName })
        SET e.observations = [o in coalesce(e.observations,[]) WHERE NOT o IN d.observations]
        """
        await self.driver.execute_query(
            query, 
            {"deletions": [deletion.model_dump() for deletion in deletions]},
            routing_control=RoutingControl.WRITE
        )
        logger.info(f"Successfully deleted observations from {len(deletions)} entities")

    async def delete_relations(self, relations: List[Relation]) -> None:
        """Delete multiple relations from the graph."""
        logger.info(f"Deleting {len(relations)} relations")
        for relation in relations:
            query = f"""
            WITH $relation as relation
            MATCH (source:Memory)-[r:`{relation.relationType}`]->(target:Memory)
            WHERE source.name = relation.source
            AND target.name = relation.target
            DELETE r
            """
            await self.driver.execute_query(
                query, 
                {"relation": relation.model_dump()},
                routing_control=RoutingControl.WRITE
            )
        logger.info(f"Successfully deleted {len(relations)} relations")

    async def read_graph(self) -> KnowledgeGraph:
        """Read the entire knowledge graph."""
        return await self.load_graph()

    async def search_memories(self, query: str) -> KnowledgeGraph:
        """Search for memories based on a query with Fulltext Search."""
        logger.info(f"Searching for memories with query: '{query}'")
        return await self.load_graph(query)

    async def find_memories_by_name(self, names: List[str]) -> KnowledgeGraph:
        """Find specific memories by their names. This does not use fulltext search."""
        logger.info(f"Finding {len(names)} memories by name")
        query = """
        MATCH (e:Memory)
        WHERE e.name IN $names
        RETURN  e.name as name, 
                e.type as type, 
                e.observations as observations
        """
        result_nodes = await self.driver.execute_query(query, {"names": names}, routing_control=RoutingControl.READ)
        entities: list[Entity] = list()
        for record in result_nodes.records:
            entities.append(Entity(
                name=record['name'],
                type=record['type'],
                observations=record.get('observations', list())
            ))
        
        # Get relations for found entities
        relations: list[Relation] = list()
        if entities:
            query = """
            MATCH (source:Memory)-[r]->(target:Memory)
            WHERE source.name IN $names OR target.name IN $names
            RETURN  source.name as source, 
                    target.name as target, 
                    type(r) as relationType
            """
            result_relations = await self.driver.execute_query(query, {"names": names}, routing_control=RoutingControl.READ)
            for record in result_relations.records:
                relations.append(Relation(
                    source=record["source"],
                    target=record["target"],
                    relationType=record["relationType"]
                ))
        
        logger.info(f"Found {len(entities)} entities and {len(relations)} relations")
        return KnowledgeGraph(entities=entities, relations=relations)

    # -------------------------------------------------------------------------
    # Conversation chunk methods (new)
    # -------------------------------------------------------------------------

    async def get_recent_chunks(self, conv_id: str, limit: int = 10) -> List[ConversationChunk]:
        """Return the most recent N chunks from a conversation in chronological order."""
        logger.info(f"Getting recent {limit} chunks for conv_id='{conv_id}'")
        query = """
        MATCH (c:ConversationChunk {conv_id: $conv_id})
        RETURN c.conv_id AS conv_id,
               c.chunk_number AS chunk_number,
               c.content AS content,
               c.timestamp AS timestamp,
               c.role AS role
        ORDER BY c.chunk_number DESC
        LIMIT $limit
        """
        result = await self.driver.execute_query(
            query,
            {"conv_id": conv_id, "limit": limit},
            routing_control=RoutingControl.READ
        )
        chunks = [
            ConversationChunk(
                conv_id=r["conv_id"],
                chunk_number=r["chunk_number"],
                content=r["content"],
                timestamp=str(r["timestamp"]),
                role=r["role"] if r["role"] else "user"
            )
            for r in result.records
        ]
        # Reverse so result is chronological (oldest first)
        return list(reversed(chunks))

    async def summarize_conversation(
        self,
        conv_id: str,
        max_input_tokens: int = 4000,
        focus: Optional[str] = None
    ) -> str:
        """Fetch conversation chunks up to max_input_tokens and return concatenated text.
        
        The returned text is intended to be passed to the calling LLM for summarization.
        This method does NOT call any LLM — it only retrieves and concatenates chunks.
        """
        logger.info(f"Summarizing conversation '{conv_id}' (max_input_tokens={max_input_tokens})")
        enc = tiktoken.get_encoding("cl100k_base")

        query = """
        MATCH (c:ConversationChunk {conv_id: $conv_id})
        RETURN c.chunk_number AS chunk_number,
               c.content AS content,
               c.role AS role
        ORDER BY c.chunk_number ASC
        """
        result = await self.driver.execute_query(
            query,
            {"conv_id": conv_id},
            routing_control=RoutingControl.READ
        )

        focus_note = f"Focus on: {focus}\n\n" if focus else ""
        parts: List[str] = []
        total_tokens = len(enc.encode(focus_note))

        for record in result.records:
            role = record["role"] if record["role"] else "user"
            content = record["content"]
            line = f"[{role}]: {content}\n"
            line_tokens = len(enc.encode(line))
            if total_tokens + line_tokens > max_input_tokens:
                break
            parts.append(line)
            total_tokens += line_tokens

        logger.info(f"Summarize: assembled {len(parts)} chunks, ~{total_tokens} tokens")
        return focus_note + "".join(parts)

    async def branch_conversation(
        self,
        source_conv_id: str,
        new_conv_id: str,
        carry_over_summary: str
    ) -> Dict[str, str]:
        """Start a new conversation branch seeded with a carry-over summary as chunk 0."""
        logger.info(f"Branching '{source_conv_id}' -> '{new_conv_id}'")
        timestamp = datetime.now(timezone.utc).isoformat()
        query = """
        CREATE (c:ConversationChunk {
            conv_id: $conv_id,
            chunk_number: 0,
            content: $content,
            timestamp: $timestamp,
            role: 'system'
        })
        """
        await self.driver.execute_query(
            query,
            {"conv_id": new_conv_id, "content": carry_over_summary, "timestamp": timestamp},
            routing_control=RoutingControl.WRITE
        )
        return {
            "status": "success",
            "source_conv_id": source_conv_id,
            "new_conv_id": new_conv_id,
            "message": f"New branch '{new_conv_id}' created with carry-over summary as chunk 0."
        }

    async def search_by_date(
        self,
        conv_id: str,
        since: str,
        until: str,
        limit: int = 50
    ) -> List[ConversationChunk]:
        """Find conversation chunks whose timestamp falls within [since, until].
        
        Both since and until should be ISO 8601 strings (e.g. '2024-01-15T00:00:00Z').
        String comparison works correctly for ISO 8601 format.
        """
        logger.info(f"search_by_date conv_id='{conv_id}' since='{since}' until='{until}'")
        query = """
        MATCH (c:ConversationChunk {conv_id: $conv_id})
        WHERE c.timestamp >= $since AND c.timestamp <= $until
        RETURN c.conv_id AS conv_id,
               c.chunk_number AS chunk_number,
               c.content AS content,
               c.timestamp AS timestamp,
               c.role AS role
        ORDER BY c.timestamp ASC
        LIMIT $limit
        """
        result = await self.driver.execute_query(
            query,
            {"conv_id": conv_id, "since": since, "until": until, "limit": limit},
            routing_control=RoutingControl.READ
        )
        return [
            ConversationChunk(
                conv_id=r["conv_id"],
                chunk_number=r["chunk_number"],
                content=r["content"],
                timestamp=str(r["timestamp"]),
                role=r["role"] if r["role"] else "user"
            )
            for r in result.records
        ]

    async def export_conversation_json(
        self,
        conv_id: str,
        include_embeddings: bool = False
    ) -> ConversationExport:
        """Export all chunks for a conversation plus the full knowledge graph as a JSON-serialisable object."""
        logger.info(f"Exporting conversation '{conv_id}' (include_embeddings={include_embeddings})")
        chunks = await self.get_recent_chunks(conv_id, limit=100_000)
        graph = await self.read_graph()
        return ConversationExport(
            conv_id=conv_id,
            chunks=chunks,
            entities=graph.entities,
            relations=graph.relations
        )

    async def prune_old_chunks(
        self,
        conv_id: str,
        keep_last_n: Optional[int] = None,
        older_than_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Delete old conversation chunks.
        
        Provide keep_last_n to keep only the N most recent chunks, or
        older_than_days to delete chunks older than that many days.
        Both can be provided simultaneously.
        """
        logger.info(f"prune_old_chunks conv_id='{conv_id}' keep_last_n={keep_last_n} older_than_days={older_than_days}")
        total_deleted = 0

        if keep_last_n is not None:
            query = """
            MATCH (c:ConversationChunk {conv_id: $conv_id})
            WITH c ORDER BY c.chunk_number DESC SKIP $skip
            WITH collect(c) AS to_delete
            UNWIND to_delete AS c
            DELETE c
            RETURN count(c) AS deleted
            """
            result = await self.driver.execute_query(
                query,
                {"conv_id": conv_id, "skip": keep_last_n},
                routing_control=RoutingControl.WRITE
            )
            if result.records:
                total_deleted += result.records[0].get("deleted", 0) or 0

        if older_than_days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
            query = """
            MATCH (c:ConversationChunk {conv_id: $conv_id})
            WHERE c.timestamp < $cutoff
            WITH collect(c) AS to_delete
            UNWIND to_delete AS c
            DELETE c
            RETURN count(c) AS deleted
            """
            result = await self.driver.execute_query(
                query,
                {"conv_id": conv_id, "cutoff": cutoff},
                routing_control=RoutingControl.WRITE
            )
            if result.records:
                total_deleted += result.records[0].get("deleted", 0) or 0

        return {
            "status": "success",
            "conv_id": conv_id,
            "chunks_deleted": total_deleted
        }
