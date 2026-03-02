import json
import logging
from typing import Literal, Optional

from neo4j import AsyncGraphDatabase
from pydantic import Field
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from fastmcp.server import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent
from neo4j.exceptions import Neo4jError
from mcp.types import ToolAnnotations

from .neo4j_memory import (
    Neo4jMemory,
    Entity,
    Relation,
    ObservationAddition,
    ObservationDeletion,
    KnowledgeGraph,
    ConversationChunk,
    ConversationExport,
    calculate_tokens,
)
from .utils import format_namespace

# Set up logging
logger = logging.getLogger('mcp_neo4j_memory')
logger.setLevel(logging.INFO)


def create_mcp_server(memory: Neo4jMemory, namespace: str = "") -> FastMCP:
    """Create an MCP server instance for memory management."""
    
    namespace_prefix = format_namespace(namespace)
    mcp: FastMCP = FastMCP("mcp-neo4j-memory", stateless_http=True)

    @mcp.tool(
        name=namespace_prefix + "read_graph",
        annotations=ToolAnnotations(title="Read Graph", 
                                          readOnlyHint=True, 
                                          destructiveHint=False, 
                                          idempotentHint=True, 
                                          openWorldHint=True))
    async def read_graph() -> ToolResult:
        """Read the entire knowledge graph with all entities and relationships.
        
        Returns the complete memory graph including all stored entities and their relationships.
        Use this to get a full overview of stored knowledge.
        
        Returns:
            KnowledgeGraph: Complete graph with all entities and relations
            
        Example response:
        {
            "entities": [
                {"name": "John Smith", "type": "person", "observations": ["Works at Neo4j"]},
                {"name": "Neo4j Inc", "type": "company", "observations": ["Graph database company"]}
            ],
            "relations": [
                {"source": "John Smith", "target": "Neo4j Inc", "relationType": "WORKS_AT"}
            ]
        }
        """
        logger.info("MCP tool: read_graph")
        try:
            result = await memory.read_graph()
            return ToolResult(content=[TextContent(type="text", text=result.model_dump_json())],
                          structured_content=result)
        except Neo4jError as e:
            logger.error(f"Neo4j error reading full knowledge graph: {e}")
            raise ToolError(f"Neo4j error reading full knowledge graph: {e}")
        except Exception as e:
            logger.error(f"Error reading full knowledge graph: {e}")
            raise ToolError(f"Error reading full knowledge graph: {e}")

    @mcp.tool(
        name=namespace_prefix + "create_entities",
        annotations=ToolAnnotations(title="Create Entities", 
                                          readOnlyHint=False, 
                                          destructiveHint=False, 
                                          idempotentHint=True, 
                                          openWorldHint=True))
    async def create_entities(entities: list[Entity] = Field(..., description="List of entities to create with name, type, and observations")) -> ToolResult:
        """Create multiple new entities in the knowledge graph.
        
        Creates new memory entities with their associated observations. If an entity with the same name
        already exists, this operation will merge the observations with existing ones.
        
            
        Returns:
            list[Entity]: The created entities with their final state
            
        Example call:
        {
            "entities": [
                {
                    "name": "Alice Johnson",
                    "type": "person",
                    "observations": ["Software engineer", "Lives in Seattle", "Enjoys hiking"]
                },
                {
                    "name": "Microsoft",
                    "type": "company", 
                    "observations": ["Technology company", "Headquartered in Redmond, WA"]
                }
            ]
        }
        """
        logger.info(f"MCP tool: create_entities ({len(entities)} entities)")
        try:
            entity_objects = [Entity.model_validate(entity) for entity in entities]
            result = await memory.create_entities(entity_objects)
            return ToolResult(content=[TextContent(type="text", text=json.dumps([e.model_dump() for e in result]))],
                          structured_content={"result": result})
        except Neo4jError as e:
            logger.error(f"Neo4j error creating entities: {e}")
            raise ToolError(f"Neo4j error creating entities: {e}")
        except Exception as e:
            logger.error(f"Error creating entities: {e}")
            raise ToolError(f"Error creating entities: {e}")

    @mcp.tool(
        name=namespace_prefix + "create_relations",
        annotations=ToolAnnotations(title="Create Relations", 
                                          readOnlyHint=False, 
                                          destructiveHint=False, 
                                          idempotentHint=True, 
                                          openWorldHint=True))
    async def create_relations(relations: list[Relation] = Field(..., description="List of relations to create between existing entities")) -> ToolResult:
        """Create multiple new relationships between existing entities in the knowledge graph.
        
        Creates directed relationships between entities that already exist. Both source and target
        entities must already be present in the graph. Use descriptive relationship types.
        
        Returns:
            list[Relation]: The created relationships
            
        Example call:
        {
            "relations": [
                {
                    "source": "Alice Johnson",
                    "target": "Microsoft", 
                    "relationType": "WORKS_AT"
                },
                {
                    "source": "Alice Johnson",
                    "target": "Seattle",
                    "relationType": "LIVES_IN"
                }
            ]
        }
        """
        logger.info(f"MCP tool: create_relations ({len(relations)} relations)")
        try:
            relation_objects = [Relation.model_validate(relation) for relation in relations]
            result = await memory.create_relations(relation_objects)
            return ToolResult(content=[TextContent(type="text", text=json.dumps([r.model_dump() for r in result]))],
                          structured_content={"result": result})
        except Neo4jError as e:
            logger.error(f"Neo4j error creating relations: {e}")
            raise ToolError(f"Neo4j error creating relations: {e}")
        except Exception as e:
            logger.error(f"Error creating relations: {e}")
            raise ToolError(f"Error creating relations: {e}")

    @mcp.tool(
        name=namespace_prefix + "add_observations",
        annotations=ToolAnnotations(title="Add Observations", 
                                          readOnlyHint=False, 
                                          destructiveHint=False, 
                                          idempotentHint=True, 
                                          openWorldHint=True))
    async def add_observations(observations: list[ObservationAddition] = Field(..., description="List of observations to add to existing entities")) -> ToolResult:
        """Add new observations/facts to existing entities in the knowledge graph.
        
        Appends new observations to entities that already exist. The entity must be present
        in the graph before adding observations. Each observation should be a distinct fact.
        
        Returns:
            list[dict]: Details about the added observations including entity name and new facts
            
        Example call:
        {
            "observations": [
                {
                    "entityName": "Alice Johnson",
                    "observations": ["Promoted to Senior Engineer", "Completed AWS certification"]
                },
                {
                    "entityName": "Microsoft",
                    "observations": ["Launched new AI products", "Stock price increased 15%"]
                }
            ]
        }
        """
        logger.info(f"MCP tool: add_observations ({len(observations)} additions)")
        try:
            observation_objects = [ObservationAddition.model_validate(obs) for obs in observations]
            result = await memory.add_observations(observation_objects)
            return ToolResult(content=[TextContent(type="text", text=json.dumps(result))],
                          structured_content={"result": result})
        except Neo4jError as e:
            logger.error(f"Neo4j error adding observations: {e}")
            raise ToolError(f"Neo4j error adding observations: {e}")
        except Exception as e:
            logger.error(f"Error adding observations: {e}")
            raise ToolError(f"Error adding observations: {e}")

    @mcp.tool(
        name=namespace_prefix + "delete_entities",
        annotations=ToolAnnotations(title="Delete Entities", 
                                          readOnlyHint=False, 
                                          destructiveHint=True, 
                                          idempotentHint=True, 
                                          openWorldHint=True))
    async def delete_entities(entityNames: list[str] = Field(..., description="List of exact entity names to delete permanently")) -> ToolResult:
        """Delete entities and all their associated relationships from the knowledge graph.
        
        Permanently removes entities from the graph along with all relationships they participate in.
        This is a destructive operation that cannot be undone. Entity names must match exactly.
        
        Returns:
            str: Success confirmation message
            
        Example call:
        {
            "entityNames": ["Old Company", "Outdated Person"]
        }
        
        Warning: This will delete the entities and ALL relationships they're involved in.
        """
        logger.info(f"MCP tool: delete_entities ({len(entityNames)} entities)")
        try:
            await memory.delete_entities(entityNames)
            return ToolResult(content=[TextContent(type="text", text="Entities deleted successfully")],
                              structured_content={"result": "Entities deleted successfully"})
        except Neo4jError as e:
            logger.error(f"Neo4j error deleting entities: {e}")
            raise ToolError(f"Neo4j error deleting entities: {e}")
        except Exception as e:
            logger.error(f"Error deleting entities: {e}")
            raise ToolError(f"Error deleting entities: {e}")

    @mcp.tool(
        name=namespace_prefix + "delete_observations",
        annotations=ToolAnnotations(title="Delete Observations", 
                                          readOnlyHint=False, 
                                          destructiveHint=True, 
                                          idempotentHint=True, 
                                          openWorldHint=True))
    async def delete_observations(deletions: list[ObservationDeletion] = Field(..., description="List of specific observations to remove from entities")) -> ToolResult:
        """Delete specific observations from existing entities in the knowledge graph.
        
        Removes specific observation texts from entities. The observation text must match exactly
        what is stored. The entity will remain but the specified observations will be deleted.
        
        Returns:
            str: Success confirmation message
            
        Example call:
        {
            "deletions": [
                {
                    "entityName": "Alice Johnson",
                    "observations": ["Old job title", "Outdated phone number"]
                },
                {
                    "entityName": "Microsoft", 
                    "observations": ["Former CEO information"]
                }
            ]
        }
        
        Note: Observation text must match exactly (case-sensitive) to be deleted.
        """
        logger.info(f"MCP tool: delete_observations ({len(deletions)} deletions)")
        try:    
            deletion_objects = [ObservationDeletion.model_validate(deletion) for deletion in deletions]
            await memory.delete_observations(deletion_objects)
            return ToolResult(content=[TextContent(type="text", text="Observations deleted successfully")],
                          structured_content={"result": "Observations deleted successfully"})
        except Neo4jError as e:
            logger.error(f"Neo4j error deleting observations: {e}")
            raise ToolError(f"Neo4j error deleting observations: {e}")
        except Exception as e:
            logger.error(f"Error deleting observations: {e}")
            raise ToolError(f"Error deleting observations: {e}")

    @mcp.tool(
        name=namespace_prefix + "delete_relations",
        annotations=ToolAnnotations(title="Delete Relations", 
                                          readOnlyHint=False, 
                                          destructiveHint=True, 
                                          idempotentHint=True, 
                                          openWorldHint=True))
    async def delete_relations(relations: list[Relation] = Field(..., description="List of specific relationships to delete from the graph")) -> ToolResult:
        """Delete specific relationships between entities in the knowledge graph.
        
        Removes relationships while keeping the entities themselves. The source, target, and 
        relationship type must match exactly for deletion. This only affects the relationships,
        not the entities they connect.
        
        Returns:
            str: Success confirmation message
            
        Example call:
        {
            "relations": [
                {
                    "source": "Alice Johnson",
                    "target": "Old Company",
                    "relationType": "WORKS_AT"
                },
                {
                    "source": "John Smith", 
                    "target": "Former City",
                    "relationType": "LIVES_IN"
                }
            ]
        }
        
        Note: All fields (source, target, relationType) must match exactly for deletion.
        """
        logger.info(f"MCP tool: delete_relations ({len(relations)} relations)")
        try:
            relation_objects = [Relation.model_validate(relation) for relation in relations]
            await memory.delete_relations(relation_objects)
            return ToolResult(content=[TextContent(type="text", text="Relations deleted successfully")],
                          structured_content={"result": "Relations deleted successfully"})
        except Neo4jError as e:
            logger.error(f"Neo4j error deleting relations: {e}")
            raise ToolError(f"Neo4j error deleting relations: {e}")
        except Exception as e:
            logger.error(f"Error deleting relations: {e}")
            raise ToolError(f"Error deleting relations: {e}")

    @mcp.tool(
        name=namespace_prefix + "search_memories",
        annotations=ToolAnnotations(title="Search Memories", 
                                          readOnlyHint=True, 
                                          destructiveHint=False, 
                                          idempotentHint=True, 
                                          openWorldHint=True))
    async def search_memories(query: str = Field(..., description="Fulltext search query to find entities by name, type, or observations")) -> ToolResult:
        """Search for entities in the knowledge graph using fulltext search.
        
        Searches across entity names, types, and observations using Neo4j's fulltext index.
        Returns matching entities and their related connections. Supports partial matches
        and multiple search terms.
        
        Returns:
            KnowledgeGraph: Subgraph containing matching entities and their relationships
            
        Example call:
        {
            "query": "engineer software"
        }
        
        This searches for entities containing "engineer" or "software" in their name, type, or observations.
        """
        logger.info(f"MCP tool: search_memories ('{query}')")
        try:
            result = await memory.search_memories(query)
            return ToolResult(content=[TextContent(type="text", text=result.model_dump_json())],
                              structured_content=result)
        except Neo4jError as e:
            logger.error(f"Neo4j error searching memories: {e}")
            raise ToolError(f"Neo4j error searching memories: {e}")
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise ToolError(f"Error searching memories: {e}")
        
    @mcp.tool(
        name=namespace_prefix + "find_memories_by_name",
        annotations=ToolAnnotations(title="Find Memories by Name",
                                          readOnlyHint=True, 
                                          destructiveHint=False, 
                                          idempotentHint=True, 
                                          openWorldHint=True))
    async def find_memories_by_name(names: list[str] = Field(..., description="List of exact entity names to retrieve")) -> ToolResult:
        """Find specific entities by their exact names.
        
        Retrieves entities that exactly match the provided names, along with all their
        relationships and connected entities. Use this when you know the exact entity names.
        
        Returns:
            KnowledgeGraph: Subgraph containing the specified entities and their relationships
            
        Example call:
        {
            "names": ["Alice Johnson", "Microsoft", "Seattle"]
        }
        
        This retrieves the entities with exactly those names plus their connections.
        """
        logger.info(f"MCP tool: find_memories_by_name ({len(names)} names)")
        try:
            result = await memory.find_memories_by_name(names)
            return ToolResult(content=[TextContent(type="text", text=result.model_dump_json())],
                              structured_content=result)
        except Neo4jError as e:
            logger.error(f"Neo4j error finding memories by name: {e}")
            raise ToolError(f"Neo4j error finding memories by name: {e}")
        except Exception as e:
            logger.error(f"Error finding memories by name: {e}")
            raise ToolError(f"Error finding memories by name: {e}")

    # -------------------------------------------------------------------------
    # New context-management tools
    # -------------------------------------------------------------------------

    @mcp.tool(
        name=namespace_prefix + "get_recent_chunks",
        annotations=ToolAnnotations(
            title="Get Recent Chunks",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def get_recent_chunks(
        conv_id: str = Field(..., description="Conversation ID to retrieve chunks from"),
        limit: int = Field(10, description="Maximum number of recent chunks to return (default 10)", ge=1, le=10000),
    ) -> ToolResult:
        """Return the most recent N conversation chunks in chronological order.
        
        Use this to quickly load only the last few turns of a conversation without
        fetching the entire history. Helps avoid context window overflow.
        
        Returns:
            list[ConversationChunk]: Chunks ordered oldest-first (chronological)
            
        Example call:
        {
            "conv_id": "session-2024-01-15",
            "limit": 5
        }
        """
        logger.info(f"MCP tool: get_recent_chunks conv_id='{conv_id}' limit={limit}")
        try:
            result = await memory.get_recent_chunks(conv_id, limit)
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps([c.model_dump() for c in result]))],
                structured_content={"result": [c.model_dump() for c in result]},
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error getting recent chunks: {e}")
            raise ToolError(f"Neo4j error getting recent chunks: {e}")
        except Exception as e:
            logger.error(f"Error getting recent chunks: {e}")
            raise ToolError(f"Error getting recent chunks: {e}")

    @mcp.tool(
        name=namespace_prefix + "summarize_conversation",
        annotations=ToolAnnotations(
            title="Summarize Conversation",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def summarize_conversation(
        conv_id: str = Field(..., description="Conversation ID to summarize"),
        max_input_tokens: int = Field(4000, description="Maximum tokens of conversation history to include (default 4000)", ge=100, le=100000),
        focus: Optional[str] = Field(None, description="Optional focus hint prepended to the output, e.g. 'key decisions made'"),
    ) -> ToolResult:
        """Retrieve and concatenate conversation chunks up to a token limit.
        
        Returns the raw conversation text (up to max_input_tokens tokens) so the calling
        LLM can summarize it. This tool does NOT call any LLM — it only fetches and
        concatenates stored chunks. Use it when context is getting large (~6000 tokens).
        
        Returns:
            str: Concatenated conversation text ready for summarization
            
        Example call:
        {
            "conv_id": "session-2024-01-15",
            "max_input_tokens": 3000,
            "focus": "key decisions and action items"
        }
        """
        logger.info(f"MCP tool: summarize_conversation conv_id='{conv_id}'")
        try:
            result = await memory.summarize_conversation(conv_id, max_input_tokens, focus)
            return ToolResult(
                content=[TextContent(type="text", text=result)],
                structured_content={"text": result},
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error summarizing conversation: {e}")
            raise ToolError(f"Neo4j error summarizing conversation: {e}")
        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}")
            raise ToolError(f"Error summarizing conversation: {e}")

    @mcp.tool(
        name=namespace_prefix + "branch_conversation",
        annotations=ToolAnnotations(
            title="Branch Conversation",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    async def branch_conversation(
        source_conv_id: str = Field(..., description="The existing conversation ID to branch from"),
        new_conv_id: str = Field(..., description="The new conversation ID for the branch"),
        carry_over_summary: str = Field(..., description="Summary text to carry over as the first chunk (chunk 0) of the new branch"),
    ) -> ToolResult:
        """Start a new conversation branch carrying over a summary from the old one.
        
        Use this when the current conversation context is full. The carry_over_summary
        becomes chunk 0 (role='system') of the new branch, preserving key context.
        The original conversation is not modified.
        
        Returns:
            dict: Success status and the new conversation ID
            
        Example call:
        {
            "source_conv_id": "session-2024-01-15",
            "new_conv_id": "session-2024-01-15-branch-2",
            "carry_over_summary": "We discussed graph database design. Key decisions: use Neo4j, model users as nodes."
        }
        """
        logger.info(f"MCP tool: branch_conversation '{source_conv_id}' -> '{new_conv_id}'")
        try:
            result = await memory.branch_conversation(source_conv_id, new_conv_id, carry_over_summary)
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(result))],
                structured_content=result,
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error branching conversation: {e}")
            raise ToolError(f"Neo4j error branching conversation: {e}")
        except Exception as e:
            logger.error(f"Error branching conversation: {e}")
            raise ToolError(f"Error branching conversation: {e}")

    @mcp.tool(
        name=namespace_prefix + "search_by_date",
        annotations=ToolAnnotations(
            title="Search by Date",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def search_by_date(
        conv_id: str = Field(..., description="Conversation ID to search within"),
        since: str = Field(..., description="Start of time window as ISO 8601 string, e.g. '2024-01-15T00:00:00Z'"),
        until: str = Field(..., description="End of time window as ISO 8601 string, e.g. '2024-01-15T23:59:59Z'"),
        limit: int = Field(50, description="Maximum number of chunks to return (default 50)", ge=1, le=10000),
    ) -> ToolResult:
        """Find conversation chunks created within a specific time window.
        
        Useful for questions like "What did we discuss last week?" or "Show me everything
        since yesterday." Both since and until are inclusive ISO 8601 timestamps.
        
        Returns:
            list[ConversationChunk]: Matching chunks ordered by timestamp ascending
            
        Example call:
        {
            "conv_id": "session-2024-01-15",
            "since": "2024-01-15T09:00:00Z",
            "until": "2024-01-15T17:00:00Z"
        }
        """
        logger.info(f"MCP tool: search_by_date conv_id='{conv_id}' since='{since}' until='{until}'")
        try:
            result = await memory.search_by_date(conv_id, since, until, limit)
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps([c.model_dump() for c in result]))],
                structured_content={"result": [c.model_dump() for c in result]},
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error searching by date: {e}")
            raise ToolError(f"Neo4j error searching by date: {e}")
        except Exception as e:
            logger.error(f"Error searching by date: {e}")
            raise ToolError(f"Error searching by date: {e}")

    @mcp.tool(
        name=namespace_prefix + "export_conversation_json",
        annotations=ToolAnnotations(
            title="Export Conversation JSON",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def export_conversation_json(
        conv_id: str = Field(..., description="Conversation ID to export"),
        include_embeddings: bool = Field(False, description="Whether to include embedding vectors (default false — not yet implemented, reserved for future use)"),
    ) -> ToolResult:
        """Export a conversation and the full knowledge graph as clean JSON.
        
        Returns all conversation chunks plus all Memory entities and relations.
        Useful for backup, debugging, moving to another system, or offline analysis.
        
        Returns:
            ConversationExport: JSON object with conv_id, chunks, entities, relations
            
        Example call:
        {
            "conv_id": "session-2024-01-15"
        }
        """
        logger.info(f"MCP tool: export_conversation_json conv_id='{conv_id}'")
        try:
            result = await memory.export_conversation_json(conv_id, include_embeddings)
            return ToolResult(
                content=[TextContent(type="text", text=result.model_dump_json())],
                structured_content=result,
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error exporting conversation: {e}")
            raise ToolError(f"Neo4j error exporting conversation: {e}")
        except Exception as e:
            logger.error(f"Error exporting conversation: {e}")
            raise ToolError(f"Error exporting conversation: {e}")

    @mcp.tool(
        name=namespace_prefix + "prune_old_chunks",
        annotations=ToolAnnotations(
            title="Prune Old Chunks",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    async def prune_old_chunks(
        conv_id: str = Field(..., description="Conversation ID whose chunks should be pruned"),
        keep_last_n: Optional[int] = Field(None, description="Keep only the N most recent chunks; delete the rest", ge=0),
        older_than_days: Optional[int] = Field(None, description="Delete chunks older than this many days", ge=0),
    ) -> ToolResult:
        """Delete old conversation chunks to prevent the database from growing forever.
        
        Provide keep_last_n to keep only the most recent N chunks, or older_than_days
        to delete chunks older than that many days. Both can be provided simultaneously
        (both conditions are applied independently).
        
        WARNING: This permanently deletes chunks and cannot be undone.
        
        Returns:
            dict: Success status and number of chunks deleted
            
        Example call — keep only last 20 chunks:
        {
            "conv_id": "session-2024-01-15",
            "keep_last_n": 20
        }
        
        Example call — delete chunks older than 30 days:
        {
            "conv_id": "session-2024-01-15",
            "older_than_days": 30
        }
        """
        logger.info(f"MCP tool: prune_old_chunks conv_id='{conv_id}' keep_last_n={keep_last_n} older_than_days={older_than_days}")
        if keep_last_n is None and older_than_days is None:
            raise ToolError("At least one of keep_last_n or older_than_days must be provided.")
        try:
            result = await memory.prune_old_chunks(conv_id, keep_last_n, older_than_days)
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(result))],
                structured_content=result,
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error pruning chunks: {e}")
            raise ToolError(f"Neo4j error pruning chunks: {e}")
        except Exception as e:
            logger.error(f"Error pruning chunks: {e}")
            raise ToolError(f"Error pruning chunks: {e}")

    @mcp.tool(
        name=namespace_prefix + "calculate_tokens",
        annotations=ToolAnnotations(
            title="Calculate Tokens",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def calculate_tokens(
        text: str = Field(..., description="The text to count tokens for"),
        encoding: str = Field("cl100k_base", description="Tiktoken encoding to use (default 'cl100k_base', compatible with GPT-4 and most modern models)"),
    ) -> ToolResult:
        """Count how many tokens a piece of text uses — proactively check 'am I about to overflow?'
        
        PRIMARY USE CASE: Call this tool on the current conversation history or prompt to measure
        its token length. If token_count is approaching your model's context limit, call
        summarize_conversation to compress old turns, then branch_conversation to start fresh.
        This prevents context window freeze / overflow without losing important information.
        
        Recommended overflow-prevention workflow:
        1. calculate_tokens(current_history) → check token_count
        2. If token_count > ~6000 (for an 8k model): call summarize_conversation
        3. Then call branch_conversation with the summary to start a fresh context window
        
        Uses tiktoken locally — no network call, instant result.
        
        Returns:
            dict: token_count (int), encoding (str), character_count (int), approx_cost_note (str)
            
        Example call:
        {
            "text": "Hello, how are you today?",
            "encoding": "cl100k_base"
        }
        
        Example response:
        {
            "token_count": 7,
            "encoding": "cl100k_base",
            "character_count": 26,
            "approx_cost_note": "Token count only; cost depends on your model/provider."
        }
        """
        logger.info(f"MCP tool: calculate_tokens (encoding='{encoding}', text_len={len(text)})")
        try:
            result = calculate_tokens(text, encoding)
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(result))],
                structured_content=result,
            )
        except Exception as e:
            logger.error(f"Error calculating tokens: {e}")
            raise ToolError(f"Error calculating tokens: {e}")

    return mcp


async def main(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    transport: Literal["stdio", "sse", "http"] = "stdio",
    namespace: str = "",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp/",
    allow_origins: list[str] = [],
    allowed_hosts: list[str] = [],
) -> None:
    logger.info(f"Starting Neo4j MCP Memory Server")
    logger.info(f"Connecting to Neo4j with DB URL: {neo4j_uri}")

    # Connect to Neo4j
    neo4j_driver = AsyncGraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password), 
        database=neo4j_database
    )
    
    # Verify connection
    try:
    await neo4j_driver.verify_connectivity()
    logger.info(f"Connected to Neo4j at {neo4j_uri}")
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {e}")
    await neo4j_driver.close()
    exit(1)

    # Initialize memory
    memory = Neo4jMemory(neo4j_driver)
    logger.info("Neo4jMemory initialized")
    
    # Create fulltext index
    await memory.create_fulltext_index()
    
    # Configure security middleware
    custom_middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        ),
        Middleware(TrustedHostMiddleware,
                   allowed_hosts=allowed_hosts)
    ]

    # Create MCP server
    mcp = create_mcp_server(memory, namespace)
    logger.info("MCP server created")

    # Run the server with the specified transport
    logger.info(f"Starting server with transport: {transport}")
    match transport:
        case "http":
            logger.info(f"HTTP server starting on {host}:{port}{path}")
            await mcp.run_http_async(host=host, port=port, path=path, middleware=custom_middleware)
        case "stdio":
            logger.info("STDIO server starting")
            await mcp.run_stdio_async()
        case "sse":
            logger.info(f"SSE server starting on {host}:{port}{path}")
            await mcp.run_http_async(host=host, port=port, path=path, middleware=custom_middleware, transport="sse")
        case _:
            raise ValueError(f"Unsupported transport: {transport}")
