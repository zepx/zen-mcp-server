"""
Consensus tool for multi-model perspective gathering and validation
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from mcp.types import TextContent
from pydantic import Field, field_validator

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from config import (
    CONSENSUS_PROVIDER_CONCURRENCY,
    DEFAULT_CONSENSUS_CONCURRENCY,
    DEFAULT_CONSENSUS_MAX_INSTANCES_PER_COMBINATION,
    DEFAULT_CONSENSUS_TIMEOUT,
)
from systemprompts import CONSENSUS_PROMPT

from .base import BaseTool, ToolRequest

logger = logging.getLogger(__name__)


class ConsensusRequest(ToolRequest):
    """Request model for consensus tool"""

    prompt: str = Field(
        ...,
        description=(
            "Description of what to get consensus on, testing objectives, and specific scope/focus areas. "
            "Be as detailed as possible about the proposal, plan, or idea you want multiple perspectives on."
        ),
    )
    models: list[str] = Field(
        ...,
        description=(
            "List of models to consult for consensus. Supports stance specification using 'model:stance' format. "
            "Stances: ':for' (supportive perspective), ':against' (critical perspective), or no stance (neutral). "
            "Examples: ['o3:for', 'pro:against', 'grok-3'] creates a debate format. "
            "Maximum 2 instances per model+stance combination."
        ),
    )
    files: Optional[list[str]] = Field(
        default_factory=list,
        description="Optional files or directories for additional context (must be absolute paths)",
    )
    images: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "Optional images showing expected UI changes, design requirements, "
            "or visual references for the consensus analysis"
        ),
    )
    focus_areas: Optional[list[str]] = Field(
        default_factory=list,
        description="Specific aspects to focus on (e.g., 'performance', 'security', 'user experience')",
    )

    @field_validator("models", mode="before")
    @classmethod
    def validate_models_not_empty(cls, v):
        if not v:
            raise ValueError("At least one model must be specified")
        return v


class ConsensusTool(BaseTool):
    """Multi-model consensus tool for gathering diverse perspectives on technical proposals"""

    def __init__(self):
        super().__init__()
        # Per-provider semaphores for rate limiting using config values
        self._provider_semaphores = {
            provider: asyncio.Semaphore(limit) for provider, limit in CONSENSUS_PROVIDER_CONCURRENCY.items()
        }
        self._default_semaphore = asyncio.Semaphore(DEFAULT_CONSENSUS_CONCURRENCY)  # Fallback for unknown providers

    def get_name(self) -> str:
        return "consensus"

    def get_description(self) -> str:
        return (
            "MULTI-MODEL CONSENSUS - Gather diverse perspectives from multiple AI models on technical proposals, "
            "plans, and ideas. Perfect for validation, feasibility assessment, and getting comprehensive "
            "viewpoints on complex decisions. Supports stance steering (:for/:against) to create structured "
            "debates and balanced analysis. Use this when you need expert validation from multiple models "
            "with different perspectives."
        )

    def get_input_schema(self) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Description of what to get consensus on, testing objectives, and specific scope/focus areas. "
                        "Be as detailed as possible about the proposal, plan, or idea you want multiple perspectives on."
                    ),
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of models to consult for consensus. Supports stance specification using 'model:stance' format. "
                        "Stances: ':for' (supportive perspective), ':against' (critical perspective), or no stance (neutral). "
                        "Examples: ['o3:for', 'pro:against', 'grok-3'] creates a debate format. "
                        "Maximum 2 instances per model+stance combination."
                    ),
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional files or directories for additional context (must be absolute paths)",
                },
                "images": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional images showing expected UI changes, design requirements, "
                        "or visual references for the consensus analysis"
                    ),
                },
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific aspects to focus on (e.g., 'performance', 'security', 'user experience')",
                },
                "temperature": {
                    "type": "number",
                    "description": "Temperature (0-1, default 0.2 for consistency)",
                    "minimum": 0,
                    "maximum": 1,
                    "default": self.get_default_temperature(),
                },
                "thinking_mode": {
                    "type": "string",
                    "enum": ["minimal", "low", "medium", "high", "max"],
                    "description": (
                        "Thinking depth: minimal (0.5% of model max), low (8%), medium (33%), "
                        "high (67%), max (100% of model max)"
                    ),
                },
                "use_websearch": {
                    "type": "boolean",
                    "description": (
                        "Enable web search for documentation, best practices, and current information. "
                        "Particularly useful for: brainstorming sessions, architectural design discussions, "
                        "exploring industry best practices, working with specific frameworks/technologies, "
                        "researching solutions to complex problems, or when current documentation and "
                        "community insights would enhance the analysis."
                    ),
                    "default": True,
                },
                "continuation_id": {
                    "type": "string",
                    "description": (
                        "Thread continuation ID for multi-turn conversations. Can be used to continue "
                        "conversations across different tools. Only provide this if continuing a previous "
                        "conversation thread."
                    ),
                },
            },
            "required": ["prompt", "models"],
        }

        return schema

    def get_system_prompt(self) -> str:
        return CONSENSUS_PROMPT

    def get_default_temperature(self) -> float:
        return 0.2  # Lower temperature for more consistent consensus responses

    def get_model_category(self) -> "ToolModelCategory":
        """Consensus uses extended reasoning models for deep analysis"""
        from tools.models import ToolModelCategory

        return ToolModelCategory.EXTENDED_REASONING

    def get_request_model(self):
        return ConsensusRequest

    def _parse_model_and_stance(self, model_entry: str) -> tuple[str, str]:
        """Parse model entry like 'o3:against' into model and stance."""
        if ":" in model_entry:
            model_name, stance = model_entry.split(":", 1)
            model_name = model_name.strip()
            stance = stance.strip()

            # Check for empty model name
            if not model_name:
                return None, "model name cannot be empty"

            # Handle empty stance (treat as neutral)
            if not stance:
                stance = "neutral"

            # Validate stance - only allow for/against/neutral
            if stance not in {"for", "against", "neutral"}:
                return None, f"invalid stance '{stance}' (must be 'for', 'against', or omitted for neutral)"

            return model_name, stance

        # No colon - just model name
        model_name = model_entry.strip()
        if not model_name:
            return None, "model name cannot be empty"

        return model_name, "neutral"  # Default stance

    def _validate_model_combinations(self, model_entries: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
        """Validate model+stance combinations and enforce limits.

        Returns:
            tuple: (valid_combinations, skipped_entries)
            - Each combination can appear max 2 times
            - Same model+stance limited to 2 instances
        """
        valid_combinations = []
        skipped_entries = []
        combination_counts = {}  # Track (model, stance) -> count

        for entry in model_entries:
            result = self._parse_model_and_stance(entry)
            if result[0] is None:
                # Invalid stance - add to skipped entries
                skipped_entries.append(f"{entry} ({result[1]})")
                continue

            model_name, stance = result
            combination_key = (model_name, stance)

            current_count = combination_counts.get(combination_key, 0)
            if current_count >= DEFAULT_CONSENSUS_MAX_INSTANCES_PER_COMBINATION:
                # Already have max instances of this model+stance combination
                skipped_entries.append(
                    f"{model_name}:{stance} (max {DEFAULT_CONSENSUS_MAX_INSTANCES_PER_COMBINATION} instances)"
                )
                continue

            combination_counts[combination_key] = current_count + 1
            valid_combinations.append((model_name, stance))

        return valid_combinations, skipped_entries

    def _get_stance_enhanced_prompt(self, stance: str) -> str:
        """Get the system prompt with stance injection based on the stance."""
        base_prompt = self.get_system_prompt()

        stance_prompts = {
            "for": """Your perspective is SUPPORTIVE of the proposal. Focus on:
- Why this idea could work well
- Potential benefits and positive outcomes
- Creative solutions to overcome challenges
- Ways to maximize success and value
- Building upon the strengths of the approach

Be constructive and solution-oriented while still being realistic about challenges.""",
            "against": """Your perspective is CRITICAL of the proposal. Focus on:
- Potential problems and failure modes
- Risks, downsides, and negative consequences
- Why this approach might not work
- Missing considerations and overlooked issues
- Alternative approaches that might be better

Be skeptical and thorough in identifying issues while remaining constructive.""",
            "neutral": "No stance specified - provide balanced analysis considering both positive and negative aspects equally.",
        }

        stance_prompt = stance_prompts.get(stance, stance_prompts["neutral"])

        # Validate stance placeholder exists exactly once
        if base_prompt.count("{stance_prompt}") != 1:
            raise ValueError(
                "System prompt must contain exactly one '{stance_prompt}' placeholder, "
                f"found {base_prompt.count('{stance_prompt}')}"
            )

        # Inject stance into the system prompt
        return base_prompt.replace("{stance_prompt}", stance_prompt)

    async def _get_single_response(
        self, provider, model_name: str, stance: str, prompt: str, request: ConsensusRequest
    ) -> dict[str, Any]:
        """Get response from a single model with proper error handling and timeouts."""
        # Handle provider type safely - some providers return string, others return Enum
        ptype = provider.get_provider_type()
        provider_type = getattr(ptype, "value", ptype)
        semaphore = self._provider_semaphores.get(provider_type, self._default_semaphore)

        async with semaphore:  # Rate limiting per provider
            try:
                # Apply timeout to prevent hanging
                response = await asyncio.wait_for(
                    asyncio.to_thread(  # Best practice for wrapping sync functions
                        provider.generate_content,
                        prompt=prompt,
                        model_name=model_name,
                        system_prompt=self._get_stance_enhanced_prompt(stance),
                        temperature=getattr(request, "temperature", self.get_default_temperature()),
                        thinking_mode=getattr(request, "thinking_mode", "medium"),
                        images=getattr(request, "images", None) or [],
                    ),
                    timeout=DEFAULT_CONSENSUS_TIMEOUT,  # Configurable timeout per model
                )
                return {
                    "model": model_name,
                    "stance": stance,
                    "status": "success",
                    "verdict": response.content,  # Contains structured Markdown
                    "metadata": {
                        "provider": getattr(provider.get_provider_type(), "value", provider.get_provider_type()),
                        "usage": response.usage if hasattr(response, "usage") else None,
                    },
                }
            except asyncio.TimeoutError:
                return {
                    "model": model_name,
                    "stance": stance,
                    "status": "error",
                    "error": f"Request timed out after {DEFAULT_CONSENSUS_TIMEOUT} seconds",
                }
            except Exception as e:
                return {"model": model_name, "stance": stance, "status": "error", "error": str(e)}

    async def _get_consensus_responses(
        self, providers_and_models: list[tuple], prompt: str, request: ConsensusRequest
    ) -> list[dict[str, Any]]:
        """Execute all model requests concurrently with proper error handling."""

        # Create tasks for all model requests
        tasks = []
        for provider, model_name, stance in providers_and_models:
            task = self._get_single_response(provider, model_name, stance, prompt, request)
            tasks.append(task)

        # Execute all requests concurrently with proper error handling
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses and handle any gather-level exceptions
        processed_responses = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                # Extract model info from the original provider list for error reporting
                try:
                    _, model_name, stance = providers_and_models[i]
                    processed_responses.append(
                        {
                            "model": model_name,
                            "stance": stance,
                            "status": "error",
                            "error": f"Unhandled exception: {str(resp)}",
                        }
                    )
                except IndexError:
                    processed_responses.append(
                        {
                            "model": "unknown",
                            "stance": "unknown",
                            "status": "error",
                            "error": f"Unhandled exception: {str(resp)}",
                        }
                    )
            else:
                processed_responses.append(resp)

        return processed_responses

    def _format_consensus_output(self, responses: list[dict[str, Any]], skipped_entries: list[str]) -> str:
        """Format the consensus responses into structured output for Claude."""

        # Separate successful and failed responses
        successful_responses = [r for r in responses if r["status"] == "success"]
        failed_responses = [r for r in responses if r["status"] == "error"]

        # Prepare the structured output
        models_used = [
            f"{r['model']}:{r['stance']}" if r["stance"] != "neutral" else r["model"] for r in successful_responses
        ]
        models_errored = [
            f"{r['model']}:{r['stance']}" if r["stance"] != "neutral" else r["model"] for r in failed_responses
        ]

        output_data = {
            "status": "consensus_success" if successful_responses else "consensus_failed",
            "models_used": models_used,
            "models_skipped": skipped_entries,
            "models_errored": models_errored,
            "responses": responses,  # Include all responses for transparency
            "next_steps": self._get_synthesis_guidance(successful_responses, failed_responses),
        }

        return json.dumps(output_data, indent=2)

    def _get_synthesis_guidance(
        self, successful_responses: list[dict[str, Any]], failed_responses: list[dict[str, Any]]
    ) -> str:
        """Generate guidance for Claude on how to synthesize the consensus results."""

        if not successful_responses:
            return (
                "No models provided successful responses. Please retry with different models or "
                "check the error messages for guidance on resolving the issues."
            )

        if len(successful_responses) == 1:
            return (
                "Only one model provided a successful response. Synthesize based on the available "
                "perspective and indicate areas where additional expert input would be valuable "
                "due to the limited consensus data."
            )

        # Multiple successful responses - provide comprehensive synthesis guidance
        stance_counts = {"for": 0, "against": 0, "neutral": 0}
        for resp in successful_responses:
            stance = resp.get("stance", "neutral")
            stance_counts[stance] = stance_counts.get(stance, 0) + 1

        guidance = (
            "Claude, synthesize these perspectives by first identifying the key points of "
            "**agreement** and **disagreement** between the models. Then provide your final, "
            "consolidated recommendation, explaining how you weighed the different opinions and "
            "why your proposed solution is the most balanced approach. Explicitly address the "
            "most critical risks raised by each model and provide actionable next steps for implementation."
        )

        if failed_responses:
            guidance += (
                f" Note: {len(failed_responses)} model(s) failed to respond - consider this "
                "partial consensus and indicate where additional expert input would strengthen the analysis."
            )

        return guidance

    async def prepare_prompt(self, request: ConsensusRequest) -> str:
        """Prepare the consensus prompt with context files and focus areas."""
        # Check for prompt.txt in files
        prompt_content, updated_files = self.handle_prompt_file(request.files)

        # Use prompt.txt content if available, otherwise use the prompt field
        user_content = prompt_content if prompt_content else request.prompt

        # Check user input size at MCP transport boundary (before adding internal content)
        size_check = self.check_prompt_size(user_content)
        if size_check:
            # Need to return error, but prepare_prompt returns str
            # Use exception to handle this cleanly
            from tools.models import ToolOutput

            raise ValueError(f"MCP_SIZE_CHECK:{ToolOutput(**size_check).model_dump_json()}")

        # Update request files list
        if updated_files is not None:
            request.files = updated_files

        # Add focus areas if specified
        if request.focus_areas:
            focus_areas_text = "\n\nSpecific focus areas for this analysis:\n" + "\n".join(
                f"- {area}" for area in request.focus_areas
            )
            user_content += focus_areas_text

        # Add context files if provided (using centralized file handling with filtering)
        if request.files:
            file_content, processed_files = self._prepare_file_content_for_prompt(
                request.files, request.continuation_id, "Context files"
            )
            self._actually_processed_files = processed_files
            if file_content:
                user_content = f"{user_content}\n\n=== CONTEXT FILES ===\n{file_content}\n=== END CONTEXT ===="

        # Check token limits
        self._validate_token_limit(user_content, "Content")

        return user_content

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute consensus gathering from multiple models."""

        # Validate and create request
        request = ConsensusRequest(**arguments)

        # Handle conversation continuation if specified
        if request.continuation_id:
            conversation_context = await self.get_conversation_context(request.continuation_id)
            if conversation_context:
                # Add conversation context to the beginning of the prompt
                enhanced_prompt = f"{conversation_context}\n\n{request.prompt}"
                request.prompt = enhanced_prompt

        # Validate model+stance combinations and enforce limits
        valid_combinations, skipped_entries = self._validate_model_combinations(request.models)

        if not valid_combinations:
            error_output = {
                "status": "consensus_failed",
                "error": "No valid model combinations after validation",
                "models_skipped": skipped_entries,
                "next_steps": "Please provide valid model specifications. Use format 'model' or 'model:stance' where stance is 'for' or 'against'.",
            }
            return [TextContent(type="text", text=json.dumps(error_output, indent=2))]

        # Prepare the consensus prompt
        consensus_prompt = await self.prepare_prompt(request)

        # Get providers for valid model combinations
        providers_and_models = []
        for model_name, stance in valid_combinations:
            try:
                provider = self.get_model_provider(model_name)
                providers_and_models.append((provider, model_name, stance))
            except Exception as e:
                # Track failed models
                model_display = f"{model_name}:{stance}" if stance != "neutral" else model_name
                skipped_entries.append(f"{model_display} (provider not available: {str(e)})")

        if not providers_and_models:
            error_output = {
                "status": "consensus_failed",
                "error": "No model providers available",
                "models_skipped": skipped_entries,
                "next_steps": "Please check that the specified models have configured API keys and are available.",
            }
            return [TextContent(type="text", text=json.dumps(error_output, indent=2))]

        # Send to all models asynchronously
        responses = await self._get_consensus_responses(providers_and_models, consensus_prompt, request)

        # Enforce minimum success requirement - must have at least 1 successful response
        successful_responses = [r for r in responses if r["status"] == "success"]
        if not successful_responses:
            error_output = {
                "status": "consensus_failed",
                "error": "All model calls failed - no successful responses received",
                "models_skipped": skipped_entries,
                "models_errored": [
                    f"{r['model']}:{r['stance']}" if r["stance"] != "neutral" else r["model"]
                    for r in responses
                    if r["status"] == "error"
                ],
                "next_steps": "Please retry with different models or check the error messages for guidance on resolving the issues.",
            }
            return [TextContent(type="text", text=json.dumps(error_output, indent=2))]

        # Structure the output and store in conversation memory
        consensus_output = self._format_consensus_output(responses, skipped_entries)

        # Store in conversation memory if continuation_id is provided
        if request.continuation_id:
            await self.store_conversation_turn(
                request.continuation_id,
                consensus_output,
                request.files,
                request.images,
                responses,  # Store individual responses in metadata
                skipped_entries,
            )

        return [TextContent(type="text", text=consensus_output)]

    async def store_conversation_turn(
        self,
        continuation_id: str,
        output: str,
        files: list[str],
        images: list[str],
        responses: list[dict[str, Any]],
        skipped_entries: list[str],
    ):
        """Store consensus turn in conversation memory with special metadata."""
        from utils.conversation_memory import add_turn

        # Filter successful and failed responses
        successful_responses = [r for r in responses if r["status"] == "success"]
        failed_responses = [r for r in responses if r["status"] == "error"]

        # Prepare metadata for conversation storage
        metadata = {
            "tool_type": "consensus",
            "models_used": [r["model"] for r in successful_responses],
            "models_skipped": skipped_entries,
            "models_errored": [r["model"] for r in failed_responses],
            "individual_responses": successful_responses,  # Only store successful responses
        }

        # Store the turn with special consensus metadata
        await add_turn(
            thread_id=continuation_id,
            role="assistant",
            content=output,
            files=files or [],
            images=images or [],
            tool_name="consensus",
            model_provider="consensus",  # Special provider name
            model_name="consensus",  # Special model name
            model_metadata=metadata,
        )
