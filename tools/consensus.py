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

from config import DEFAULT_CONSENSUS_MAX_INSTANCES_PER_COMBINATION
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
            "List of models to consult for consensus. Format: 'model' for neutral or 'model:stance' for positioned analysis. "
            "ONLY these stance words are supported - Supportive: 'for', 'support', 'favor'. Critical: 'against', 'oppose', 'critical'. "
            "Examples: 'o3:for', 'pro:support', 'grok:favor' (supportive); 'o3:against', 'pro:oppose', 'grok:critical' (critical); "
            "'o3', 'pro', 'grok-3' (neutral). Default to neutral unless user requests debate format or you've asked and they agreed. "
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

    def get_name(self) -> str:
        return "consensus"

    def get_description(self) -> str:
        return (
            "MULTI-MODEL CONSENSUS - Gather diverse perspectives from multiple AI models on technical proposals, "
            "plans, and ideas. Perfect for validation, feasibility assessment, and getting comprehensive "
            "viewpoints on complex decisions. Supports optional stance steering to create structured debates. "
            "Only apply stances when the user explicitly requests opposing views OR when you determine that "
            "contrasting perspectives would add significant value to the analysis. In such cases, consider "
            "asking the user: 'Would you like me to have one model argue strongly in favor and another "
            "against, to better explore the tradeoffs?' Use neutral stances by default unless there's clear "
            "benefit to debate format."
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
                        "List of models to consult for consensus. Format: 'model' for neutral or 'model:stance' for positioned analysis. "
                        "ONLY these stance words are supported - Supportive: 'for', 'support', 'favor'. Critical: 'against', 'oppose', 'critical'. "
                        "Examples: 'o3:for', 'pro:support', 'grok:favor' (supportive); 'o3:against', 'pro:oppose', 'grok:critical' (critical); "
                        "'o3', 'pro', 'grok-3' (neutral). Default to neutral unless user requests debate format or you've asked and they agreed. "
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

    def format_conversation_turn(self, turn) -> list[str]:
        """
        Format consensus turns with individual model responses for better readability.

        This custom formatting shows the individual model responses that were
        synthesized into the consensus, making it easier to understand the
        reasoning behind the final recommendation.
        """
        parts = []

        # Add files context if present
        if turn.files:
            parts.append(f"Files used in this turn: {', '.join(turn.files)}")
            parts.append("")

        # Check if this is a consensus turn with individual responses
        if turn.model_metadata and turn.model_metadata.get("individual_responses"):
            individual_responses = turn.model_metadata["individual_responses"]

            # Add consensus header
            models_consulted = []
            for resp in individual_responses:
                model = resp["model"]
                stance = resp.get("stance", "neutral")
                if stance != "neutral":
                    models_consulted.append(f"{model}:{stance}")
                else:
                    models_consulted.append(model)

            parts.append(f"Models consulted: {', '.join(models_consulted)}")
            parts.append("")
            parts.append("=== INDIVIDUAL MODEL RESPONSES ===")
            parts.append("")

            # Add each successful model response
            for i, response in enumerate(individual_responses):
                model_name = response["model"]
                stance = response.get("stance", "neutral")
                verdict = response["verdict"]

                stance_label = f"({stance.title()} Stance)" if stance != "neutral" else "(Neutral Analysis)"
                parts.append(f"**{model_name.upper()} {stance_label}**:")
                parts.append(verdict)

                if i < len(individual_responses) - 1:
                    parts.append("")
                    parts.append("---")
                parts.append("")

            parts.append("=== END INDIVIDUAL RESPONSES ===")
            parts.append("")
            parts.append("Claude's Synthesis:")

        # Add the actual content
        parts.append(turn.content)

        return parts

    def _parse_model_and_stance(self, model_entry: str) -> tuple[str, str]:
        """Parse model entry like 'o3:against' into model and stance."""
        # Import the generic parser from server
        from server import parse_model_option

        # Define stance synonyms
        supportive_stances = {"for", "support", "favor"}
        critical_stances = {"against", "oppose", "critical"}

        # Use the generic parser
        model_name, stance = parse_model_option(model_entry)

        if not model_name:
            return None, "model name cannot be empty"

        # If no stance provided, default to neutral
        if not stance:
            return model_name, "neutral"

        # Normalize stance to lowercase
        stance = stance.lower()

        # Map synonyms to canonical stance
        if stance in supportive_stances:
            stance = "for"
        elif stance in critical_stances:
            stance = "against"
        elif stance == "neutral":
            pass  # Already neutral
        else:
            return (
                None,
                f"invalid stance '{stance}' (must be one of: {', '.join(sorted(supportive_stances | critical_stances))}, or omitted for neutral)",
            )

        return model_name, stance

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
            "for": """SUPPORTIVE PERSPECTIVE WITH INTEGRITY

You are tasked with advocating FOR this proposal, but with CRITICAL GUARDRAILS:

MANDATORY ETHICAL CONSTRAINTS:
- This is NOT a debate for entertainment. You MUST act in good faith and in the best interest of the questioner
- You MUST think deeply about whether supporting this idea is safe, sound, and passes essential requirements
- You MUST be direct and unequivocal in saying "this is a bad idea" when it truly is
- There must be at least ONE COMPELLING reason to be optimistic, otherwise DO NOT support it

WHEN TO REFUSE SUPPORT (MUST OVERRIDE STANCE):
- If the idea is fundamentally harmful to users, project, or stakeholders
- If implementation would violate security, privacy, or ethical standards
- If the proposal is technically infeasible within realistic constraints
- If costs/risks dramatically outweigh any potential benefits

YOUR SUPPORTIVE ANALYSIS SHOULD:
- Identify genuine strengths and opportunities
- Propose solutions to overcome legitimate challenges
- Highlight synergies with existing systems
- Suggest optimizations that enhance value
- Present realistic implementation pathways

Remember: Being "for" means finding the BEST possible version of the idea IF it has merit, not blindly supporting bad ideas.""",
            "against": """CRITICAL PERSPECTIVE WITH RESPONSIBILITY

You are tasked with critiquing this proposal, but with ESSENTIAL BOUNDARIES:

MANDATORY FAIRNESS CONSTRAINTS:
- You MUST NOT oppose genuinely excellent, common-sense ideas just to be contrarian
- You MUST acknowledge when a proposal is fundamentally sound and well-conceived
- You CANNOT give harmful advice or recommend against beneficial changes
- If the idea is outstanding, say so clearly while offering constructive refinements

WHEN TO MODERATE CRITICISM (MUST OVERRIDE STANCE):
- If the proposal addresses critical user needs effectively
- If it follows established best practices with good reason
- If benefits clearly and substantially outweigh risks
- If it's the obvious right solution to the problem

YOUR CRITICAL ANALYSIS SHOULD:
- Identify legitimate risks and failure modes
- Point out overlooked complexities
- Suggest more efficient alternatives
- Highlight potential negative consequences
- Question assumptions that may be flawed

Remember: Being "against" means rigorous scrutiny to ensure quality, not undermining good ideas that deserve support.""",
            "neutral": """BALANCED PERSPECTIVE

Provide objective analysis considering both positive and negative aspects. However, if there is overwhelming evidence
that the proposal clearly leans toward being exceptionally good or particularly problematic, you MUST accurately
reflect this reality. Being "balanced" means being truthful about the weight of evidence, not artificially creating
50/50 splits when the reality is 90/10.

Your analysis should:
- Present all significant pros and cons discovered
- Weight them according to actual impact and likelihood
- If evidence strongly favors one conclusion, clearly state this
- Provide proportional coverage based on the strength of arguments
- Help the questioner see the true balance of considerations

Remember: Artificial balance that misrepresents reality is not helpful. True balance means accurate representation
of the evidence, even when it strongly points in one direction.""",
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

    def _get_single_response(
        self, provider, model_name: str, stance: str, prompt: str, request: ConsensusRequest
    ) -> dict[str, Any]:
        """Get response from a single model with MCP-safe synchronous processing."""
        logger.debug(f"Getting response from {model_name} with stance '{stance}'")

        try:
            # Direct synchronous call, no async complexity or threading
            # Rate limiting removed for simplicity - sequential processing provides natural rate limiting
            response = provider.generate_content(
                prompt=prompt,
                model_name=model_name,
                system_prompt=self._get_stance_enhanced_prompt(stance),
                temperature=getattr(request, "temperature", None) or self.get_default_temperature(),
                thinking_mode=getattr(request, "thinking_mode", "medium"),
                images=getattr(request, "images", None) or [],
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
        except Exception as e:
            logger.error(f"Error getting response from {model_name}:{stance}: {str(e)}")
            return {"model": model_name, "stance": stance, "status": "error", "error": str(e)}

    async def _get_consensus_responses(
        self, providers_and_models: list[tuple], prompt: str, request: ConsensusRequest
    ) -> list[dict[str, Any]]:
        """Execute all model requests with MCP-safe sequential processing."""

        responses = []

        # MCP-SAFE: Simple sequential processing without thread pools
        # This avoids event loop interference and is more reliable for MCP
        for provider, model_name, stance in providers_and_models:
            try:
                logger.debug(f"Processing {model_name}:{stance} sequentially")

                # Direct synchronous call - simple and MCP-safe
                # Sequential processing provides natural rate limiting
                response = self._get_single_response(provider, model_name, stance, prompt, request)
                responses.append(response)

                # Brief yield to allow MCP to process any pending messages
                # This is critical for MCP stability during long operations
                await asyncio.sleep(0.2)

            except Exception as e:
                logger.error(f"Failed to get response from {model_name}:{stance}: {str(e)}")
                responses.append(
                    {
                        "model": model_name,
                        "stance": stance,
                        "status": "error",
                        "error": f"Unhandled exception: {str(e)}",
                    }
                )

        return responses

    def _format_consensus_output(self, responses: list[dict[str, Any]], skipped_entries: list[str]) -> str:
        """Format the consensus responses into structured output for Claude."""

        logger.debug(f"Formatting consensus output for {len(responses)} responses")

        # Separate successful and failed responses
        successful_responses = [r for r in responses if r["status"] == "success"]
        failed_responses = [r for r in responses if r["status"] == "error"]

        logger.debug(f"Successful responses: {len(successful_responses)}, Failed: {len(failed_responses)}")

        # Prepare the structured output (minimize size for MCP stability)
        models_used = [
            f"{r['model']}:{r['stance']}" if r["stance"] != "neutral" else r["model"] for r in successful_responses
        ]
        models_errored = [
            f"{r['model']}:{r['stance']}" if r["stance"] != "neutral" else r["model"] for r in failed_responses
        ]

        # Prepare clean responses without truncation
        clean_responses = []
        for r in responses:
            if r["status"] == "success":
                clean_responses.append(
                    {
                        "model": r["model"],
                        "stance": r["stance"],
                        "status": r["status"],
                        "verdict": r.get("verdict", ""),
                        "metadata": r.get("metadata", {}),
                    }
                )
            else:
                clean_responses.append(
                    {
                        "model": r["model"],
                        "stance": r["stance"],
                        "status": r["status"],
                        "error": r.get("error", "Unknown error"),
                    }
                )

        output_data = {
            "status": "consensus_success" if successful_responses else "consensus_failed",
            "models_used": models_used,
            "models_skipped": skipped_entries,
            "models_errored": models_errored,
            "responses": clean_responses,
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

        # Store arguments for base class methods
        self._current_arguments = arguments

        # Check if we have pre-parsed models from server.py
        if "_parsed_models" in arguments:
            # Reconstruct the models list with stance format for validation
            reconstructed_models = []
            for pm in arguments["_parsed_models"]:
                model = pm["model"]
                option = pm["option"]
                if option:
                    reconstructed_models.append(f"{model}:{option}")
                else:
                    reconstructed_models.append(model)

            # Update arguments with reconstructed models
            arguments["models"] = reconstructed_models
            logger.debug(f"Using pre-parsed models from server: {reconstructed_models}")

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

        # Set up a dummy model context for consensus since we handle multiple models
        # This is needed for base class methods like prepare_prompt to work
        if not hasattr(self, "_model_context") or not self._model_context:
            from utils.model_context import ModelContext

            # Use the first model as the representative for token calculations
            first_model = valid_combinations[0][0] if valid_combinations else "flash"
            self._model_context = ModelContext(first_model)

        # Prepare the consensus prompt
        consensus_prompt = await self.prepare_prompt(request)

        # Get providers for valid model combinations with caching to avoid duplicate lookups
        providers_and_models = []
        provider_cache = {}  # Cache to avoid duplicate provider lookups

        for model_name, stance in valid_combinations:
            try:
                # Check cache first
                if model_name in provider_cache:
                    provider = provider_cache[model_name]
                else:
                    # Look up provider and cache it
                    provider = self.get_model_provider(model_name)
                    provider_cache[model_name] = provider

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
        logger.debug(f"Sending consensus request to {len(providers_and_models)} models")
        responses = await self._get_consensus_responses(providers_and_models, consensus_prompt, request)
        logger.debug(f"Received {len(responses)} responses from consensus models")

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

        # Yield before formatting to give MCP breathing room
        await asyncio.sleep(0.1)

        # Structure the output and store in conversation memory
        consensus_output = self._format_consensus_output(responses, skipped_entries)

        # Log response size for debugging
        output_size = len(consensus_output)
        logger.debug(f"Consensus output size: {output_size:,} characters")

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
