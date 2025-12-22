import base64
from collections import deque
from datetime import datetime
import mimetypes
import time
from typing import Any, Deque, Dict, List, Optional, Union
import uuid

from google.genai import errors

from rustic_ai.core import Agent
from rustic_ai.core.agents.commons import ErrorMessage
from rustic_ai.core.agents.commons.image_generation import ImageGenerationResponse
from rustic_ai.core.agents.commons.media import MediaLink
from rustic_ai.core.guild import agent
from rustic_ai.core.guild.agent_ext.depends.filesystem import FileSystem
from rustic_ai.core.guild.agent_ext.depends.llm.models import (
    ArrayOfContentParts,
    AssistantMessage,
    ChatCompletionError,
    ChatCompletionMessageToolCall,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionTool,
    Choice,
    FileContentPart,
    FinishReason,
    FunctionCall,
    ImageContentPart,
    ResponseCodes,
    SystemMessage,
    TextContentPart,
    ToolMessage,
    ToolType,
    UserMessage,
)
from rustic_ai.core.guild.dsl import BaseAgentProps
from rustic_ai.vertexai.client import VertexAIBase, VertexAIConf


class GeminiInteractionAgentProps(BaseAgentProps, VertexAIConf):
    """Configuration for Gemini Interaction Agent.

    Attributes:
        model_id: The Gemini model to use (e.g., "gemini-2.0-flash-exp")
        agent_id: The Gemini agent to use (e.g., "deep-research-pro-preview-12-2025")
                  Use either model_id or agent_id, not both
        temperature: Sampling temperature (0.0 to 2.0)
        max_output_tokens: Maximum number of tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        message_memory: Number of messages to keep in conversation history (0 = unlimited)
        system_instruction: Default system instruction for the model
        response_modalities: List of response modalities (e.g., ["TEXT"], ["IMAGE"], etc.)
        background: Run interactions in background mode (for long-running tasks)
        poll_interval: Polling interval in seconds for background mode (default: 10)
    """

    model_id: Optional[str] = "gemini-2.0-flash-exp"
    agent_id: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    message_memory: Optional[int] = 0
    system_instruction: Optional[str] = None
    response_modalities: Optional[List[str]] = None
    background: Optional[bool] = False
    poll_interval: Optional[int] = 10


class GeminiInteractionAgent(Agent[GeminiInteractionAgentProps], VertexAIBase):
    """A general-purpose Gemini interaction agent.

    This agent wraps the Gemini Interactions API, supporting:
    - Model mode (gemini-2.0-flash, gemini-2.5-flash, etc.)
    - Agent mode (deep-research-pro-preview, etc.)
    - Multimodal inputs (text, image, audio, video, PDF) via base64 encoding
    - Multimodal outputs (text, images) with automatic file saving
    - Tool/function calling
    - Background mode for long-running tasks with polling
    - Conversation memory and history management
    - Standard ChatCompletionRequest/Response format for framework compatibility
    """

    def __init__(self):
        """Initialize the Gemini Interaction Agent."""
        VertexAIBase.__init__(self, self.config.project_id, self.config.location)
        self.message_memory_size = self.config.message_memory or 0
        self.message_queue: Deque[Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]] = deque(
            maxlen=self.message_memory_size if self.message_memory_size > 0 else None
        )

    def _convert_to_interaction_input(
        self, messages: List[Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]]
    ) -> tuple[Union[str, List[Dict[str, Any]]], Optional[str]]:
        """Convert ChatCompletion messages to Gemini Interaction input format.

        Args:
            messages: List of ChatCompletion messages

        Returns:
            Tuple of (input string or list of input dicts, optional system instruction)
        """
        system_instruction: Optional[str] = None
        input_parts: List[Dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                if system_instruction is None:
                    system_instruction = msg.content
                continue

            if isinstance(msg, UserMessage):
                if isinstance(msg.content, str):
                    input_parts.append({"type": "text", "text": msg.content})
                elif isinstance(msg.content, ArrayOfContentParts):
                    for content_part in msg.content.root:
                        if isinstance(content_part, TextContentPart):
                            input_parts.append({"type": "text", "text": content_part.text})

                        elif isinstance(content_part, ImageContentPart):
                            image_url = content_part.image_url.url
                            if image_url.startswith("data:"):
                                mime_type = image_url.split(";")[0].split(":")[1]
                                base64_data = image_url.split(",", 1)[1]
                                input_parts.append({"type": "image", "data": base64_data, "mime_type": mime_type})
                            else:
                                self.logger.warning(f"Image URL {image_url} needs to be base64 encoded")

                        elif isinstance(content_part, FileContentPart):
                            file_url = content_part.file_url.url
                            if file_url.startswith("data:"):
                                mime_type = file_url.split(";")[0].split(":")[1]
                                base64_data = file_url.split(",", 1)[1]

                                if mime_type.startswith("audio/"):
                                    file_type = "audio"
                                elif mime_type.startswith("video/"):
                                    file_type = "video"
                                elif mime_type == "application/pdf":
                                    file_type = "document"
                                else:
                                    file_type = "document"

                                input_parts.append({"type": file_type, "data": base64_data, "mime_type": mime_type})
                            else:
                                self.logger.warning(f"File URL {file_url} needs to be base64 encoded")

            elif isinstance(msg, AssistantMessage):
                if msg.content:
                    input_parts.append({"type": "text", "text": f"Assistant: {msg.content}"})

            elif isinstance(msg, ToolMessage):
                input_parts.append({"type": "text", "text": f"Tool response: {msg.content}"})

        if len(input_parts) == 1 and input_parts[0]["type"] == "text":
            return input_parts[0]["text"], system_instruction
        return input_parts, system_instruction

    def _convert_tools(self, tools: List[ChatCompletionTool]) -> List[Dict[str, Any]]:
        """Convert ChatCompletionTool to Gemini tool format.

        Args:
            tools: List of ChatCompletionTool objects

        Returns:
            List of tool dictionaries
        """
        gemini_tools: List[Dict[str, Any]] = []

        for tool in tools:
            if tool.type == ToolType.function:
                gemini_tools.append(
                    {
                        "name": tool.function.name,
                        "description": tool.function.description or "",
                        "parameters": tool.function.parameters if tool.function.parameters else {},
                    }
                )

        return gemini_tools

    def _save_multimodal_outputs(
        self, interaction, guild_fs: FileSystem
    ) -> tuple[List[str], List[MediaLink], List[str]]:
        """Save multimodal outputs (images, etc.) to filesystem.

        Args:
            interaction: Gemini interaction response
            guild_fs: Filesystem for saving files

        Returns:
            Tuple of (text outputs list, media links list, errors list)
        """
        text_outputs: List[str] = []
        media_links: List[MediaLink] = []
        errors: List[str] = []

        if not hasattr(interaction, "outputs") or not interaction.outputs:
            return text_outputs, media_links, errors

        for output in interaction.outputs:
            if not hasattr(output, "type"):
                continue

            if output.type == "text" and hasattr(output, "text"):
                text_outputs.append(output.text)

            elif output.type == "image" and hasattr(output, "data"):
                # Decode and save base64 image
                try:
                    image_bytes = base64.b64decode(output.data)
                    ext = mimetypes.guess_extension(output.mime_type) or ".png"
                    filename = f"{uuid.uuid4()}{ext}"

                    with guild_fs.open(filename, "wb") as f:
                        f.write(image_bytes)

                    media_link = MediaLink(
                        url=filename,
                        name=filename,
                        mimetype=output.mime_type,
                        on_filesystem=True,
                    )
                    media_links.append(media_link)
                    text_outputs.append(f"[Generated image saved as: {filename}]")

                except Exception as e:
                    errors.append(f"Failed to save image: {str(e)}")

            elif hasattr(output, "text"):
                text_outputs.append(output.text)

        return text_outputs, media_links, errors

    def _convert_from_interaction_response(
        self, interaction, request_id: str, guild_fs: Optional[FileSystem] = None
    ) -> Union[ChatCompletionResponse, ImageGenerationResponse]:
        """Convert Gemini interaction response to appropriate response format.

        Args:
            interaction: Gemini interaction response
            request_id: Request ID for tracking
            guild_fs: Optional filesystem for saving multimodal outputs

        Returns:
            ChatCompletionResponse or ImageGenerationResponse
        """
        tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

        # Save multimodal outputs if filesystem is available
        if guild_fs:
            text_outputs, media_links, save_errors = self._save_multimodal_outputs(interaction, guild_fs)

            # If we have media outputs, return ImageGenerationResponse
            if media_links:
                return ImageGenerationResponse(
                    files=media_links,
                    errors=save_errors,
                    request=str({"interaction_id": interaction.id if hasattr(interaction, "id") else request_id}),
                )

            content_text = "\n".join(text_outputs) if text_outputs else None
        else:
            # No filesystem - just extract text
            content_text = None
            if hasattr(interaction, "outputs") and interaction.outputs:
                text_outputs = []
                for output in interaction.outputs:
                    if hasattr(output, "type") and output.type == "text" and hasattr(output, "text"):
                        text_outputs.append(output.text)
                    elif hasattr(output, "text"):
                        text_outputs.append(output.text)
                content_text = "\n".join(text_outputs) if text_outputs else None

            if not content_text and hasattr(interaction, "text"):
                content_text = interaction.text

        # Check for tool calls
        if hasattr(interaction, "tool_calls") and interaction.tool_calls:
            tool_calls = []
            for tool_call in interaction.tool_calls:
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type=ToolType.function,
                        function=FunctionCall(
                            name=tool_call.name if hasattr(tool_call, "name") else str(tool_call),
                            arguments=str(tool_call.args) if hasattr(tool_call, "args") else "{}",
                        ),
                    )
                )

        assistant_message = AssistantMessage(content=content_text, tool_calls=tool_calls)

        return ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            choices=[Choice(index=0, message=assistant_message, finish_reason=FinishReason.stop)],
            model=self.config.model_id or self.config.agent_id or "unknown",
            created=int(datetime.now().timestamp()),
        )

    def _poll_background_interaction(self, interaction_id: str) -> Any:
        """Poll for background interaction completion.

        Args:
            interaction_id: The interaction ID to poll

        Returns:
            Completed interaction object
        """
        poll_interval = self.config.poll_interval or 10
        self.logger.info(f"Background interaction started. ID: {interaction_id}")

        while True:
            interaction = self.genai_client.interactions.get(interaction_id)
            status = interaction.status if hasattr(interaction, "status") else "unknown"
            self.logger.info(f"Background interaction status: {status}")

            if status == "completed":
                return interaction
            elif status in ["failed", "cancelled"]:
                raise Exception(f"Background interaction {status}: {interaction_id}")

            time.sleep(poll_interval)

    @agent.processor(ChatCompletionRequest, depends_on=["filesystem:guild_fs:True"])
    def handle_chat_completion(
        self, ctx: agent.ProcessContext[ChatCompletionRequest], guild_fs: FileSystem
    ):
        """Handle ChatCompletionRequest and generate response using Gemini Interactions API.

        Args:
            ctx: Process context containing the ChatCompletionRequest
            guild_fs: Filesystem for saving multimodal outputs
        """
        try:
            payload = ctx.payload

            # Build conversation history
            all_messages: List[Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]] = (
                list(self.message_queue) + payload.messages
            )

            # Convert to Gemini interaction input
            interaction_input, extracted_system_instruction = self._convert_to_interaction_input(all_messages)

            # Build interaction parameters
            interaction_params: Dict[str, Any] = {"input": interaction_input}

            # Use either model or agent
            if self.config.agent_id:
                interaction_params["agent"] = self.config.agent_id
            else:
                interaction_params["model"] = self.config.model_id or "gemini-2.0-flash-exp"

            # Add system instruction
            system_instruction = extracted_system_instruction or self.config.system_instruction
            if system_instruction:
                interaction_params["system_instruction"] = system_instruction

            # Add generation config parameters
            if self.config.temperature is not None:
                interaction_params["temperature"] = self.config.temperature
            if self.config.max_output_tokens is not None:
                interaction_params["max_output_tokens"] = self.config.max_output_tokens
            if self.config.top_p is not None:
                interaction_params["top_p"] = self.config.top_p
            if self.config.top_k is not None:
                interaction_params["top_k"] = self.config.top_k

            # Add tools if present
            if payload.tools:
                gemini_tools = self._convert_tools(payload.tools)
                if gemini_tools:
                    interaction_params["tools"] = gemini_tools

            # Add response modalities if configured
            if self.config.response_modalities:
                interaction_params["response_modalities"] = self.config.response_modalities

            # Add background mode if configured
            if self.config.background:
                interaction_params["background"] = True

            # Call Gemini Interactions API
            interaction = self.genai_client.interactions.create(**interaction_params)

            # Poll for completion if background mode
            if self.config.background:
                interaction = self._poll_background_interaction(interaction.id)

            # Convert response
            response = self._convert_from_interaction_response(interaction, str(uuid.uuid4()), guild_fs)

            # Update message queue
            self.message_queue.extend(payload.messages)
            if isinstance(response, ChatCompletionResponse):
                if response.choices and response.choices[0].message:
                    self.message_queue.append(response.choices[0].message)

            # Send response
            ctx.send(response)

        except errors.APIError as e:
            self.logger.error(f"Gemini API error: {e.message}")
            ctx.send_error(
                ChatCompletionError(
                    status_code=ResponseCodes.INTERNAL_SERVER_ERROR,
                    message=f"Gemini API error: {e.message}",
                    response=None,
                    model=self.config.model_id or self.config.agent_id or "unknown",
                    request_messages=payload.messages if payload else [],
                )
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in Gemini interaction: {str(e)}")
            ctx.send_error(
                ErrorMessage(
                    agent_type=self.get_qualified_class_name(),
                    error_type="GeminiInteractionError",
                    error_message=str(e),
                )
            )
