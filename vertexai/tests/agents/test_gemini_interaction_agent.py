import base64
import os
from pathlib import Path

import pytest

from rustic_ai.core.agents.commons.image_generation import ImageGenerationResponse
from rustic_ai.core.guild.agent_ext.depends.filesystem import FileSystemResolver
from rustic_ai.core.guild.agent_ext.depends.llm.models import (
    ArrayOfContentParts,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionTool,
    Function,
    ImageContentPart,
    ImageUrl,
    TextContentPart,
    ToolType,
    UserMessage,
)
from rustic_ai.core.guild.builders import AgentBuilder
from rustic_ai.core.guild.dsl import DependencySpec
from rustic_ai.core.messaging.core.message import AgentTag, Message
from rustic_ai.core.utils.basic_class_utils import get_qualified_class_name
from rustic_ai.core.utils.gemstone_id import GemstoneGenerator
from rustic_ai.core.utils.priority import Priority
from rustic_ai.testing.helpers import wrap_agent_for_testing
from rustic_ai.vertexai.agents.gemini_interaction import (
    GeminiInteractionAgent,
    GeminiInteractionAgentProps,
)


@pytest.fixture
def dependency_map(tmp_path):
    """Create dependency map with filesystem for tests."""
    base = Path(tmp_path) / "gemini_agent_tests"
    return {
        "filesystem": DependencySpec(
            class_name=FileSystemResolver.get_qualified_class_name(),
            properties={
                "path_base": str(base),
                "protocol": "file",
                "storage_options": {"auto_mkdir": True},
            },
        )
    }


@pytest.fixture
def generator():
    """Create ID generator for messages."""
    return GemstoneGenerator(machine_id=1)


def _build_message(generator, payload, fmt):
    """Helper to build a test message."""
    return Message(
        id_obj=generator.get_id(Priority.NORMAL),
        topics="default_topic",
        sender=AgentTag(id="tester", name="tester"),
        payload=payload,
        format=fmt,
    )


class TestGeminiInteractionAgent:
    @pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS") == "true", reason="Skipping expensive tests")
    def test_basic_chat_completion(self, dependency_map, generator):
        """Test basic text completion with Gemini."""
        # Arrange
        agent_spec = (
            AgentBuilder(GeminiInteractionAgent)
            .set_id("gemini_agent")
            .set_name("Test Gemini Agent")
            .set_description("An agent for testing Gemini interaction")
            .build_spec()
        )

        agent, results = wrap_agent_for_testing(agent_spec, dependency_map=dependency_map)

        # Act
        request = ChatCompletionRequest(
            messages=[UserMessage(content="Say 'Hello, World!' and nothing else.")],
        )

        message = _build_message(generator, request.model_dump(), get_qualified_class_name(ChatCompletionRequest))
        agent._on_message(message)

        # Assert
        assert len(results) > 0
        assert results[0].in_response_to == message.id

        response = ChatCompletionResponse.model_validate(results[0].payload)
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS") == "true", reason="Skipping expensive tests")
    def test_multimodal_input(self, dependency_map, generator):
        """Test multimodal input with text and image (base64)."""
        # Arrange
        agent_spec = (
            AgentBuilder(GeminiInteractionAgent)
            .set_id("gemini_agent")
            .set_name("Test Gemini Agent")
            .set_description("An agent for testing Gemini interaction")
            .build_spec()
        )

        agent, results = wrap_agent_for_testing(agent_spec, dependency_map=dependency_map)

        # Act - Send text + image (using base64 data URL)
        # Create a small test image in base64 (1x1 red pixel PNG)
        tiny_png = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
            b"\xc0\x00\x00\x00\x03\x00\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        ).decode("utf-8")
        test_image_data_url = f"data:image/png;base64,{tiny_png}"

        request = ChatCompletionRequest(
            messages=[
                UserMessage(
                    content=ArrayOfContentParts(
                        root=[
                            TextContentPart(text="Describe this image."),
                            ImageContentPart(image_url=ImageUrl(url=test_image_data_url)),
                        ]
                    )
                )
            ],
        )

        message = _build_message(generator, request.model_dump(), get_qualified_class_name(ChatCompletionRequest))
        agent._on_message(message)

        # Assert
        assert len(results) > 0
        response = ChatCompletionResponse.model_validate(results[0].payload)
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    @pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS") == "true", reason="Skipping expensive tests")
    def test_tool_calling(self, dependency_map, generator):
        """Test tool/function calling capability."""
        # Arrange
        agent_spec = (
            AgentBuilder(GeminiInteractionAgent)
            .set_id("gemini_agent")
            .set_name("Test Gemini Agent")
            .set_description("An agent for testing Gemini interaction")
            .build_spec()
        )

        agent, results = wrap_agent_for_testing(agent_spec, dependency_map=dependency_map)

        # Define a simple tool
        get_weather_tool = ChatCompletionTool(
            type=ToolType.function,
            function=Function(
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            ),
        )

        # Act
        request = ChatCompletionRequest(
            messages=[UserMessage(content="What's the weather like in San Francisco?")],
            tools=[get_weather_tool],
        )

        message = _build_message(generator, request.model_dump(), get_qualified_class_name(ChatCompletionRequest))
        agent._on_message(message)

        # Assert
        assert len(results) > 0
        response = ChatCompletionResponse.model_validate(results[0].payload)
        assert response.choices is not None
        assert len(response.choices) > 0

        # Check if tool_calls were made (optional - model may or may not call the tool)
        message_data = response.choices[0].message
        if message_data.tool_calls:
            assert message_data.tool_calls[0].function.name == "get_weather"

    @pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS") == "true", reason="Skipping expensive tests")
    def test_conversation_memory(self, dependency_map, generator):
        """Test conversation memory across multiple messages."""
        # Arrange - Set message_memory to keep last 4 messages
        agent_props = GeminiInteractionAgentProps(message_memory=4)

        agent_spec = (
            AgentBuilder(GeminiInteractionAgent)
            .set_id("gemini_agent")
            .set_name("Test Gemini Agent")
            .set_description("An agent for testing Gemini interaction")
            .set_properties(agent_props)
            .build_spec()
        )

        agent, results = wrap_agent_for_testing(agent_spec, dependency_map=dependency_map)

        # Act - Send first message
        request1 = ChatCompletionRequest(
            messages=[UserMessage(content="My name is Alice. Remember this.")],
        )

        message1 = _build_message(generator, request1.model_dump(), get_qualified_class_name(ChatCompletionRequest))
        agent._on_message(message1)

        # Send second message asking to recall
        request2 = ChatCompletionRequest(
            messages=[UserMessage(content="What is my name?")],
        )

        message2 = _build_message(generator, request2.model_dump(), get_qualified_class_name(ChatCompletionRequest))
        agent._on_message(message2)

        # Assert
        assert len(results) == 2

        # Check second response mentions "Alice"
        second_response = ChatCompletionResponse.model_validate(results[1].payload)
        assert second_response.choices is not None
        assert len(second_response.choices) > 0
        content = second_response.choices[0].message.content
        assert content is not None
        # The model should remember the name from previous context
        assert "alice" in content.lower()

    @pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS") == "true", reason="Skipping expensive tests")
    def test_agent_mode_deep_research(self, dependency_map, generator):
        """Test using agent mode (deep-research-pro-preview) instead of model."""
        # Arrange - Use agent_id instead of model_id
        agent_props = GeminiInteractionAgentProps(
            agent_id="deep-research-pro-preview-12-2025", background=True, poll_interval=5
        )

        agent_spec = (
            AgentBuilder(GeminiInteractionAgent)
            .set_id("gemini_agent")
            .set_name("Test Gemini Agent")
            .set_description("An agent for testing Gemini interaction")
            .set_properties(agent_props)
            .build_spec()
        )

        agent, results = wrap_agent_for_testing(agent_spec, dependency_map=dependency_map)

        # Act
        request = ChatCompletionRequest(
            messages=[UserMessage(content="What are the key features of Google TPUs in 2025?")],
        )

        message = _build_message(generator, request.model_dump(), get_qualified_class_name(ChatCompletionRequest))
        agent._on_message(message)

        # Assert
        assert len(results) > 0
        response = ChatCompletionResponse.model_validate(results[0].payload)
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS") == "true", reason="Skipping expensive tests")
    def test_image_generation_output(self, dependency_map, generator, tmp_path):
        """Test multimodal output with image generation."""
        # Arrange - Configure for image generation
        agent_props = GeminiInteractionAgentProps(
            model_id="gemini-3-pro-image-preview", response_modalities=["IMAGE"]
        )

        agent_spec = (
            AgentBuilder(GeminiInteractionAgent)
            .set_id("gemini_agent")
            .set_name("Test Gemini Agent")
            .set_description("An agent for testing Gemini interaction")
            .set_properties(agent_props)
            .build_spec()
        )

        agent, results = wrap_agent_for_testing(agent_spec, dependency_map=dependency_map)

        # Act
        request = ChatCompletionRequest(
            messages=[UserMessage(content="Generate an image of a futuristic city.")],
        )

        message = _build_message(generator, request.model_dump(), get_qualified_class_name(ChatCompletionRequest))
        agent._on_message(message)

        # Assert
        assert len(results) > 0

        # Check if we got ImageGenerationResponse (for image outputs)
        if results[0].format == get_qualified_class_name(ImageGenerationResponse):
            response = ImageGenerationResponse.model_validate(results[0].payload)
            assert response.files is not None
            assert len(response.files) > 0
            # Verify the file was saved
            assert response.files[0].on_filesystem is True
        else:
            # Or ChatCompletionResponse with text about the generated image
            response = ChatCompletionResponse.model_validate(results[0].payload)
            assert response.choices is not None
            assert len(response.choices) > 0
