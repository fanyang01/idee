import pytest
import json
import uuid
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import base64

from idee.tools.base import (
    ToolResult, 
    TextBlock, 
    ImageBlock, 
    AudioBlock, 
    VideoBlock, 
    DocumentBlock,
    FileBlock,
    ToolOutputPlaceholder
)
from idee.agents.types import UnifiedToolResult
from idee.agents.gpt import GPTAgent
from idee.agents.gemini import GeminiAgent
from idee.agents.claude import ClaudeAgent
from idee.agents.types import OpenAIAgentConfig, GeminiAgentConfig, ClaudeAgentConfig


class TestToolResultToUnifiedToolResult:
    """Test conversion from ToolResult to UnifiedToolResult"""
    
    @pytest.mark.asyncio
    async def test_text_only_tool_result(self):
        """Test simple text-only tool result conversion"""
        tool_result = ToolResult(content=[TextBlock(text="Hello world")])
        unified = UnifiedToolResult(
            call_id="test_call_123",
            tool_name="test_tool",
            tool_output=tool_result
        )
        
        assert unified.get_text_content() == "Hello world"
        assert not unified.has_multimodal_content()
        
    @pytest.mark.asyncio
    async def test_multimodal_tool_result(self):
        """Test multimodal tool result conversion"""
        tool_result = ToolResult(content=[
            TextBlock(text="Analysis complete. "),
            ImageBlock.from_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77TwAAAABJRU5ErkJggg==", "image/png"),
            TextBlock(text=" Results attached.")
        ])
        unified = UnifiedToolResult(
            call_id="test_call_123", 
            tool_name="test_tool",
            tool_output=tool_result
        )
        
        assert unified.get_text_content() == "Analysis complete.  Results attached."
        assert unified.has_multimodal_content()
        assert len(unified.get_images()) == 1
        assert unified.get_images()[0]["type"] == "image"
        
    @pytest.mark.asyncio
    async def test_file_block_tool_result(self):
        """Test tool result with file blocks"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            temp_path = f.name
        
        try:
            tool_result = ToolResult(content=[
                TextBlock(text="File created: "),
                FileBlock.from_path(temp_path, "text/plain", "test.txt")
            ])
            unified = UnifiedToolResult(
                call_id="test_call_123",
                tool_name="test_tool", 
                tool_output=tool_result
            )
            
            assert unified.has_multimodal_content()
            assert len(unified.tool_output.get_files()) == 1
            file_block = unified.tool_output.get_files()[0]
            assert file_block.filename == "test.txt"
            assert file_block.media_type == "text/plain"
        finally:
            Path(temp_path).unlink()
            
    @pytest.mark.asyncio
    async def test_placeholder_formatting(self):
        """Test placeholder formatting for non-Claude models"""
        tool_result = ToolResult(content=[
            TextBlock(text="Generated image: "),
            ImageBlock.from_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77TwAAAABJRU5ErkJggg==", "image/png")
        ])
        unified = UnifiedToolResult(
            call_id="test_call_123",
            tool_name="test_tool",
            tool_output=tool_result
        )
        
        placeholder_text, follow_up_content = unified.to_simple_format_with_placeholders()
        
        assert "multimedia content" in placeholder_text
        assert "test_tool" in placeholder_text
        assert follow_up_content is not None
        assert follow_up_content["tool_name"] == "test_tool"
        assert len(follow_up_content["full_content"]) == 2


class TestGPTAgentFormatToolResults:
    """Test GPT agent's _format_tool_results method"""
    
    @pytest.fixture
    def gpt_agent(self):
        config = OpenAIAgentConfig(
            model="gpt-4",
            api_key="test_key",
            use_responses_api=False
        )
        return GPTAgent(config, tools=[])
    
    @pytest.fixture
    def gpt_agent_responses_api(self):
        config = OpenAIAgentConfig(
            model="gpt-4",
            api_key="test_key", 
            use_responses_api=True
        )
        return GPTAgent(config, tools=[])
    
    @pytest.mark.asyncio
    async def test_simple_text_formatting_chat_api(self, gpt_agent):
        """Test simple text formatting for Chat Completions API"""
        tool_result = ToolResult(content=[TextBlock(text="Success")])
        unified = UnifiedToolResult(
            call_id="call_123",
            tool_name="test_tool",
            tool_output=tool_result
        )
        
        formatted_results, follow_up_content = await gpt_agent._format_tool_results([unified])
        
        assert len(formatted_results) == 1
        assert formatted_results[0]["role"] == "tool"
        assert formatted_results[0]["tool_call_id"] == "call_123"
        assert formatted_results[0]["content"] == "Success"
        assert len(follow_up_content) == 0
        
    @pytest.mark.asyncio
    async def test_simple_text_formatting_responses_api(self, gpt_agent_responses_api):
        """Test simple text formatting for Responses API"""
        tool_result = ToolResult(content=[TextBlock(text="Success")])
        unified = UnifiedToolResult(
            call_id="call_123",
            tool_name="test_tool",
            tool_output=tool_result
        )
        
        formatted_results, follow_up_content = await gpt_agent_responses_api._format_tool_results([unified])
        
        assert len(formatted_results) == 1
        assert formatted_results[0]["type"] == "function_call_output"
        assert formatted_results[0]["call_id"] == "call_123"
        assert formatted_results[0]["output"] == "Success"
        assert len(follow_up_content) == 0
        
    @pytest.mark.asyncio
    async def test_multimodal_formatting_with_placeholders(self, gpt_agent):
        """Test multimodal content formatting with placeholders"""
        tool_result = ToolResult(content=[
            TextBlock(text="Image generated: "),
            ImageBlock.from_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77TwAAAABJRU5ErkJggg==", "image/png")
        ])
        unified = UnifiedToolResult(
            call_id="call_123",
            tool_name="test_tool",
            tool_output=tool_result
        )
        
        formatted_results, follow_up_content = await gpt_agent._format_tool_results([unified])
        
        assert len(formatted_results) == 1
        assert "multimedia content" in formatted_results[0]["content"]
        assert "test_tool" in formatted_results[0]["content"]
        assert len(follow_up_content) == 1
        assert follow_up_content[0]["tool_name"] == "test_tool"


class TestGPTAgentMultimodalProcessing:
    """Test GPT agent's multimodal content processing methods"""
    
    @pytest.fixture
    def gpt_agent_chat(self):
        config = OpenAIAgentConfig(
            model="gpt-4",
            api_key="test_key",
            use_responses_api=False
        )
        return GPTAgent(config, tools=[])
    
    @pytest.fixture
    def gpt_agent_responses(self):
        config = OpenAIAgentConfig(
            model="gpt-4",
            api_key="test_key",
            use_responses_api=True
        )
        return GPTAgent(config, tools=[])
    
    @pytest.mark.asyncio
    async def test_process_multimodal_content_blocks_chat(self, gpt_agent_chat):
        """Test multimodal content processing for Chat Completions API"""
        content_blocks = [
            TextBlock(text="Here is the result: "),
            ImageBlock.from_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77TwAAAABJRU5ErkJggg==", "image/png")
        ]
        
        with patch.object(gpt_agent_chat, '_upload_file', return_value=None):
            result = await gpt_agent_chat._process_multimodal_content_blocks_chat(content_blocks)
        
        assert len(result) == 2
        
        # First should be text
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Here is the result: "
        
        # Second should be image
        assert result[1]["type"] == "image"
        assert result[1]["image_url"]["url"].startswith("data:image/png;base64,")
        
    @pytest.mark.asyncio
    async def test_process_multimodal_content_blocks_responses(self, gpt_agent_responses):
        """Test multimodal content processing for Responses API"""
        content_blocks = [
            TextBlock(text="Here is the result: "),
            ImageBlock.from_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77TwAAAABJRU5ErkJggg==", "image/png")
        ]
        
        with patch.object(gpt_agent_responses, '_upload_file', return_value=None):
            result = await gpt_agent_responses._process_multimodal_content_blocks_responses(content_blocks)
        
        assert len(result) == 2
        
        # First should be text
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == "Here is the result: "
        
        # Second should be image
        assert result[1]["type"] == "input_image"
        assert result[1]["image_url"].startswith("data:image/png;base64,")
        
    @pytest.mark.asyncio
    async def test_file_block_processing_chat(self, gpt_agent_chat):
        """Test file block processing for Chat Completions API"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            temp_path = f.name
        
        try:
            content_blocks = [
                TextBlock(text="File created: "),
                FileBlock.from_path(temp_path, "text/plain", "test.txt")
            ]
            
            with patch.object(gpt_agent_chat, '_upload_file', return_value="file_123"):
                result = await gpt_agent_chat._process_multimodal_content_blocks_chat(content_blocks)
            
            assert len(result) == 2
            assert result[0]["type"] == "text"
            assert result[1]["type"] == "file"
            assert result[1]["file"]["file_id"] == "file_123"
            
        finally:
            Path(temp_path).unlink()
            
    @pytest.mark.asyncio
    async def test_file_block_processing_responses(self, gpt_agent_responses):
        """Test file block processing for Responses API"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            temp_path = f.name
        
        try:
            content_blocks = [
                TextBlock(text="File created: "),
                FileBlock.from_path(temp_path, "text/plain", "test.txt")
            ]
            
            with patch.object(gpt_agent_responses, '_upload_file', return_value="file_123"):
                result = await gpt_agent_responses._process_multimodal_content_blocks_responses(content_blocks)
            
            assert len(result) == 2
            assert result[0]["type"] == "input_text"
            assert result[1]["type"] == "input_file"
            assert result[1]["file_id"] == "file_123"
            
        finally:
            Path(temp_path).unlink()
            
    @pytest.mark.asyncio
    async def test_image_file_fallback_chat(self, gpt_agent_chat):
        """Test image file fallback to base64 for Chat API when upload fails"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            # Write minimal PNG data
            f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77TwAAAABJRU5ErkJggg=="))
            temp_path = f.name
        
        try:
            content_blocks = [
                FileBlock.from_path(temp_path, "image/png", "test.png")
            ]
            
            with patch.object(gpt_agent_chat, '_upload_file', return_value=None):
                result = await gpt_agent_chat._process_multimodal_content_blocks_chat(content_blocks)
            
            assert len(result) == 1
            assert result[0]["type"] == "image"
            assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")
            
        finally:
            Path(temp_path).unlink()
            
    @pytest.mark.asyncio
    async def test_pdf_file_fallback_responses(self, gpt_agent_responses):
        """Test PDF file fallback to base64 for Responses API when upload fails"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b"%PDF-1.4\ntest content")
            temp_path = f.name
        
        try:
            content_blocks = [
                FileBlock.from_path(temp_path, "application/pdf", "test.pdf")
            ]
            
            with patch.object(gpt_agent_responses, '_upload_file', return_value=None):
                result = await gpt_agent_responses._process_multimodal_content_blocks_responses(content_blocks)
            
            assert len(result) == 1
            assert result[0]["type"] == "input_file"
            assert result[0]["filename"] == "test.pdf"
            assert result[0]["file_data"].startswith("data:application/pdf;base64,")
            
        finally:
            Path(temp_path).unlink()
            
    @pytest.mark.asyncio
    async def test_unsupported_file_type_error(self, gpt_agent_chat):
        """Test error handling for unsupported file types"""
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as f:
            f.write(b"unknown content")
            temp_path = f.name
        
        try:
            content_blocks = [
                FileBlock.from_path(temp_path, "application/unknown", "test.unknown")
            ]
            
            with patch.object(gpt_agent_chat, '_upload_file', return_value=None):
                with pytest.raises(ValueError, match="Unsupported file type"):
                    await gpt_agent_chat._process_multimodal_content_blocks_chat(content_blocks)
            
        finally:
            Path(temp_path).unlink()
            
    @pytest.mark.asyncio
    async def test_append_follow_up_content_chat(self, gpt_agent_chat):
        """Test follow-up content appending for Chat Completions API"""
        current_messages = []
        follow_up_content = [
            {
                "tool_name": "test_tool",
                "id": "placeholder_123",
                "full_content": [
                    TextBlock(text="Result: "),
                    ImageBlock.from_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77TwAAAABJRU5ErkJggg==", "image/png")
                ]
            }
        ]
        
        with patch.object(gpt_agent_chat, '_upload_file', return_value=None):
            await gpt_agent_chat._append_follow_up_content(current_messages, follow_up_content)
        
        assert len(current_messages) == 1
        message = current_messages[0]
        assert message["role"] == "user"
        assert isinstance(message["content"], list)
        
        # Should have opening tag, content, closing tag
        content_parts = message["content"]
        assert len(content_parts) >= 4  # At least opening, text, image, closing
        
        # Check for ToolOutput wrapper
        assert any('<ToolOutput name="test_tool"' in part.get("text", "") for part in content_parts)
        assert any('</ToolOutput>' in part.get("text", "") for part in content_parts)
        
    @pytest.mark.asyncio
    async def test_append_follow_up_content_responses(self, gpt_agent_responses):
        """Test follow-up content appending for Responses API"""
        current_messages = []
        follow_up_content = [
            {
                "tool_name": "test_tool",
                "id": "placeholder_123",
                "full_content": [
                    TextBlock(text="Result: "),
                    ImageBlock.from_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77TwAAAABJRU5ErkJggg==", "image/png")
                ]
            }
        ]
        
        with patch.object(gpt_agent_responses, '_upload_file', return_value=None):
            await gpt_agent_responses._append_follow_up_content(current_messages, follow_up_content)
        
        assert len(current_messages) == 1
        message = current_messages[0]
        assert message["role"] == "user"
        assert isinstance(message["content"], list)
        
        # Should have opening tag, content, closing tag
        content_parts = message["content"]
        assert len(content_parts) >= 4  # At least opening, text, image, closing
        
        # Check for ToolOutput wrapper and input_text type
        assert any('<ToolOutput name="test_tool"' in part.get("text", "") for part in content_parts)
        assert any(part.get("type") == "input_text" for part in content_parts)
        
    @pytest.mark.asyncio
    async def test_multiple_follow_up_content(self, gpt_agent_chat):
        """Test multiple follow-up content items in single message"""
        current_messages = []
        follow_up_content = [
            {
                "tool_name": "tool_1",
                "id": "id_1",
                "full_content": [TextBlock(text="First result")]
            },
            {
                "tool_name": "tool_2", 
                "id": "id_2",
                "full_content": [TextBlock(text="Second result")]
            }
        ]
        
        await gpt_agent_chat._append_follow_up_content(current_messages, follow_up_content)
        
        assert len(current_messages) == 1
        message = current_messages[0]
        content_text = " ".join(part.get("text", "") for part in message["content"])
        
        # Should contain both tool outputs
        assert 'tool_1' in content_text
        assert 'tool_2' in content_text
        assert 'First result' in content_text
        assert 'Second result' in content_text


class TestGPTAgentFileUpload:
    """Test GPT agent's file upload functionality"""
    
    @pytest.fixture
    def gpt_agent(self):
        config = OpenAIAgentConfig(
            model="gpt-4",
            api_key="test_key",
            use_responses_api=False
        )
        return GPTAgent(config, tools=[])
    
    @pytest.mark.asyncio
    async def test_upload_file_from_path(self, gpt_agent):
        """Test file upload from file path"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            temp_path = f.name
        
        try:
            file_block = FileBlock.from_path(temp_path, "text/plain", "test.txt")
            
            # Mock the async client's file upload
            mock_response = Mock()
            mock_response.id = "file_123"
            
            with patch.object(gpt_agent.client.files, 'create', return_value=mock_response) as mock_create:
                result = await gpt_agent._upload_file(file_block)
                
                assert result == "file_123"
                mock_create.assert_called_once()
                
        finally:
            Path(temp_path).unlink()
    
    @pytest.mark.asyncio
    async def test_upload_file_from_content(self, gpt_agent):
        """Test file upload from bytes content"""
        file_content = b"Test file content"
        file_block = FileBlock.from_content(file_content, "text/plain", "test.txt")
        
        # Mock the async client's file upload
        mock_response = Mock()
        mock_response.id = "file_456"
        
        with patch.object(gpt_agent.client.files, 'create', return_value=mock_response) as mock_create:
            result = await gpt_agent._upload_file(file_block)
            
            assert result == "file_456"
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upload_file_failure(self, gpt_agent):
        """Test file upload failure handling"""
        file_block = FileBlock.from_content(b"test", "text/plain", "test.txt")
        
        # Mock the async client to raise an exception
        with patch.object(gpt_agent.client.files, 'create', side_effect=Exception("Upload failed")):
            result = await gpt_agent._upload_file(file_block)
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_upload_file_no_client(self, gpt_agent):
        """Test file upload when client is not available"""
        file_block = FileBlock.from_content(b"test", "text/plain", "test.txt")
        
        # Remove client
        gpt_agent.client = None
        
        result = await gpt_agent._upload_file(file_block)
        assert result is None
        
    @pytest.mark.asyncio
    async def test_upload_file_no_files_attribute(self, gpt_agent):
        """Test file upload when client has no files attribute"""
        file_block = FileBlock.from_content(b"test", "text/plain", "test.txt")
        
        # Mock client without files attribute
        mock_client = Mock()
        del mock_client.files  # Remove files attribute
        gpt_agent.client = mock_client
        
        result = await gpt_agent._upload_file(file_block)
        assert result is None
        

class TestGeminiAgentFormatToolResults:
    """Test Gemini agent's _format_tool_results method"""
    
    @pytest.fixture
    def gemini_agent(self):
        config = GeminiAgentConfig(
            model="gemini-1.5-pro",
            api_key="test_key"
        )
        return GeminiAgent(config, tools=[])
    
    @pytest.mark.asyncio
    async def test_simple_text_formatting(self, gemini_agent):
        """Test simple text formatting for Gemini"""
        tool_result = ToolResult(content=[TextBlock(text="Success")])
        unified = UnifiedToolResult(
            call_id="call_123",
            tool_name="test_tool",
            tool_output=tool_result
        )
        
        parts, follow_up_content = await gemini_agent._format_tool_results([unified])
        
        assert len(parts) == 1
        # Check that it's a Part with FunctionResponse
        part = parts[0]
        assert hasattr(part, 'function_response')
        assert part.function_response.name == "test_tool"
        assert part.function_response.response == {"result": "Success"}
        assert len(follow_up_content) == 0
        
    @pytest.mark.asyncio
    async def test_multimodal_formatting_with_placeholders(self, gemini_agent):
        """Test multimodal content formatting with placeholders"""
        tool_result = ToolResult(content=[
            TextBlock(text="Image generated: "),
            ImageBlock.from_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77TwAAAABJRU5ErkJggg==", "image/png")
        ])
        unified = UnifiedToolResult(
            call_id="call_123",
            tool_name="test_tool",
            tool_output=tool_result
        )
        
        parts, follow_up_content = await gemini_agent._format_tool_results([unified])
        
        assert len(parts) == 1
        part = parts[0]
        response_content = part.function_response.response["result"]
        assert "multimedia content" in response_content
        assert "test_tool" in response_content
        assert len(follow_up_content) == 1
        

class TestClaudeAgentFormatToolResults:
    """Test Claude agent's _format_tool_results method"""
    
    @pytest.fixture
    def claude_agent(self):
        config = ClaudeAgentConfig(
            model="claude-3-5-sonnet-20241022",
            api_key="test_key"
        )
        return ClaudeAgent(config, tools=[])
    
    @pytest.mark.asyncio
    async def test_simple_text_formatting(self, claude_agent):
        """Test simple text formatting for Claude"""
        tool_result = ToolResult(content=[TextBlock(text="Success")])
        unified = UnifiedToolResult(
            call_id="call_123",
            tool_name="test_tool", 
            tool_output=tool_result
        )
        
        blocks = await claude_agent._format_tool_results([unified])
        
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_result"
        assert blocks[0]["tool_use_id"] == "call_123"
        assert blocks[0]["content"] == "Success"
        
    @pytest.mark.asyncio
    async def test_multimodal_native_formatting(self, claude_agent):
        """Test Claude's native multimodal formatting"""
        tool_result = ToolResult(content=[
            TextBlock(text="Image generated: "),
            ImageBlock.from_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77TwAAAABJRU5ErkJggg==", "image/png")
        ])
        unified = UnifiedToolResult(
            call_id="call_123",
            tool_name="test_tool",
            tool_output=tool_result
        )
        
        blocks = await claude_agent._format_tool_results([unified])
        
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_result"
        assert blocks[0]["tool_use_id"] == "call_123"
        
        # Check multimodal content structure
        content = blocks[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        
        # First block should be text
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Image generated: "
        
        # Second block should be image
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["media_type"] == "image/png"
        
    @pytest.mark.asyncio
    async def test_file_block_processing(self, claude_agent):
        """Test file block processing in Claude"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            temp_path = f.name
        
        try:
            tool_result = ToolResult(content=[
                TextBlock(text="File created: "),
                FileBlock.from_path(temp_path, "text/plain", "test.txt")
            ])
            unified = UnifiedToolResult(
                call_id="call_123",
                tool_name="test_tool",
                tool_output=tool_result
            )
            
            with patch.object(claude_agent, '_upload_file', return_value=None):
                blocks = await claude_agent._format_tool_results([unified])
                
                assert len(blocks) == 1
                content = blocks[0]["content"]
                assert len(content) == 2
                
                # Check file is processed correctly
                assert content[1]["type"] == "text"  # Fallback for non-image files
                assert "test.txt" in content[1]["text"]
        finally:
            Path(temp_path).unlink()


class TestEndToEndMultimodalChain:
    """Test end-to-end multimodal tool output chain"""
    
    def create_mock_tool_result(self):
        """Create a mock tool result with multimodal content"""
        return ToolResult(content=[
            TextBlock(text="Analysis complete. "),
            ImageBlock.from_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77TwAAAABJRU5ErkJggg==", "image/png"),
            TextBlock(text=" Chart generated.")
        ])
    
    @pytest.mark.asyncio
    async def test_claude_end_to_end_multimodal(self):
        """Test full chain: ToolResult -> UnifiedToolResult -> Claude formatting"""
        # Step 1: Create ToolResult
        tool_result = self.create_mock_tool_result()
        
        # Step 2: Convert to UnifiedToolResult
        unified = UnifiedToolResult(
            call_id="test_call_123",
            tool_name="analyze_data",
            tool_output=tool_result
        )
        
        # Step 3: Format for Claude
        config = ClaudeAgentConfig(model="claude-3-5-sonnet-20241022", api_key="test_key")
        claude_agent = ClaudeAgent(config, tools=[])
        
        blocks = await claude_agent._format_tool_results([unified])
        
        # Verify end-to-end chain
        assert len(blocks) == 1
        block = blocks[0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "test_call_123"
        
        content = block["content"]
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Analysis complete. "
        assert content[1]["type"] == "image"
        assert content[2]["type"] == "text"
        assert content[2]["text"] == " Chart generated."
        
    @pytest.mark.asyncio
    async def test_gpt_end_to_end_multimodal(self):
        """Test full chain: ToolResult -> UnifiedToolResult -> GPT formatting"""
        # Step 1: Create ToolResult
        tool_result = self.create_mock_tool_result()
        
        # Step 2: Convert to UnifiedToolResult
        unified = UnifiedToolResult(
            call_id="test_call_123",
            tool_name="analyze_data",
            tool_output=tool_result
        )
        
        # Step 3: Format for GPT
        config = OpenAIAgentConfig(model="gpt-4", api_key="test_key", use_responses_api=False)
        gpt_agent = GPTAgent(config, tools=[])
        
        formatted_results, follow_up_content = await gpt_agent._format_tool_results([unified])
        
        # Verify end-to-end chain
        assert len(formatted_results) == 1
        result = formatted_results[0]
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "test_call_123"
        assert "multimedia content" in result["content"]
        
        # Check follow-up content
        assert len(follow_up_content) == 1
        assert follow_up_content[0]["tool_name"] == "analyze_data"
        assert len(follow_up_content[0]["full_content"]) == 3
        
    @pytest.mark.asyncio
    async def test_gemini_end_to_end_multimodal(self):
        """Test full chain: ToolResult -> UnifiedToolResult -> Gemini formatting"""
        # Step 1: Create ToolResult
        tool_result = self.create_mock_tool_result()
        
        # Step 2: Convert to UnifiedToolResult
        unified = UnifiedToolResult(
            call_id="test_call_123",
            tool_name="analyze_data",
            tool_output=tool_result
        )
        
        # Step 3: Format for Gemini
        config = GeminiAgentConfig(model="gemini-1.5-pro", api_key="test_key")
        gemini_agent = GeminiAgent(config, tools=[])
        
        parts, follow_up_content = await gemini_agent._format_tool_results([unified])
        
        # Verify end-to-end chain
        assert len(parts) == 1
        part = parts[0]
        assert hasattr(part, 'function_response')
        assert part.function_response.name == "analyze_data"
        
        response_content = part.function_response.response["result"]
        assert "multimedia content" in response_content
        
        # Check follow-up content
        assert len(follow_up_content) == 1
        assert follow_up_content[0]["tool_name"] == "analyze_data"
        
    @pytest.mark.asyncio
    async def test_error_handling_in_chain(self):
        """Test error handling throughout the multimodal chain"""
        # Create a tool result with an error
        tool_result = ToolResult(error="Tool execution failed")
        unified = UnifiedToolResult(
            call_id="test_call_123",
            tool_name="failing_tool",
            tool_output=tool_result,
            is_error=True
        )
        
        # Test Claude error handling
        config = ClaudeAgentConfig(model="claude-3-5-sonnet-20241022", api_key="test_key")
        claude_agent = ClaudeAgent(config, tools=[])
        
        blocks = await claude_agent._format_tool_results([unified])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_result"
        assert blocks[0]["tool_use_id"] == "test_call_123"
        
        # Test GPT error handling
        config = OpenAIAgentConfig(model="gpt-4", api_key="test_key", use_responses_api=False)
        gpt_agent = GPTAgent(config, tools=[])
        
        formatted_results, _ = await gpt_agent._format_tool_results([unified])
        assert len(formatted_results) == 1
        assert formatted_results[0]["role"] == "tool"
        assert formatted_results[0]["tool_call_id"] == "test_call_123"
        
    @pytest.mark.asyncio
    async def test_mixed_content_types(self):
        """Test handling of mixed content types in multimodal chain"""
        tool_result = ToolResult(content=[
            TextBlock(text="Multi-format output: "),
            ImageBlock.from_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77TwAAAABJRU5ErkJggg==", "image/png"),
            AudioBlock.from_base64("UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=", "audio/wav"),
            TextBlock(text=" Processing complete.")
        ])
        
        unified = UnifiedToolResult(
            call_id="test_call_123",
            tool_name="multi_format_tool",
            tool_output=tool_result
        )
        
        # Test Claude's native handling
        config = ClaudeAgentConfig(model="claude-3-5-sonnet-20241022", api_key="test_key")
        claude_agent = ClaudeAgent(config, tools=[])
        
        blocks = await claude_agent._format_tool_results([unified])
        content = blocks[0]["content"]
        
        # Should have all content types preserved
        assert len(content) == 4
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        assert content[2]["type"] == "audio"
        assert content[3]["type"] == "text"