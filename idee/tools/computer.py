"""
Computer use tool adapted from Anthropic's quickstart demo.
Allows the agent to interact with the screen, keyboard, and mouse.
"""

import asyncio
import base64
import os
import shlex
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict, Dict, Any, Optional, Tuple, Union, List
from uuid import uuid4
import platform
import sys

from .base import BaseTool, ToolError, ToolResult, run_command, maybe_truncate

import logging
logger = logging.getLogger(__name__)

# Try to import PyAutoGUI
try:
    import pyautogui
    # Disable PyAutoGUI's fail-safe (moving mouse to corner to abort)
    pyautogui.FAILSAFE = False
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    pyautogui = None
    PYAUTOGUI_AVAILABLE = False
    logger.warning("PyAutoGUI not available. Computer tool may not work properly. Install with: pip install pyautogui")

OUTPUT_DIR = "/tmp/outputs"
TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

# Computer actions for different API versions
Action_20241022 = Literal[
    "key",
    "type", 
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]

Action_20250124 = Union[
    Action_20241022,
    Literal[
        "left_mouse_down",
        "left_mouse_up", 
        "scroll",
        "hold_key",
        "wait",
        "triple_click",
    ]
]

ScrollDirection = Literal["up", "down", "left", "right"]

class Resolution(TypedDict):
    width: int
    height: int

# Recommended scaling targets
MAX_SCALING_TARGETS: Dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10  
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}

# PyAutoGUI handles all click types programmatically

class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"

def chunks(s: str, chunk_size: int) -> List[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]

class ComputerTool(BaseTool):
    """
    Computer use tool that allows interaction with screen, keyboard, and mouse.
    Supports vendor-specific implementations for better performance.
    """
    name: str = "computer"
    description: str = """Cross-platform computer interaction tool using PyAutoGUI. Works on macOS, Windows, and Linux.

- Take screenshots to see the current state of the screen
- Click on coordinates [x, y] to interact with UI elements  
- Type text or press keys to input data
- Use mouse actions to navigate and interact

Actions available:
- screenshot: Take a screenshot of the current screen
- left_click: Click at specific coordinates [x, y]
- right_click: Right-click at specific coordinates [x, y]
- middle_click: Middle-click at specific coordinates [x, y]
- double_click: Double-click at specific coordinates [x, y]
- type: Type the specified text
- key: Press a keyboard key (e.g. Return, Tab, Escape, etc.)
- mouse_move: Move mouse to coordinates [x, y]
- left_click_drag: Drag from current position to coordinates [x, y]
- cursor_position: Get current cursor position

Requires PyAutoGUI: pip install pyautogui
Note: On macOS, you may need to grant accessibility permissions to your terminal/IDE."""
    
    # Standard schema for regular API usage
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "screenshot", "left_click", "right_click", "middle_click",
                    "double_click", "mouse_move", "left_click_drag", "key",
                    "type", "cursor_position"
                ],
                "description": "The action to perform"
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "The [x, y] coordinate to interact with"
            },
            "text": {
                "type": "string", 
                "description": "Text to type (for 'type' action)"
            },
            "key": {
                "type": "string",
                "description": "Key to press (for 'key' action, e.g. 'Return', 'Tab')"
            }
        },
        "required": ["action"]
    }
    
    # Vendor-specific tool specifications with model support
    vendor_specs: Dict[str, Dict[str, Dict[str, Any]]] = {
        "anthropic": {
            # Claude 4 models: claude-opus-4-*, claude-sonnet-4-*, claude-haiku-4-*
            "claude-*-4-*": {
                "type": "computer_20250124",
                "name": "computer",
                "display_height_px": 768,
                "display_width_px": 1024,
                "display_number": None
            },
            # Claude 3.5 models: claude-3-5-sonnet-*, claude-3-5-haiku-*
            "claude-3-5-*": {
                "type": "computer_20241022",
                "name": "computer",
                "display_height_px": 768,
                "display_width_px": 1024,
                "display_number": None
            },
            # Default fallback for newer Claude models (use latest computer API)
            "claude-*": {
                "type": "computer_20250124",
                "name": "computer",
                "display_height_px": 768,
                "display_width_px": 1024,
                "display_number": None
            }
        }
    }

    def __init__(self, display_height_px: int = 768, display_width_px: int = 1024, display_number: Optional[int] = None):
        """
        Initialize the computer tool.
        
        Args:
            display_height_px: Height of the display in pixels
            display_width_px: Width of the display in pixels  
            display_number: X11 display number (None for default)
        """
        self.width = display_width_px
        self.height = display_height_px
        self.display_num = display_number
        
        self._screenshot_delay = 2.0
        self._scaling_enabled = True
        
        # Ensure output directory exists
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        
        # Check for required dependencies
        self.system = platform.system()
        self.pyautogui_available = PYAUTOGUI_AVAILABLE
        
        if not self.pyautogui_available:
            logger.warning("PyAutoGUI not available - computer tool may not work properly. Install with: pip install pyautogui")
        else:
            logger.info(f"Computer tool initialized for {self.system} using PyAutoGUI")
            # Configure PyAutoGUI settings
            pyautogui.PAUSE = 0.1  # Small pause between actions
            if self.system == "Darwin":  # macOS
                # Get actual screen size
                screen_size = pyautogui.size()
                logger.info(f"Detected screen size: {screen_size.width}x{screen_size.height}")

    @property
    def options(self) -> Dict[str, Any]:
        """Get display options for vendor tool specification."""
        return {
            "display_height_px": self.height,
            "display_width_px": self.width,
            "display_number": self.display_num,
        }

    async def run(self, action: str, coordinate: Optional[List[int]] = None, 
                  text: Optional[str] = None, key: Optional[str] = None, **kwargs) -> ToolResult:
        """
        Execute a computer action.
        
        Args:
            action: The action to perform
            coordinate: [x, y] coordinates for click/move actions
            text: Text to type (for 'type' action)
            key: Key to press (for 'key' action)
            **kwargs: Additional arguments
            
        Returns:
            ToolResult with action output
        """
        try:
            if action == "screenshot":
                return await self._screenshot()
            elif action in ["left_click", "right_click", "middle_click", "double_click"]:
                if coordinate is None:
                    raise ToolError(f"coordinate is required for {action}")
                return await self._click(action, coordinate[0], coordinate[1])
            elif action == "mouse_move":
                if coordinate is None:
                    raise ToolError("coordinate is required for mouse_move")
                return await self._mouse_move(coordinate[0], coordinate[1])
            elif action == "left_click_drag":
                if coordinate is None:
                    raise ToolError("coordinate is required for left_click_drag")
                return await self._click_drag(coordinate[0], coordinate[1])
            elif action == "type":
                if text is None:
                    raise ToolError("text is required for type action")
                return await self._type(text)
            elif action == "key":
                if key is None:
                    raise ToolError("key is required for key action")
                return await self._key(key)
            elif action == "cursor_position":
                return await self._cursor_position()
            else:
                raise ToolError(f"Unknown action: {action}")
                
        except ToolError:
            raise
        except Exception as e:
            logger.exception(f"Error in computer tool action {action}: {e}")
            raise ToolError(f"Computer tool error: {e}")

    async def _screenshot(self) -> ToolResult:
        """Take a screenshot of the current screen."""
        if not self.pyautogui_available:
            raise ToolError("PyAutoGUI is required for screenshots. Install with: pip install pyautogui")
            
        screenshot_path = Path(OUTPUT_DIR) / f"screenshot_{uuid4().hex}.png"
        
        try:
            # Use PyAutoGUI for cross-platform screenshot
            screenshot = pyautogui.screenshot()
            screenshot.save(screenshot_path)
                
            if not screenshot_path.exists():
                raise ToolError("Screenshot file was not created")
                
            # Read and encode screenshot
            with open(screenshot_path, "rb") as f:
                screenshot_data = f.read()
                
            base64_image = base64.b64encode(screenshot_data).decode()
            
            # Clean up
            screenshot_path.unlink(missing_ok=True)
            
            return ToolResult(
                output=f"Screenshot taken successfully",
                base64_image=base64_image
            )
            
        except Exception as e:
            screenshot_path.unlink(missing_ok=True)
            raise ToolError(f"Screenshot failed: {e}")

    async def _click(self, action: str, x: int, y: int) -> ToolResult:
        """Perform a mouse click action."""
        if not self.pyautogui_available:
            raise ToolError("PyAutoGUI is required for click actions")
            
        try:
            # Move to position first
            pyautogui.moveTo(x, y)
            
            # Perform the appropriate click action
            if action == "left_click":
                pyautogui.click(x, y, button='left')
            elif action == "right_click":
                pyautogui.click(x, y, button='right')
            elif action == "middle_click":
                pyautogui.click(x, y, button='middle')
            elif action == "double_click":
                pyautogui.doubleClick(x, y)
            else:
                raise ToolError(f"Unknown click action: {action}")
                
            return ToolResult(output=f"Performed {action} at ({x}, {y})")
            
        except Exception as e:
            raise ToolError(f"Click failed: {e}")

    async def _mouse_move(self, x: int, y: int) -> ToolResult:
        """Move the mouse to specified coordinates."""
        if not self.pyautogui_available:
            raise ToolError("PyAutoGUI is required for mouse actions")
            
        try:
            pyautogui.moveTo(x, y)
            return ToolResult(output=f"Moved mouse to ({x}, {y})")
            
        except Exception as e:
            raise ToolError(f"Mouse move failed: {e}")

    async def _click_drag(self, x: int, y: int) -> ToolResult:
        """Perform a click and drag action."""
        if not self.pyautogui_available:
            raise ToolError("PyAutoGUI is required for drag actions")
            
        try:
            # Get current mouse position
            current_x, current_y = pyautogui.position()
            # Drag from current position to target with left button
            pyautogui.dragTo(x, y, duration=0.5, button='left')
            return ToolResult(output=f"Dragged from ({current_x}, {current_y}) to ({x}, {y})")
            
        except Exception as e:
            raise ToolError(f"Click drag failed: {e}")

    async def _type(self, text: str) -> ToolResult:
        """Type the specified text."""
        if not self.pyautogui_available:
            raise ToolError("PyAutoGUI is required for typing")
            
        try:
            # Use PyAutoGUI to type text
            pyautogui.typewrite(text)
            
            # Truncate output to prevent context overflow
            output_text = maybe_truncate(f"Typed: {text}")
            return ToolResult(output=output_text)
            
        except Exception as e:
            raise ToolError(f"Type failed: {e}")

    async def _key(self, key: str) -> ToolResult:
        """Press a keyboard key."""
        if not self.pyautogui_available:
            raise ToolError("PyAutoGUI is required for key presses")
            
        try:
            # Map common key names to PyAutoGUI format
            key_mapping = {
                'Return': 'enter',
                'Tab': 'tab',
                'Escape': 'esc',
                'Space': 'space',
                'BackSpace': 'backspace',
                'Delete': 'delete',
                'Up': 'up',
                'Down': 'down',
                'Left': 'left',
                'Right': 'right',
                'Home': 'home',
                'End': 'end',
                'Page_Up': 'pageup',
                'Page_Down': 'pagedown',
                'F1': 'f1', 'F2': 'f2', 'F3': 'f3', 'F4': 'f4',
                'F5': 'f5', 'F6': 'f6', 'F7': 'f7', 'F8': 'f8',
                'F9': 'f9', 'F10': 'f10', 'F11': 'f11', 'F12': 'f12',
                'ctrl': 'ctrl', 'alt': 'alt', 'shift': 'shift', 'cmd': 'cmd'
            }
            
            # Convert key name if needed
            pyautogui_key = key_mapping.get(key, key.lower())
            
            # Press the key
            pyautogui.press(pyautogui_key)
            
            return ToolResult(output=f"Pressed key: {key}")
            
        except Exception as e:
            raise ToolError(f"Key press failed: {e}")

    async def _cursor_position(self) -> ToolResult:
        """Get the current cursor position."""
        if not self.pyautogui_available:
            raise ToolError("PyAutoGUI is required for cursor position")
            
        try:
            x, y = pyautogui.position()
            return ToolResult(output=f"Cursor position: ({x}, {y})")
            
        except Exception as e:
            raise ToolError(f"Get cursor position failed: {e}")