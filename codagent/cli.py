#!/usr/bin/env python3
"""
Command-line interface for the CodAgent tool.
"""

import argparse
import os
import sys
import re
from pathlib import Path
import google.generativeai as genai
from tqdm import tqdm
import time
import colorama
from colorama import Fore, Back, Style
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style as PromptStyle
import glob
import subprocess
from prompt_toolkit.formatted_text import ANSI

# Initialize colorama
colorama.init(autoreset=True)

# --- Border Characters ---
TL = '╭' # Top Left
TR = '╮' # Top Right
BL = '╰' # Bottom Left
BR = '╯' # Bottom Right
H = '─'  # Horizontal Line
V = '│'  # Vertical Line

# --- Helper Function for Visible Length ---
def visible_len(text):
    """Calculates the visible length of a string by removing ANSI escape codes."""
    # Regex to remove ANSI escape codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', text))

# --- Helper Function for Boxed Output ---
def print_boxed(title, content, color=Fore.CYAN, width=80):
    """Prints content inside a simulated rounded border, aware of ANSI codes."""
    lines = content.splitlines()
    # Use visible_len for calculating max content width
    max_visible_line_width = max(visible_len(line) for line in lines) if lines else 0
    # Use visible_len for title width as well
    visible_title_width = visible_len(title)

    inner_width = max(visible_title_width, max_visible_line_width)
    box_width = inner_width + 4 # Account for padding and vertical borders

    # Ensure box_width doesn't exceed desired maximum width
    box_width = min(box_width, width)
    inner_width = box_width - 4 # Recalculate inner_width based on final box_width

    # --- Top border ---
    print(f"{color}{TL}{H * (box_width - 2)}{TR}{Style.RESET_ALL}")

    # --- Title line ---
    # Calculate padding needed based on visible width
    title_padding_total = inner_width - visible_title_width
    title_pad_left = title_padding_total // 2
    title_pad_right = title_padding_total - title_pad_left
    # Print title with manual padding around the original colored string
    print(f"{color}{V} {' ' * title_pad_left}{Style.BRIGHT}{title}{Style.NORMAL}{' ' * title_pad_right} {V}{Style.RESET_ALL}")

    # --- Separator ---
    # Use H (horizontal line) which doesn't have color codes itself
    print(f"{color}{V}{H * inner_width}{V}{Style.RESET_ALL}") # This line is usually fine

    # --- Content lines ---
    for line in lines:
        v_len = visible_len(line)
        # Simple wrapping (split long lines) - based on visible length
        while v_len > inner_width:
             # This naive split won't handle color codes spanning the split well
             # For simplicity, we just split; complex color handling across wraps is hard
             split_point = inner_width
             # Try to find a good split point based on visible chars (approximate)
             current_visible = 0
             actual_split_idx = 0
             for i, char in enumerate(line):
                 if not re.match(r'\x1B', char) and not (char == '\x1B' and line[i+1:i+2] == '['): # Simple check for start of escape
                      current_visible += 1
                 actual_split_idx = i + 1
                 if current_visible >= inner_width:
                     break

             part1 = line[:actual_split_idx]
             part2 = line[actual_split_idx:]
             padding_needed = inner_width - visible_len(part1)
             print(f"{color}{V} {part1}{' ' * padding_needed} {V}{Style.RESET_ALL}")
             line = part2
             v_len = visible_len(line)

        # Print remaining part or short line with correct padding
        padding_needed = inner_width - v_len
        print(f"{color}{V} {line}{' ' * padding_needed} {V}{Style.RESET_ALL}")


    # --- Bottom border ---
    print(f"{color}{BL}{H * (box_width - 2)}{BR}{Style.RESET_ALL}")


def check_api_key():
    """Check if Google API key is set in environment variables."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(f"{Fore.YELLOW}Google API key not found in environment variables.{Style.RESET_ALL}")
        
        if os.name == "nt":  # Windows
            print("You need to set the GOOGLE_API_KEY environment variable.")
            print("\nYou can set it temporarily with:")
            print(f"{Fore.CYAN}set GOOGLE_API_KEY=your_api_key{Style.RESET_ALL}")
            print("\nOr permanently via System Properties > Advanced > Environment Variables.")
        else:  # Linux/Unix
            print("You need to set the GOOGLE_API_KEY environment variable.")
            print("\nYou can set it temporarily with:")
            print(f"{Fore.CYAN}export GOOGLE_API_KEY=your_api_key{Style.RESET_ALL}")
            print("\nOr add it to your ~/.bashrc or ~/.zshrc file.")
        
        # Ask for API key
        print(Style.RESET_ALL, end='') # Explicitly reset colors before input
        api_key = input("\nEnter your Google API key: ").strip()
        if not api_key:
            print(f"{Fore.RED}No API key provided. Exiting.{Style.RESET_ALL}")
            sys.exit(1)
        
        os.environ["GOOGLE_API_KEY"] = api_key
        print(f"{Fore.GREEN}API key set for this session.{Style.RESET_ALL}")
    
    return api_key

def initialize_genai(model_name):
    """Initialize the Google GenerativeAI with the given model."""
    api_key = check_api_key()
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        print(f"{Fore.RED}Error initializing the model: {e}{Style.RESET_ALL}")
        sys.exit(1)

def get_system_prompt():
    """Generate a comprehensive system prompt for the AI."""
    current_dir = os.getcwd()
    
    system_prompt = f"""
**You are CodAgent:** An AI assistant specializing in code generation and modification within a user's local file system. Your goal is to understand user requests and translate them into precise file operations or terminal commands.

**Your Working Directory:** `{current_dir}`

---

**CORE BEHAVIORS & CAPABILITIES:**

1.  **Autonomous Agent:** You decide when your response is complete.
    *   If you need more steps or space, simply end your response normally. You will be prompted to continue.
    *   When your entire thought process and all actions for the current request are finished, end your *final* message with the `[END]` tag.
    *   **Example Multi-Turn:**
        ```
        === Response ===
        Okay, first I'll create the main Python file...
        [CREATE=main.py]
        # Initial code
[/CREATE]

        === Response ===
        Now, let's add the helper function.
        [REPLACE=main.py]
        # Initial code
        [TO]
        # Initial code
        def helper():
            pass
[/REPLACE]
        Almost done...

        === Response ===
        Finally, adding the main execution block.
        [REPLACE=main.py]
        def helper():
            pass
        [TO]
        def helper():
            pass

        if __name__ == "__main__":
            helper()
[/REPLACE]
        All set! [END]
        ```

2.  **File Operations:** Use specific tags to manage files:
    *   **Create:** `[CREATE=path/to/filename.ext]...content...[/CREATE]`
    *   **Replace:** `[REPLACE=path/to/filename.ext]...exact_content_to_replace...[TO]...new_content...[/REPLACE]`
    *   **CRITICAL:** Ensure `[REPLACE]` blocks contain the *exact* content to be replaced, including all whitespace and indentation. Use the file context provided.

3.  **Terminal Execution:** Execute shell commands:
    *   **Syntax:** `[TERMINAL]...command...[/TERMINAL]`
    *   **Example:** `[TERMINAL]pip install requests[/TERMINAL]`
    *   **IMPORTANT:** Explain *why* a command is needed. The user *must* approve execution. You will receive the command's output (stdout/stderr) for context in the next turn.

---

**⚠️ CRITICAL REQUIREMENTS - MUST FOLLOW ⚠️**

*   **CODE EXACTNESS (REPLACE):** The content between `[REPLACE=...]` and `[TO]` *must perfectly match* the existing code in the file, character for character, including **all leading/trailing whitespace and indentation**. This is the most common point of failure. Double-check against the file context. If you cannot provide the exact block, the operation will likely fail.
*   **INDENTATION:** Preserve original indentation in `[REPLACE]` tags. For `[CREATE]` or new code in `[REPLACE]`, follow the existing file's indentation style or standard conventions (e.g., 4 spaces for Python).
*   **NO MARKDOWN CODE BLOCKS:** Never use triple backticks (\\`) in your response outside of the file context display. Use the specified tags (`[CREATE]`, `[REPLACE]`, `[TERMINAL]`) instead.

---

**GENERAL GUIDELINES:**

*   **Clarity:** Explain your plan and the purpose of your code/commands.
*   **Context:** Use the provided `FILE CONTEXT` and `CONVERSATION HISTORY` to inform your actions. Refer to existing files accurately.
*   **Completeness:** Provide functional code snippets or commands.
*   **Safety:** Be cautious with terminal commands, especially those that modify files or system state (`rm`, `mv`, etc.). Always explain the command's effect.
*   **Ask:** If a request is ambiguous, ask for clarification.

---

**USER WORKFLOW:**

1.  User provides a request.
2.  You analyze, potentially ask questions, and respond with explanations and tagged operations (`[CREATE]`, `[REPLACE]`, `[TERMINAL]`).
3.  If your response does not end with `[END]`, CodAgent prompts you to continue.
4.  Once you provide `[END]`, CodAgent previews the collected operations/commands to the user.
5.  User approves or rejects.
6.  If approved, CodAgent applies the changes/executes commands. Terminal output is provided back to you in the *next* turn's history.
7.  CodAgent waits for the user's next request.

---

Remember to use the provided **FILE CONTEXT** below, which shows the current state of modified/created files.
"""
    return system_prompt

def parse_end_response(response_text):
    """Parse the response to check if it contains the END tag at the end."""
    # Check if the response ends with [END] tag
    if response_text.strip().endswith("[END]"):
        # Remove the [END] tag and return True to indicate this is the end
        return response_text.strip()[:-5].strip(), True # Length of "[END]" is 5
    
    # No END tag found, return the original response and False
    return response_text, False

def parse_terminal_commands(response_text):
    """Parse the response text to extract terminal commands."""
    terminal_commands = []
    
    # Find TERMINAL commands
    terminal_pattern = r"\[TERMINAL\](.*?)\[/TERMINAL\]"
    for match in re.finditer(terminal_pattern, response_text, re.DOTALL):
        command = match.group(1).strip()
        terminal_commands.append(command)
    
    return terminal_commands

def execute_terminal_command(command):
    """Execute a terminal command and capture its output."""
    # Keep immediate execution log outside the final box for clarity
    print("-" * 30)
    print(f"{Style.BRIGHT}{Fore.YELLOW}Executing Command:{Style.RESET_ALL} {Fore.WHITE}{command}{Style.RESET_ALL}")
    
    output = ""
    errors = ""
    return_code = -1
    exec_log = [] # Log for the final box

    try:
        result = subprocess.run(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False
        )
        output = result.stdout.strip()
        errors = result.stderr.strip()
        return_code = result.returncode
        
        if output:
            exec_log.append(f"{Fore.CYAN}--- Output ---{Style.RESET_ALL}")
            exec_log.extend(output.splitlines()) # Add each line separately
            exec_log.append(f"{Fore.CYAN}--------------{Style.RESET_ALL}")

        if errors:
            exec_log.append(f"{Fore.RED}--- Errors ---{Style.RESET_ALL}")
            exec_log.extend(errors.splitlines())
            exec_log.append(f"{Fore.RED}------------{Style.RESET_ALL}")

        if return_code == 0:
             exec_log.append(f"{Fore.GREEN}✓ Command finished successfully (Exit Code: 0){Style.RESET_ALL}")
        else:
             exec_log.append(f"{Fore.RED}✗ Command failed (Exit Code: {return_code}){Style.RESET_ALL}")

    except Exception as e:
        errors = str(e)
        exec_log.append(f"{Fore.RED}✗ Error executing command:{Style.RESET_ALL} {e}")
        return_code = -1 # Indicate failure

    # Print execution log in a box
    box_color = Fore.RED if return_code != 0 else Fore.GREEN
    print_boxed(f"Execution Result: {command}", "\n".join(exec_log), color=box_color)
    print("-" * 30) # Separator after box

    return {"stdout": output, "stderr": errors, "returncode": return_code}

def strip_code_fences(content):
    """Removes leading/trailing markdown code fences (```lang...``` or ```...```)."""
    # Pattern to match optional language and the fences
    # Handles ```python ... ``` or ``` ... ```
    pattern = r"^\s*```[\w]*\n?(.*?)?\n?```\s*$"
    match = re.match(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        # Return the content inside the fences, stripping outer whitespace
        return match.group(1).strip() if match.group(1) else ""
    # If no fences found, just strip outer whitespace
    return content.strip()

def parse_file_operations(response_text):
    """Parse the response text to extract file operations."""
    # Clean the response text (remove [END] tag if present) before parsing ops
    cleaned_response, _ = parse_end_response(response_text)
    
    file_operations = []
    
    # Find CREATE operations using the cleaned response
    create_pattern = r"\[CREATE=([^\]]+)\](.*?)\[/CREATE\]"
    for match in re.finditer(create_pattern, cleaned_response, re.DOTALL):
        filename = match.group(1).strip()
        # --- Strip content AND code fences for CREATE ---
        raw_content = match.group(2)
        content = strip_code_fences(raw_content)
        if content: # Add operation only if content is not empty after stripping
            file_operations.append({
                "type": "create",
                "filename": filename,
                "content": content
            })
        else:
             print(f"{Fore.YELLOW}Warning: Skipping CREATE operation for '{filename}' because content was empty after stripping.{Style.RESET_ALL}")
    
    
    # Find REPLACE operations using the cleaned response
    replace_pattern = r"\[REPLACE=([^\]]+)\](.*?)\[TO\](.*?)\[/REPLACE\]"
    for match in re.finditer(replace_pattern, cleaned_response, re.DOTALL):
        filename = match.group(1).strip()
        # --- Strip content AND code fences for REPLACE ---
        raw_old_content = match.group(2)
        raw_new_content = match.group(3)
        
        old_content = strip_code_fences(raw_old_content)
        new_content = strip_code_fences(raw_new_content)

        if old_content: # Add operation only if old_content is not empty after stripping
            file_operations.append({
                "type": "replace",
                "filename": filename,
                 "old_content": old_content, # Store stripped version
                 "new_content": new_content  # Store stripped version
            })
        else:
              print(f"{Fore.YELLOW}Warning: Skipping REPLACE operation for '{filename}' because the 'old content' block was empty after stripping.{Style.RESET_ALL}")
    
    return file_operations

def show_diff(old_lines, new_lines):
    """Show colored diff between old and new content."""
    import difflib
    
    diff = difflib.ndiff(old_lines, new_lines)
    print(f"{Fore.MAGENTA}--- Diff Start ---{Style.RESET_ALL}")
    for line in diff:
        if line.startswith('+ '):
            print(f"{Fore.GREEN}+{Style.RESET_ALL} {line[2:]}") # Green for additions
        elif line.startswith('- '):
            print(f"{Fore.RED}-{Style.RESET_ALL} {line[2:]}")   # Red for deletions
        elif line.startswith('? '):
            # Optionally show context hints in a different color
            # print(f"{Fore.CYAN}{line}{Style.RESET_ALL}") 
            continue
        else:
            print(f"  {line[2:]}") # Keep indentation for context lines
    print(f"{Fore.MAGENTA}--- Diff End ---{Style.RESET_ALL}")

def preview_changes(file_operations):
    """Preview changes to be made to files."""
    preview_content = [] # Collect content for the box

    if not file_operations:
        preview_content.append(f"{Fore.YELLOW}No file operations proposed.{Style.RESET_ALL}")
        print_boxed("File Operations Preview", "\n".join(preview_content), color=Fore.YELLOW)
        return True # Nothing to confirm

    operations_present = False
    for op in file_operations:
        operations_present = True
        preview_content.append("-" * 30) # Separator within the box content
        if op["type"] == "create":
            preview_content.append(f"{Style.BRIGHT}{Fore.GREEN}CREATE File:{Style.RESET_ALL} {Fore.WHITE}{op['filename']}{Style.RESET_ALL}")
            preview_content.append(f"{Fore.YELLOW}Content Preview (first 5 lines):{Style.RESET_ALL}")
            content_lines = op["content"].splitlines()
            for line in content_lines[:5]:
                 preview_content.append(f"{Fore.GREEN}  {line}{Style.RESET_ALL}") # Keep indent
            if len(content_lines) > 5:
                preview_content.append(f"{Fore.GREEN}  ...{Style.RESET_ALL}")
            preview_content.append("") # Add empty line for spacing
        
        elif op["type"] == "replace":
            preview_content.append(f"{Style.BRIGHT}{Fore.YELLOW}REPLACE in File:{Style.RESET_ALL} {Fore.WHITE}{op['filename']}{Style.RESET_ALL}")
            preview_content.append(f"{Fore.YELLOW}Changes:{Style.RESET_ALL}")

            # Use difflib to generate diff lines for preview content
            import difflib
            diff = difflib.ndiff(op["old_content"].splitlines(), op["new_content"].splitlines())
            preview_content.append(f"{Fore.MAGENTA}--- Diff Start ---{Style.RESET_ALL}")
            diff_lines_added = 0
            for line in diff:
                if diff_lines_added >= 10: # Limit diff preview size inside box
                     preview_content.append(f"{Fore.MAGENTA}  ... (diff truncated) ...{Style.RESET_ALL}")
                     break
                if line.startswith('+ '):
                    preview_content.append(f"{Fore.GREEN}+{Style.RESET_ALL} {line[2:]}")
                    diff_lines_added += 1
                elif line.startswith('- '):
                    preview_content.append(f"{Fore.RED}-{Style.RESET_ALL} {line[2:]}")
                    diff_lines_added += 1
                elif line.startswith('? '):
                    continue
                else:
                    # Only show limited context lines in preview box
                    if diff_lines_added < 5: # Show a bit of context at start
                         preview_content.append(f"  {line[2:]}")
                         diff_lines_added += 1
            preview_content.append(f"{Fore.MAGENTA}--- Diff End ---{Style.RESET_ALL}")
            preview_content.append("") # Add empty line for spacing

    if not operations_present:
         preview_content.append(f"{Fore.YELLOW}No file operations were parsed from the response.{Style.RESET_ALL}")
         print_boxed("File Operations Preview", "\n".join(preview_content), color=Fore.YELLOW)
         return True # Nothing to confirm


    # Print the collected content inside a box
    print_boxed("File Operations Preview", "\n".join(preview_content), color=Fore.CYAN)

    print("-" * 30) # Separator outside the box before confirmation
    # Use startswith('y') for more robust check
    raw_confirm = input(f"{Style.BRIGHT}{Fore.CYAN}Apply these file changes? (y/n): {Style.RESET_ALL}")
    confirm = raw_confirm.lower().strip()

    # --- Add Debugging ---
    print(f"{Style.DIM}[Debug] Raw input: {repr(raw_confirm)}, Processed confirm: {repr(confirm)}, Comparison result (startswith): {confirm.startswith('y')}{Style.RESET_ALL}")
    # --- End Debugging ---

    # --- Changed Comparison ---
    return confirm.startswith('y')

def apply_changes(file_operations):
    """Apply the file operations."""
    failed_ops = []
    successful_ops = []
    apply_log = [] # Collect log messages for the box
    
    for op in file_operations:
        if op["type"] == "create":
            try:
                directory = os.path.dirname(op["filename"])
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                    apply_log.append(f"{Fore.YELLOW}  Created directory: {directory}{Style.RESET_ALL}")
                content_to_write = op["content"].replace('\r\n', '\n')
                with open(op["filename"], "w", newline='\n') as f:
                    f.write(content_to_write)
                apply_log.append(f"{Fore.GREEN}✓ SUCCESS:{Style.RESET_ALL} Created {Fore.WHITE}{op['filename']}{Style.RESET_ALL}")
                successful_ops.append(op)
            except Exception as e:
                apply_log.append(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} Could not create {Fore.WHITE}{op['filename']}{Style.RESET_ALL}: {e}")
                failed_ops.append(op)
        
        elif op["type"] == "replace":
            try:
                with open(op["filename"], "r") as f:
                    content = f.read()
                
                if op["old_content"] in content:
                    new_content_applied = content.replace(op["old_content"], op["new_content"], 1)
                    with open(op["filename"], "w", newline='\n') as f:
                        f.write(new_content_applied.replace('\r\n', '\n'))
                    apply_log.append(f"{Fore.GREEN}✓ SUCCESS:{Style.RESET_ALL} Updated {Fore.WHITE}{op['filename']}{Style.RESET_ALL} (Exact Match)")
                    successful_ops.append(op)
                else:
                    apply_log.append(f"{Fore.YELLOW}  Exact match for REPLACE in '{op['filename']}' failed. Trying indentation-aware fallback...{Style.RESET_ALL}")
                    # Debug info can still print normally outside the box for clarity
                    print(f"{Style.DIM}--- Debug Info for Exact Match Failure ---{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}  File Content (repr, first 100 chars):{Style.RESET_ALL}\n    {repr(content[:100])}")
                    print(f"{Fore.CYAN}  AI 'Old Content' (repr):{Style.RESET_ALL}\n    {repr(op['old_content'])}")
                    print(f"{Style.DIM}------------------------------------------{Style.RESET_ALL}")

                    success = handle_indentation_mismatch(op["filename"], op["old_content"], op["new_content"])
                    
                    if success:
                        apply_log.append(f"{Fore.GREEN}✓ SUCCESS:{Style.RESET_ALL} Applied indentation-aware replacement in {Fore.WHITE}{op['filename']}{Style.RESET_ALL}")
                        successful_ops.append(op)
                    else:
                        apply_log.append(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} Could not find/replace content block in {Fore.WHITE}{op['filename']}{Style.RESET_ALL}.")
                        failed_ops.append(op)

            except FileNotFoundError:
                apply_log.append(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} File {Fore.WHITE}{op['filename']}{Style.RESET_ALL} not found for REPLACE.")
                failed_ops.append(op)
            except Exception as e:
                apply_log.append(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} Error processing REPLACE for {Fore.WHITE}{op['filename']}{Style.RESET_ALL}: {e}")
                failed_ops.append(op)
    
    # Print the apply log inside a box
    box_color = Fore.RED if failed_ops else Fore.GREEN
    print_boxed("Applying File Operations Results", "\n".join(apply_log), color=box_color)

    return {"successful": successful_ops, "failed": failed_ops}

def handle_indentation_mismatch(filename, old_content, new_content):
    """Handle replacements with indentation mismatches."""
    try:
        # Use newline=None to preserve original line endings during read
        with open(filename, "r", newline=None) as f:
            file_lines = f.readlines() # Reads lines keeping original endings
            
        # Normalize newlines from AI content just for processing here
        old_lines = old_content.replace('\r\n', '\n').splitlines() 
        
        # Create a list of non-empty lines, stripped of LEADING whitespace only
        stripped_old_lines = [line.lstrip() for line in old_lines if line.strip()]
        
        if not stripped_old_lines:
             print(f"{Fore.YELLOW}  [Debug Indent Handler] AI 'old_content' was empty or only whitespace. Cannot match.{Style.RESET_ALL}")
             return False
        
        # If the content is only one line, handle it specially
        if len(stripped_old_lines) == 1:
            # Pass original file_lines with potentially mixed endings
            return handle_single_line_replacement(filename, old_content, new_content, file_lines)
        
        # If we have multiple non-empty lines, try a more sophisticated matching
        # Pass original file_lines with potentially mixed endings
        return handle_multiline_replacement(filename, stripped_old_lines, new_content, file_lines)
    
    except Exception as e:
        print(f"{Fore.RED}Error during indentation-aware replacement: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return False

def handle_multiline_replacement(filename, stripped_old_lines, new_content, file_lines):
    """Handle more complex multiline replacements with flexible indentation matching."""
    print(f"{Fore.CYAN}  [Debug Indent Handler] Searching for {len(stripped_old_lines)} lines. First line pattern: '{stripped_old_lines[0]}'{Style.RESET_ALL}")
    
    potential_starts = []
    found_first_line = False # Debug flag
    
    # Process file lines consistently: normalize newlines *after* reading, then strip
    normalized_file_lines_stripped = [line.replace('\r\n', '\n').rstrip('\n').lstrip() for line in file_lines]

    for i, stripped_file_line in enumerate(normalized_file_lines_stripped):
        # Debug print for comparison - uncomment if needed
        # print(f"    [Debug] Comparing file line {i+1} stripped: '{stripped_file_line}' == pattern: '{stripped_old_lines[0]}'")
        if stripped_file_line and stripped_file_line == stripped_old_lines[0]:
            potential_starts.append(i)
            found_first_line = True 

    # More detailed debug message if the first line isn't found
    if not found_first_line:
        print(f"{Fore.RED}  [Debug Indent Handler] Failed to find any match for the first stripped line pattern: '{stripped_old_lines[0]}'{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  [Debug Indent Handler] Stripped file lines searched:{Style.RESET_ALL}")
        for idx, fl_line_stripped in enumerate(normalized_file_lines_stripped):
             # Limit printing for very long files
             if idx < 20 or idx > len(normalized_file_lines_stripped) - 5 : 
                 print(f"    {idx+1}: '{fl_line_stripped}'")
             elif idx == 20:
                 print("     ...")

        return False 
    
    if not potential_starts:
         print(f"{Fore.RED}× FAILED: Could not find any potential starting points (logic error?).{Style.RESET_ALL}")
         return False
    
    print(f"{Fore.CYAN}  [Debug Indent Handler] Found {len(potential_starts)} potential starting indices: {potential_starts}{Style.RESET_ALL}")
    
    # --- Try each potential starting point ---
    for start_idx in potential_starts:
        match_found = True
        indentation_patterns = []
        matching_indices = [] # Store the original indices from file_lines
        current_file_idx = start_idx # Track index in the original file_lines
        
        # Check if this starting point leads to a full match
        for j, pattern_line in enumerate(stripped_old_lines):
            # Need to find the next non-empty line in the file from current_file_idx onwards
            found_matching_file_line = False
            while current_file_idx < len(file_lines):
                file_line_raw = file_lines[current_file_idx]
                # Normalize and strip the current file line for comparison
                stripped_current_file_line = file_line_raw.replace('\r\n', '\n').rstrip('\n').lstrip()

                # If the stripped file line is empty, skip it and check the next one
                if not stripped_current_file_line:
                    current_file_idx += 1
                    continue
            
                # Now compare the non-empty stripped file line with the pattern line
                # print(f"    [Debug] Matching pattern '{pattern_line}' with file line {current_file_idx+1} stripped '{stripped_current_file_line}'")
                if stripped_current_file_line == pattern_line:
                    # Store the indentation from the *original raw* file line
                    original_indentation = file_line_raw[:len(file_line_raw) - len(file_line_raw.lstrip())]
                    indentation_patterns.append(original_indentation)
                    matching_indices.append(current_file_idx)
                    current_file_idx += 1 # Move to next file line for the next pattern
                    found_matching_file_line = True
                    break # Found match for this pattern line, move to next pattern line
            else:
                # Mismatch found for this pattern line at this file position
                match_found = False
                break # Stop checking this potential_start

            # If we reached end of file while looking or if inner loop broke due to mismatch
            if not found_matching_file_line or not match_found:
                 match_found = False
                 break # Stop checking this potential_start

        # --- Check if a full match was found for this potential_start ---
        if match_found and len(matching_indices) == len(stripped_old_lines):
            print(f"{Fore.GREEN}✓ SUCCESS:{Style.RESET_ALL} Found matching content block starting at line {matching_indices[0]+1}.")
            
            # Determine the base indentation (from first matching line's original indent)
            base_indentation = indentation_patterns[0]
            
            # Apply the replacement with proper indentation, preserving original line endings where possible
            new_lines_with_indent = []
            # Normalize AI's new content newlines for processing
            normalized_new_lines = new_content.replace('\r\n','\n').splitlines()
            
            for new_line in normalized_new_lines:
                stripped_new = new_line.lstrip()
                if stripped_new: # If line has content
                    # Indent based on the first line's indent, keep original newline type if possible
                    # For simplicity, we'll write with '\n' consistently now
                     new_lines_with_indent.append(f"{base_indentation}{stripped_new}\n")
                else: # Preserve empty lines
                     new_lines_with_indent.append("\n")
            
            # Replace the lines in the original file_lines list
            # We need the start index and the end index of the matched block
            start_replace_idx = matching_indices[0]
            end_replace_idx = matching_indices[-1] 
            
            # Replace slice in original list
            file_lines[start_replace_idx : end_replace_idx + 1] = new_lines_with_indent
            
            # Write back to the file using consistent newlines
            try:
                with open(filename, "w", newline='\n') as f:
                    f.writelines(file_lines)
                print(f"{Fore.GREEN}✓ SUCCESS:{Style.RESET_ALL} Replaced content in {Fore.WHITE}{filename}{Style.RESET_ALL} using indentation handler.")
                return True # Successful replacement
            except Exception as write_error:
                 print(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} Error writing replaced content to {Fore.WHITE}{filename}{Style.RESET_ALL}: {write_error}")
                 return False # Write failed


    # If we loop through all potential_starts and none result in a full match
    print(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} Could not find a complete matching block in {Fore.WHITE}{filename}{Style.RESET_ALL} after checking all potential starts.")
    
    # --- Fallback: Fuzzy Match (Optional, can be noisy/dangerous) ---
    # Consider if fuzzy matching is desired or too risky. Let's disable it for now.
    # print(f"{Fore.YELLOW}  Skipping fuzzy match fallback.{Style.RESET_ALL}")
    # return attempt_fuzzy_match(filename, stripped_old_lines, new_content, file_lines)
    return False

def handle_single_line_replacement(filename, old_content, new_content, file_lines):
    """Handle single line replacements with indentation preservation."""
    # Normalize AI content for comparison
    old_stripped = old_content.replace('\r\n', '\n').strip() 
    success = False
    
    print(f"{Fore.CYAN}  [Debug Indent Handler] Looking for single-line match: '{old_stripped}'{Style.RESET_ALL}")
    
    match_index = -1
    original_indentation = ""

    # Try exact content match first (stripping file lines for comparison)
    for i, line_raw in enumerate(file_lines):
        stripped_line = line_raw.replace('\r\n', '\n').strip() # Normalize and strip file line
        if stripped_line == old_stripped:
            # Found match - preserve the original indentation
            original_indentation = line_raw[:len(line_raw) - len(line_raw.lstrip())]
            match_index = i
            print(f"{Fore.GREEN}  [Debug Indent Handler] Found exact single-line match at line {i+1}. Indentation: '{original_indentation}'{Style.RESET_ALL}")
            success = True
            break
    
    if success:
            # Apply indentation to new content (potentially multi-line)
        new_lines_with_indent = []
        normalized_new_lines = new_content.replace('\r\n','\n').splitlines()
        for new_line in normalized_new_lines:
            stripped_new = new_line.lstrip()
            if stripped_new:
                new_lines_with_indent.append(f"{original_indentation}{stripped_new}\n")
            else:
                new_lines_with_indent.append("\n")
            
        # Replace the line in the file_lines list
        file_lines[match_index : match_index+1] = new_lines_with_indent
            
            # Write back to the file
        try:
            with open(filename, "w", newline='\n') as f:
                f.writelines(file_lines)
            print(f"{Fore.GREEN}✓ SUCCESS:{Style.RESET_ALL} Replaced single line in {Fore.WHITE}{filename}{Style.RESET_ALL} with indentation preserved.")
            return True
        except Exception as write_error:
             print(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} Error writing replaced single line to {Fore.WHITE}{filename}{Style.RESET_ALL}: {write_error}")
             return False
             
    # If no exact match, maybe try fuzzy later, but let's rely on exact for now.
    print(f"{Fore.YELLOW}  [Debug Indent Handler] No exact single-line match found.{Style.RESET_ALL}")
    return False

def process_add_command(target):
    """Process the /add command to include file or directory content."""
    if not os.path.exists(target):
        print(f"{Fore.RED}Error: {target} does not exist.{Style.RESET_ALL}")
        return None
    
    content = []
    
    if os.path.isfile(target):
        # Single file
        try:
            with open(target, 'r', encoding='utf-8') as f:
                file_content = f.read()
            content.append(f"**File: {target}**\n```\n{file_content}\n```\n")
            
        except UnicodeDecodeError:
            print(f"{Fore.YELLOW}Warning: {target} appears to be a binary file and cannot be read as text.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error reading {target}: {e}{Style.RESET_ALL}")
    
    elif os.path.isdir(target):
        # Directory - only add files, not subdirectories
        files = [f for f in glob.glob(os.path.join(target, '*')) if os.path.isfile(f)]
        
        if not files:
            print(f"{Fore.YELLOW}No files found in {target}.{Style.RESET_ALL}")
            return None
        
        print(f"{Fore.CYAN}Found {len(files)} files in {target}.{Style.RESET_ALL}")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                content.append(f"**File: {file_path}**\n```\n{file_content}\n```\n")
                
            except UnicodeDecodeError:
                print(f"{Fore.YELLOW}Warning: Skipping {file_path} (appears to be a binary file).{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error reading {file_path}: {e}{Style.RESET_ALL}")
    
    if not content:
        print(f"{Fore.YELLOW}No readable content found.{Style.RESET_ALL}")
        return None
    
    return "\n".join(content)

def generate_file_context(file_history):
    """Generate a context string about files that have been created or modified."""
    context_lines = []
    
    # Add information about files created in this session
    if file_history["created"]:
        context_lines.append(f"\n{Fore.GREEN}Files CREATED this session:{Style.RESET_ALL}")
        for file in file_history["created"]:
            context_lines.append(f"- {Fore.WHITE}{file}{Style.RESET_ALL}")
    
    # Add information about files modified in this session
    if file_history["modified"]:
        context_lines.append(f"\n{Fore.YELLOW}Files MODIFIED this session:{Style.RESET_ALL}")
        for file in file_history["modified"]:
            context_lines.append(f"- {Fore.WHITE}{file}{Style.RESET_ALL}")
    
    if not file_history["created"] and not file_history["modified"]:
        context_lines.append(f"\n{Fore.CYAN}No files created or modified yet this session.{Style.RESET_ALL}")
    
    # Add CRITICAL warning about using correct filenames
    context_lines.append(f"\n{Fore.RED}⚠️ CRITICAL:{Style.RESET_ALL} Always use exact filenames from context.")
    
    # Add the CURRENT CONTENT of all created/modified files to provide accurate context
    context_lines.append(f"\n{Style.BRIGHT}{Fore.MAGENTA}--- CURRENT FILE CONTENTS ---{Style.RESET_ALL}")
    all_files = sorted(list(set(file_history["created"] + file_history["modified"])))
    
    if not all_files:
         context_lines.append(f"{Fore.CYAN}No files to show content for yet.{Style.RESET_ALL}")
    else:
        for filename in all_files:
            try:
                if os.path.exists(filename):
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Add file delimiters and content
                        context_lines.append(f"\n{Fore.CYAN}=== START: {filename} ==={Style.RESET_ALL}")
                        # Limit preview in context box for brevity? Or show all? Let's show all for now.
                        context_lines.append(f"```")
                        context_lines.extend(content.splitlines()) # Add lines individually
                        context_lines.append(f"```")
                        context_lines.append(f"{Fore.CYAN}=== END: {filename} ==={Style.RESET_ALL}")
                else:
                    context_lines.append(f"\n{Fore.YELLOW}--- {filename} --- [FILE DELETED/MOVED]{Style.RESET_ALL}")
            except Exception as e:
                    context_lines.append(f"\n{Fore.RED}--- {filename} --- [ERROR READING: {str(e)}]{Style.RESET_ALL}")

    # Return the formatted box string - Note: This might get long!
    # Instead of boxing the whole context, maybe just return lines and box it where it's used?
    # For now, let's return the raw context string, boxing is better done dynamically.
    # print_boxed("File Context", "\n".join(context_lines), color=Fore.MAGENTA, width=100) # Example if we wanted to print it
    
    # Let's add simple delimiters instead of boxing the whole context which could be huge
    context_header = f"{Style.BRIGHT}{Fore.MAGENTA}--- FILE CONTEXT ---{Style.RESET_ALL}\n"
    context_footer = f"\n{Style.BRIGHT}{Fore.MAGENTA}--- END FILE CONTEXT ---{Style.RESET_ALL}"
    return context_header + "\n".join(context_lines) + context_footer

def retry_replacements(file_operations, model, user_query, file_history):
    """Retry failed replacements with more context to the model."""
    # Check which operations *still* failed after the apply_changes attempt
    failed_operations = []
    for op in file_operations: # These are ops that failed apply_changes
        if op["type"] == "replace":
            try:
                 # Double check if it exists now (maybe indentation handler worked?)
                with open(op["filename"], "r") as f:
                    content = f.read()
                if op["old_content"] not in content:
                    # Still not found exactly, add to retry list
                    failed_operations.append(op)
                else:
                    print(f"{Fore.YELLOW}  Note: Operation for {op['filename']} seems corrected now.{Style.RESET_ALL}")

            except FileNotFoundError:
                print(f"{Fore.YELLOW}  Note: File {op['filename']} not found for retry check.{Style.RESET_ALL}")
                continue # Skip if file doesn't exist
            except Exception as e:
                print(f"{Fore.RED}  Error re-checking failed op {op['filename']}: {e}{Style.RESET_ALL}")
                failed_operations.append(op) # Retry if check failed
    
    if not failed_operations:
        # Reset attempt count if nothing needs retrying
        if hasattr(retry_replacements, 'attempt_count'):
            retry_replacements.attempt_count = 0
        print(f"{Fore.GREEN}No remaining failed replacements to retry.{Style.RESET_ALL}")
        return  # Nothing to retry
    
    print(f"\n{Fore.YELLOW}Retrying {len(failed_operations)} failed REPLACE operation(s)...{Style.RESET_ALL}")
    
    # ... (rest of retry logic: attempt count, building message, calling model) ...

    # Important: Make sure the retry prompt also emphasizes EXACT match requirement
    retry_message = f"⚠️ Attempt {getattr(retry_replacements, 'attempt_count', 1)}/3: The following REPLACE operations failed because the 'old content' block was not found exactly as provided. \n"
    retry_message += "Please carefully review the CURRENT FILE CONTENTS provided in the context and generate new [REPLACE] operations with the *exact* content (including whitespace and indentation) that exists in the file.\n\n"
    
    for op in failed_operations:
        retry_message += f"File: {op['filename']}\n"
        retry_message += f"AI-Provided Content Not Found:\n```\n{op['old_content']}\n```\n\n"
    
    retry_message += "Provide corrected [REPLACE] blocks below:"

    # ... (rest of retry: get system prompt, file context, call model, process response) ...

def process_mentions(user_input):
    """
    Find @path/to/file mentions in user input, read the files,
    and prepend their content to the input string for the model.
    """
    mention_pattern = r"@([\w\/\.\-\_]+)" # Regex to find @ followed by file path characters
    mentions = re.findall(mention_pattern, user_input)
    
    prepended_content = ""
    mentioned_files = set() # Keep track to avoid duplicates

    if not mentions:
        return user_input # No mentions found, return original input

    print(f"{Style.DIM}--- Processing Mentions ---{Style.RESET_ALL}")
    
    for filepath in mentions:
        if filepath in mentioned_files:
            continue # Skip already processed mentions
            
        full_path = os.path.abspath(filepath) # Get absolute path
        
        if os.path.exists(full_path) and os.path.isfile(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                print(f"{Fore.CYAN}  Injecting content from: {filepath}{Style.RESET_ALL}")
                
                prepended_content += f"\n{Style.BRIGHT}{Fore.MAGENTA}--- MENTIONED FILE: {filepath} ---{Style.RESET_ALL}\n"
                prepended_content += f"```\n{file_content}\n```\n"
                prepended_content += f"{Style.BRIGHT}{Fore.MAGENTA}--- END MENTIONED FILE: {filepath} ---{Style.RESET_ALL}\n\n"
                
                mentioned_files.add(filepath)

            except UnicodeDecodeError:
                print(f"{Fore.YELLOW}  Warning: Cannot read mentioned binary file: {filepath}{Style.RESET_ALL}")
                prepended_content += f"\n{Fore.YELLOW}[CodAgent Note: Mentioned file '{filepath}' is likely binary and could not be read.]{Style.RESET_ALL}\n\n"
                mentioned_files.add(filepath)
            except Exception as e:
                print(f"{Fore.RED}  Error reading mentioned file {filepath}: {e}{Style.RESET_ALL}")
                prepended_content += f"\n{Fore.RED}[CodAgent Note: Error reading mentioned file '{filepath}'.]{Style.RESET_ALL}\n\n"
                mentioned_files.add(filepath)
        else:
            print(f"{Fore.YELLOW}  Warning: Mentioned file not found or is not a file: {filepath}{Style.RESET_ALL}")
            # Optionally inform the model the file wasn't found
            prepended_content += f"\n{Fore.YELLOW}[CodAgent Note: Mentioned file '{filepath}' not found.]{Style.RESET_ALL}\n\n"
            mentioned_files.add(filepath) # Add even if not found to avoid reprocessing

    if prepended_content:
         print(f"{Style.DIM}--- End Processing Mentions ---{Style.RESET_ALL}")
         # Return the prepended content followed by the original user input
         return prepended_content + user_input
    else:
        # Return original input if no valid mentions were processed
        return user_input

def chat_with_model(model):
    """Start an interactive chat with the model."""
    # History file in the current directory
    history_file = os.path.join(os.getcwd(), ".chat.history.codagent")
    
    # Track added files/folders for context management
    added_context = {}
    
    # Track all files created or modified during this session
    file_history = {
        "created": [],
        "modified": [],
        "current_workspace": []
    }
    
    # Initialize the file history with existing files in the workspace
    for file in glob.glob("**/*", recursive=True):
        if os.path.isfile(file) and not file.startswith('.') and not file.startswith('__pycache__'):
            file_history["current_workspace"].append(file)
    
    # Check if we have previous chat history to load
    previous_conversation = []
    if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
        print(f"{Fore.GREEN}Found existing chat history in this directory.{Style.RESET_ALL}")
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                # Read the last 20 lines from history to get recent context
                lines = f.readlines()
                if lines:
                    # Extract the last few user inputs from history
                    user_inputs = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
                    if user_inputs:
                        # Add the last few exchanges to initialize conversation context
                        previous_conversation = [
                            {"role": "user", "parts": [user_inputs[-1]]},
                            {"role": "model", "parts": ["I remember our previous conversation in this directory. How can I help you today?"]}
                        ]
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load previous conversation context: {e}{Style.RESET_ALL}")
    
    # Create history file if it doesn't exist
    Path(history_file).touch(exist_ok=True)
    
    # Custom styling for the prompt using prompt_toolkit's native styling
    # Using bold white for the prompt itself for visibility
    style = PromptStyle.from_dict({
        'prompt': 'bold #ffffff', 
    })
    
    print(f"\n{Style.BRIGHT}{Fore.CYAN}=== CodAgent Interactive Session ==={Style.RESET_ALL}\n")
    print(f"{Fore.GREEN}Type '/help' for commands, 'exit' or Ctrl+C to quit.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Working Directory: {Fore.WHITE}{os.getcwd()}{Style.RESET_ALL}")
    print("-" * 40) # Separator
    
    # Keep track of conversation to maintain context
    conversation_history = []
    
    while True:
        try:
            # Use plain prompt text without colorama formatting
            # Define rprompt using ANSI wrapper for correct color handling
            rprompt_text = f"[{Fore.CYAN}{os.path.basename(os.getcwd())}{Style.RESET_ALL}]"
            raw_user_input = prompt(
                "CodAgent >>> ",
                history=FileHistory(history_file),
                auto_suggest=AutoSuggestFromHistory(),
                style=style,
                rprompt=ANSI(rprompt_text) # Use ANSI wrapper here
            )
            
            if raw_user_input.lower().strip() in ['exit', 'quit', 'q']:
                print(f"{Fore.YELLOW}Exiting CodAgent session.{Style.RESET_ALL}")
                break
            
            if not raw_user_input.strip(): # Skip empty input
                continue

            # --- Process mentions BEFORE adding to history or processing commands ---
            user_input_processed = process_mentions(raw_user_input)
            
            # Add the potentially modified user message to conversation history
            conversation_history.append({"role": "user", "content": user_input_processed})

            # Process commands using the raw_user_input
            if raw_user_input.startswith('/'):
                command_processed = True # Flag to check if we should skip model call
                if raw_user_input.startswith('/add '):
                    target = raw_user_input[5:].strip()
                    content = process_add_command(target) # Assuming this function prints its own status
                    if content:
                        added_context[target] = content
                        print(f"{Fore.GREEN}✓ Added context from:{Style.RESET_ALL} {target}")
                        # Maybe add a system message instead of formulating a user query?
                        conversation_history.append({"role": "system", "content": f"Context from '{target}' was added."})
                    else:
                        print(f"{Fore.RED}✗ Failed to add context from:{Style.RESET_ALL} {target}")
            
                elif raw_user_input.startswith('/remove '):
                    target = raw_user_input[8:].strip()
                    if target in added_context:
                        del added_context[target]
                        print(f"{Fore.YELLOW}Removed {target} from the conversation context.{Style.RESET_ALL}")
                        # Let the model know what's been removed
                        conversation_history.append({"role": "system", "content": f"User removed the file/directory '{target}' from our conversation context. Please disregard it in future responses."})
                    else:
                        print(f"{Fore.RED}Error: {target} was not found in the current context.{Style.RESET_ALL}")
                        print(f"{Fore.YELLOW}Current context includes: {', '.join(added_context.keys()) if added_context else 'nothing'}{Style.RESET_ALL}")

                elif raw_user_input.startswith('/list'):
                    if added_context:
                        print(f"\n{Fore.CYAN}Files/directories in current context:{Style.RESET_ALL}")
                        for idx, item in enumerate(added_context.keys(), 1):
                            print(f"{Fore.GREEN}{idx}.{Style.RESET_ALL} {item}")
                        print()
                    else:
                        print(f"\n{Fore.YELLOW}No files or directories in current context.{Style.RESET_ALL}\n")
                    continue
                
                elif raw_user_input.startswith('/files'):
                    print(f"\n{Fore.CYAN}Files created in this session:{Style.RESET_ALL}")
                    if file_history["created"]:
                        for idx, file in enumerate(file_history["created"], 1):
                            print(f"{Fore.GREEN}{idx}.{Style.RESET_ALL} {file}")
                    else:
                        print(f"{Fore.YELLOW}No files created yet.{Style.RESET_ALL}")
                    
                    print(f"\n{Fore.CYAN}Files modified in this session:{Style.RESET_ALL}")
                    if file_history["modified"]:
                        for idx, file in enumerate(file_history["modified"], 1):
                            print(f"{Fore.YELLOW}{idx}.{Style.RESET_ALL} {file}")
                    else:
                        print(f"{Fore.YELLOW}No files modified yet.{Style.RESET_ALL}")
                    print()
                    continue
            
                elif raw_user_input.startswith('/refresh'):
                    print(f"\n{Fore.CYAN}Refreshing file content awareness...{Style.RESET_ALL}")
                    # Force re-read all files to refresh context
                    for filename in set(file_history["created"] + file_history["modified"]):
                        if os.path.exists(filename):
                            try:
                                with open(filename, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    print(f"{Fore.GREEN}Refreshed:{Style.RESET_ALL} {filename}")
                            except Exception as e:
                                print(f"{Fore.RED}Error reading {filename}: {e}{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.YELLOW}Missing:{Style.RESET_ALL} {filename} (file no longer exists)")
                    
                    # Add message to conversation history about the refresh
                    conversation_history.append({"role": "system", "content": "I've refreshed your awareness of all file contents. Please use the current file contents for any operations."})
                    print(f"\n{Fore.GREEN}All files refreshed in model's context.{Style.RESET_ALL}\n")
                
                elif raw_user_input.startswith('/help'):
                    help_text = """
Available Commands:
/add <file_or_dir>    - Add file or folder content to the conversation.
/// Example: /add src/main.py
/remove <file_or_dir> - Remove a file or folder from the context.
/// Example: /remove src/main.py
/list                - List all files/folders in the current context.
/files               - List all files created or modified during this session.
/refresh             - Update the model's awareness with the latest file contents.
/help                - Show this help message.
exit, quit, q       - Exit CodAgent.
"""
                    print(help_text)
                    continue
                else:
                    print(f"{Fore.RED}Unknown command: {raw_user_input}{Style.RESET_ALL}")
                    command_processed = False # Treat as regular input if command is unknown

                is_local_only_command = raw_user_input.startswith(('/list', '/files', '/help', '/remove')) 
                if command_processed and is_local_only_command:
                   if conversation_history and conversation_history[-1]["role"] == "user":
                       conversation_history.pop() 
                   continue 

            # --- Model Interaction ---
            
            # Generate the file context to include with the system prompt
            file_context = generate_file_context(file_history)
            system_prompt = get_system_prompt()
            
            # Construct the prompt for the model
            model_prompt_parts = [system_prompt, file_context] 
            
            # Add relevant conversation history
            history_to_include = min(10, len(conversation_history)) 
            prompt_history = []
            for i in range(len(conversation_history) - history_to_include, len(conversation_history)):
                entry = conversation_history[i]
                role = entry['role'].upper()
                if role == 'SYSTEM':
                     role = 'SYSTEM NOTE' 
                prompt_history.append(f"{role}: {entry['content']}")
            
            if prompt_history:
                 model_prompt_parts.append(f"{Style.BRIGHT}{Fore.MAGENTA}--- CONVERSATION HISTORY ---{Style.RESET_ALL}\n" + "\n\n".join(prompt_history))
                 model_prompt_parts.append(f"{Style.BRIGHT}{Fore.MAGENTA}--- END HISTORY ---{Style.RESET_ALL}")

            final_prompt_string = "\n\n".join(model_prompt_parts)

            # Store complete raw responses for this turn
            all_responses_this_turn = []
            is_end_of_turn = False
            
            print(f"\n{Style.DIM}--- CodAgent Thinking ---{Style.RESET_ALL}")
            
            current_prompt_for_model = final_prompt_string 
            
            # Loop for multi-step responses until [END] is received
            while not is_end_of_turn:
                # Print segment header
                print(f"\n{Style.BRIGHT}{Fore.GREEN}>>> AI Response Segment >>>{Style.RESET_ALL}")

                current_segment_text = ""
                stream_error_occurred = False
                try:
                    # --- Streaming Generation ---
                    response_stream = model.generate_content(current_prompt_for_model, stream=True)
                    for chunk in response_stream:
                        try:
                            chunk_text = chunk.text
                            print(chunk_text, end='', flush=True)
                            current_segment_text += chunk_text
                        except ValueError:
                            print(f"\n{Fore.YELLOW}[CodAgent Note: Received non-text chunk...] {Style.RESET_ALL}", flush=True)
                        except Exception as e_text_access:
                             print(f"\n{Fore.RED}Error processing stream chunk: {e_text_access}{Style.RESET_ALL}", flush=True)
                    print() # Newline after segment stream

                except Exception as model_error:
                     print(f"\n{Back.RED}{Fore.WHITE} ERROR during streaming request: {model_error} {Style.RESET_ALL}")
                     current_segment_text = "[END]"
                     is_end_of_turn = True
                     stream_error_occurred = True

                # Check for [END] tag
                if not stream_error_occurred:
                     _, is_end_of_turn_from_tag = parse_end_response(current_segment_text)
                     if is_end_of_turn_from_tag:
                          is_end_of_turn = True

                all_responses_this_turn.append(current_segment_text)

                if not is_end_of_turn:
                    print(f"\n{Style.DIM}--- CodAgent Continuing ---{Style.RESET_ALL}")
                    current_prompt_for_model += f"\n\nMODEL: {current_segment_text}\n\nCONTINUE:"
                else:
                     # --- REMOVE Boxing of Final AI Response ---
                     # The segments were already printed above. No need to print combined again.
                     # full_segment_text_cleaned = parse_end_response("\n".join(all_responses_this_turn))[0]
                     # print() # Add newline before box
                     # print_boxed("CodAgent Final Response", full_segment_text_cleaned, color=Fore.GREEN) # REMOVED THIS
                     print(f"\n{Style.DIM}--- CodAgent Finished Turn ---{Style.RESET_ALL}")

            # --- End of Agent Turn ---
            
            # Combine all raw responses from this turn for parsing operations
            full_raw_response = "\n\n".join(all_responses_this_turn)
            
            # Get the final cleaned response for history
            final_cleaned_response = parse_end_response(full_raw_response)[0] 
            if final_cleaned_response:
                conversation_history.append({"role": "model", "content": final_cleaned_response})

            # --- Post-Response Processing (using full_raw_response for parsing) ---
            
            # Parse terminal commands
            terminal_commands = parse_terminal_commands(full_raw_response) 
            if terminal_commands:
                # Box the proposed commands
                command_list_str = "\n".join([f"  [{idx+1}] {cmd}" for idx, cmd in enumerate(terminal_commands)])
                print() 
                print_boxed("Proposed Terminal Commands", command_list_str, color=Fore.YELLOW)
                # Confirmation outside box
                confirm = input(f"\n{Style.BRIGHT}{Back.RED}{Fore.WHITE} WARNING: Execute these terminal commands? (y/n): {Style.RESET_ALL} ").lower().strip()
                if confirm == 'y':
                    # execute_terminal_command prints its own box
                    # ... (execution logic) ...
                    command_results = []
                    # print(f"\n{Style.BRIGHT}{Fore.CYAN}=== Executing Terminal Commands ==={Style.RESET_ALL}") # execute_terminal_command prints header now
                    for cmd in terminal_commands:
                        result = execute_terminal_command(cmd)
                        command_results.append({ "command": cmd, "result": result })
                    # ... (add results to history) ...
                else:
                    # ... (handle skipped) ...
                    print(f"{Fore.YELLOW}✗ Terminal commands skipped by user.{Style.RESET_ALL}")
                    conversation_history.append({"role": "system", "content": "User chose not to execute the proposed terminal commands."})


            # Parse file operations (now correctly handles stripped content)
            file_operations = parse_file_operations(full_raw_response)
            applied_file_ops = False
            final_apply_result = {"successful": [], "failed": []} 
            if file_operations:
                # preview_changes prints its own box
                if preview_changes(file_operations):
                    # apply_changes prints its own box
                    result = apply_changes(file_operations) 
                    final_apply_result = result 
                    applied_file_ops = True
                    # ... (add summary to history, handle retries) ...
                    op_summary = f"File Operations Attempted:\nSuccessful: {len(result['successful'])}, Failed: {len(result['failed'])}"
                    conversation_history.append({"role": "system", "content": op_summary})
                    if result["failed"]:
                        print(f"{Fore.YELLOW}Attempting to retry failed file operations...{Style.RESET_ALL}")
                        retry_result = retry_replacements(result["failed"], model, user_input_processed, file_history) 
                    elif result["successful"]:
                        print(f"{Fore.GREEN}✓ All file operations completed successfully!{Style.RESET_ALL}")

                else:
                     # ... (handle skipped) ...
                     print(f"{Fore.YELLOW}✗ File operations skipped by user.{Style.RESET_ALL}")
                     conversation_history.append({"role": "system", "content": "User chose not to apply the proposed file operations."})

            # Update file history
            if applied_file_ops:
                 # ... (update file history based on final_apply_result) ...
                 for op in final_apply_result.get("successful", []): 
                        if op["type"] == "create":
                            if op["filename"] not in file_history["created"]:
                                file_history["created"].append(op["filename"])
                        elif op["type"] == "replace":
                            if op["filename"] not in file_history["modified"] and op["filename"] not in file_history["created"]:
                                file_history["modified"].append(op["filename"])
                    
            print("-" * 40) # End of turn separator
        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupt received. Exiting...{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n{Back.RED}{Fore.WHITE} UNEXPECTED ERROR: {e} {Style.RESET_ALL}")
            import traceback
            traceback.print_exc() 
            conversation_history.append({"role": "system", "content": f"An unexpected error occurred: {e}"})

def main():
    """Main entry point for the CLI."""
    # --- Clear Terminal ---
    os.system('cls' if os.name == 'nt' else 'clear')

    # --- Print Title ---
    title = r"""
 ██████╗  ██████╗ ██████╗  █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔════╝ ██╔═══██╗██╔══██╗██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
██║      ██║   ██║██║  ██║███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
██║      ██║   ██║██║  ██║██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
██║      ██║   ██║██████╔╝██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
 ╚██████ ╚██████╔╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝
"""
    print(f"{Fore.CYAN}{Style.BRIGHT}{title}{Style.RESET_ALL}")
    print("-" * 60) # Separator after title
    print(f"{Fore.GREEN}Welcome! Type '/help' for commands or 'exit' to quit.{Style.RESET_ALL}")
    print("-" * 60)


    parser = argparse.ArgumentParser(description="CodAgent - AI-powered code generation tool")
    parser.add_argument("--model", default="gemini-1.5-pro-latest", help="Google Gemini model to use")

    args = parser.parse_args()

    print(f"{Fore.CYAN}Initializing CodAgent with model: {Fore.GREEN}{args.model}{Style.RESET_ALL}")
    model = initialize_genai(args.model)
    chat_with_model(model)

if __name__ == "__main__":
    main() 