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
from prompt_toolkit.completion import Completer, Completion, PathCompleter, WordCompleter, CompleteEvent
from prompt_toolkit.document import Document
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
    # Use a more comprehensive regex for ANSI escape sequences.
    # Handles CSI sequences (like colors) and other escape types.
    ansi_escape = re.compile(r'''
        \x1B  # ESC
        (?:   # Non-capturing group
            [@-Z\\-_] | # Single character escapes
            \[ [0-?]* [ -/]* [@-~] # CSI sequence
        )
    ''', re.VERBOSE)
    return len(ansi_escape.sub('', text))

# --- Helper Function for Boxed Output ---
def print_boxed(title, content, color=Fore.CYAN, width=None):
    """Prints content inside a simulated rounded border, aware of ANSI codes."""
    # Get terminal width, default to 80 if unavailable or too small
    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 80
    max_width = width if width is not None else term_width
    max_width = max(max_width, 20) # Ensure a minimum reasonable width

    lines = content.splitlines()
    max_visible_line_width = max((visible_len(line) for line in lines), default=0)
    visible_title_width = visible_len(title)

    # Calculate necessary inner width, constrained by max_width
    required_inner_width = max(visible_title_width, max_visible_line_width)
    box_width = min(required_inner_width + 4, max_width) # Add padding+borders, limit by max_width
    inner_width = box_width - 4 # Final inner width based on constrained box_width

    # --- Top border ---
    print(f"{color}{TL}{H * (box_width - 2)}{TR}{Style.RESET_ALL}")

    # --- Title line ---
    title_padding_total = inner_width - visible_title_width
    title_pad_left = title_padding_total // 2
    title_pad_right = title_padding_total - title_pad_left
    print(f"{color}{V} {' ' * title_pad_left}{Style.BRIGHT}{title}{Style.NORMAL}{' ' * title_pad_right} {V}{Style.RESET_ALL}")

    # --- Separator ---
    print(f"{color}{V}{H * inner_width}{V}{Style.RESET_ALL}")

    # --- Content lines ---
    for line in lines:
        v_len = visible_len(line)
        padding_needed = inner_width - v_len
        # Ensure padding isn't negative if line somehow exceeds inner_width (shouldn't happen with wrapping)
        padding_needed = max(0, padding_needed)

        # Basic wrapping (split long lines) - based on visible length
        # This part is complex with ANSI codes, keep simple for now
        # TODO: Improve wrapping logic for lines containing ANSI codes if needed
        if v_len > inner_width:
             # Crude split for now, might break colors across lines
             # A proper ANSI-aware wrapper would be needed for perfection
             print(f"{color}{V} {line[:inner_width]}{' ' * (inner_width-visible_len(line[:inner_width]))} {V}{Style.RESET_ALL}") # Attempt to pad truncated line
             # Don't print rest for now to avoid complex state
             print(f"{color}{V} ... (line truncated) ... {' ' * (inner_width - 24)} {V}{Style.RESET_ALL}")
        else:
            # Print line with calculated padding
            print(f"{color}{V} {line}{' ' * padding_needed} {V}{Style.RESET_ALL}")


    # --- Bottom border ---
    print(f"{color}{BL}{H * (box_width - 2)}{BR}{Style.RESET_ALL}")

# --- Function to get Codebase Structure ---
def get_codebase_structure(startpath='.', ignore_dirs=None, ignore_files=None):
    """Generates a tree-like string representation of the directory structure."""
    if ignore_dirs is None:
        ignore_dirs = {'.git', '.vscode', '__pycache__', 'node_modules', '.idea', 'venv', '.env'}
    if ignore_files is None:
        ignore_files = {'.DS_Store'}

    tree = []
    for root, dirs, files in os.walk(startpath, topdown=True):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        # Filter files
        files = [f for f in files if f not in ignore_files and not f.startswith('.')]

        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        tree.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in sorted(files): # Sort files for consistent output
            tree.append(f"{subindent}{f}")

    # Remove the first line if it's just './' or '.'
    if tree and (tree[0] == './' or tree[0] == '.'):
        tree = tree[1:]

    return '\n'.join(tree)
# --- End Function ---

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

**Context Awareness:**
*   **File Mentions (User):** The user can mention files using `@path/to/file`. When they do, the content of that file is automatically provided to you in the prompt under a `--- MENTIONED FILE: ... ---` block. Use this content for context.
*   **Codebase Structure (User):** The user can use `@codebase` to provide you with a tree view of the project structure under a `--- CODEBASE STRUCTURE ---` block. Use this to understand the project layout.
*   **Referring to Files (AI):** In *your* responses, you can *refer* to files using the `@path/to/file` syntax (e.g., "I will modify `@src/utils.py`"). This helps the user understand your plan but does **not** automatically load the file content for you again. Rely on the context provided by user mentions or the `FILE CONTEXT` section.
*   **FILE CONTEXT Section:** Below the history, a `--- FILE CONTEXT ---` section shows the *current content* of files modified or created *during this session*. This section is **CRITICAL** and should be used as the **primary source** for the `...exact_content_to_replace...` part of any `====== REPLACE` operation on these files, as it reflects their most up-to-date state in the current session.
*   **Selected Files (User Action):** If you use the `====== ASK_FOR_FILES` tag, the user's selected files will be provided in the next turn under `--- SELECTED FILE: ... ---` blocks.

---

**CORE BEHAVIORS & CAPABILITIES:**

1.  **Autonomous Agent & `[END]` Tag Usage:** You decide when your response completes the *entire task* requested by the user.
    *   If you need more steps, explanation, or multiple operations (e.g., create a file, then modify it), simply end your response normally. You will be prompted to continue.
    *   Only when your entire thought process and *all* planned actions (code generation, file operations, terminal commands, explanations) for the *current user request* are fully finished, should you end your *final* message with the `[END]` tag. **Do not** use `[END]` prematurely after just one operation if more are needed to fulfill the user's overall goal.
    *   **Do not** use `[END]` if you are using `====== ASK_FOR_FILES`, as you need the user's response.
    *   **Example Multi-Turn (Illustrates correct `[END]` usage):**
        ```
        === User Request ===
        Create a simple Python script `app.py` that prints "Hello" and then add a function `greet(name)` to it.

        === AI Response 1 ===
        Okay, first I'll create the initial `app.py` file.
        ====== CREATE app.py
        print("Hello")
        ====== CEND
        Next, I will add the function.

        === AI Response 2 ===
        Now, using the content from the FILE CONTEXT, I'll add the `greet` function.
        ====== REPLACE app.py
        print("Hello")
        ====== TO
        print("Hello")

        def greet(name):
            print(f"Hello, {{name}}!")
        ====== REND
        The script is now complete with the initial print and the function. [END]
        ```

2.  **File Operations:** Use specific tags to manage files:
    *   **Create:**
        ```
        ====== CREATE path/to/filename.ext
        ...content...
        ====== CEND
        ```
        *(Content should NOT be wrapped in ``` unless the content itself is markdown)*
    *   **Replace:**
        ```
        ====== REPLACE path/to/filename.ext
        ...exact_content_to_replace...
        ====== TO
        ...new_content...
        ====== REND
        ```
        **IMPORTANT:** The format MUST be EXACTLY as shown.
        - Start with `====== REPLACE path/to/filename.ext` on its own line.
        - Follow with the `...exact_content_to_replace...` block.
        - Follow with `====== TO` on its own line.
        - Follow with the `...new_content...` block.
        - End with `====== REND` on its own line.
        - The `...exact_content_to_replace...` and `...new_content...` blocks themselves should **NOT** be wrapped in markdown code fences (```).
    *   **CRITICAL (REPLACE Source):** For the `...exact_content_to_replace...` block, **you MUST use the current file content**. Prioritize the `--- FILE CONTEXT ---` section if the file has been modified this session. If the file was provided via `--- MENTIONED FILE ---` or `--- SELECTED FILE ---`, use that. **Do not guess or hallucinate the content to be replaced.**
    *   **CRITICAL (REPLACE Structure):** The structure *must* be exactly as shown above. `====== REPLACE path/to/file.ext` *must* be the first line. `====== TO` *must* be on its own line between the old and new content. `====== REND` *must* be at the end.
    *   **CRITICAL (REPLACE Content Match):** The `...exact_content_to_replace...` block *must perfectly match* the existing code in the file (from context), **including all whitespace and indentation**.

3.  **Terminal Execution:** Execute shell commands:
    *   **Syntax:**
        ```
        ====== TERMINAL
        ...command...
        ====== TEND
        ```
    *   **Example:**
        ```
        ====== TERMINAL
        pip install requests
        ====== TEND
        ```
    *   **IMPORTANT:** Explain *why* a command is needed. The user *must* approve execution. You will receive the command's output (stdout/stderr) for context in the next turn's history (as a SYSTEM NOTE).

4.  **Asking for Files:** If you need the content of specific files to proceed (e.g., after reviewing `@codebase`), use the `====== ASK_FOR_FILES` tag.
    *   **Syntax:**
        ```
        ====== ASK_FOR_FILES
        path/relative/to/file1.ext
        path/to/file2.py
        ...
        ====== AEND
        ```
    *   List the desired files (relative paths) one per line between the tags.
    *   Explain *why* you need these files.
    *   The user will be prompted to select files, and their content will be provided in the next turn under `--- SELECTED FILE: ... ---` blocks.
    *   **Do not use `[END]` when using this tag.**

---

**⚠️ CRITICAL REQUIREMENTS - MUST FOLLOW ⚠️**

*   **USE PROVIDED CONTEXT:** Always base your file modifications (`====== REPLACE`) on the actual content provided in the `--- FILE CONTEXT ---`, `--- MENTIONED FILE ---`, or `--- SELECTED FILE ---` sections. Do not invent content.
*   **TAG STRUCTURE (REPLACE):**
    *   Ensure the `====== REPLACE` tag structure starts *exactly* with `====== REPLACE path/to/filename.ext` on the first line.
    *   Ensure `====== TO` is on its own line between old and new content.
    *   Ensure `====== REND` is on its own line at the end.
*   **CODE EXACTNESS (REPLACE):** The content between `====== REPLACE ...` and `====== TO` *must perfectly match* the existing code in the file (from context), character for character, including **all leading/trailing whitespace and indentation**.
*   **INDENTATION:** Preserve original indentation in `====== REPLACE` tags. For `====== CREATE` or new code in `====== REPLACE`, follow the existing file's indentation style or standard conventions (e.g., 4 spaces for Python).
*   **NO MARKDOWN CODE BLOCKS INSIDE TAGS:** **Do not** use triple backticks (```) to wrap the code *inside* the `...content...` sections of `====== CREATE` or `====== REPLACE` tags. The code should be placed directly within the tags. **Also, do not wrap the entire `====== TAG ... ====== TEND` block itself in ```.** Only use ``` if the file content *itself* is intended to be markdown text containing a code block. Use the specified tags (`====== CREATE`, `====== REPLACE`, `====== TERMINAL`, `====== ASK_FOR_FILES`) for operations.
*   **`[END]` TAG PLACEMENT:** Only use `[END]` in your *final response* for a given user request, after all explanations and operations are complete. **Never** use `[END]` if you have used `====== ASK_FOR_FILES` in the same response.

---

**GENERAL GUIDELINES:**

*   **Clarity:** Explain your plan and the purpose of your code/commands/requests. Refer to files using `@path/to/file` where appropriate.
*   **Context Reliance:** Base your actions and especially file edits on the provided `--- MENTIONED FILE ---`, `--- CODEBASE STRUCTURE ---`, `--- FILE CONTEXT ---`, `--- SELECTED FILE ---`, and `--- CONVERSATION HISTORY ---` sections.
*   **Inferring Target File (from @codebase):** If the user provides `@codebase` context but asks for a general change without specifying a file (e.g., "improve the python code"), *attempt* to identify a single, likely target file from the structure (e.g., `main.py`, `app.py`, or the only `.py` file). **State your assumed target file** (e.g., "Assuming you want to modify `@app.py`, I will...") before proceeding. If you cannot make a reasonable assumption (e.g., multiple candidates, no clear entry point), then ask the user for clarification using the `@path/to/file` syntax or use `====== ASK_FOR_FILES` if you have specific candidates.
*   **Completeness:** Provide functional code snippets or commands.
*   **Safety:** Be cautious with terminal commands, especially those that modify files or system state (`rm`, `mv`, etc.). Always explain the command's effect.
*   **Ask (Ambiguity):** If the provided context seems insufficient or ambiguous for a requested edit *even after attempting inference*, ask for clarification (e.g., using `====== ASK_FOR_FILES`) instead of proceeding with potentially incorrect changes.

---

**USER WORKFLOW:**

1.  User provides a request, possibly using `@mentions`.
2.  CodAgent processes mentions, prepends context, and sends the full prompt to you.
3.  You analyze, potentially ask questions (or state assumptions if inferring), and respond with explanations and tagged operations (`====== CREATE`, `====== REPLACE`, `====== TERMINAL`, `====== ASK_FOR_FILES`). You might use `====== ASK_FOR_FILES` here.
4.  **If `====== ASK_FOR_FILES` is used:** CodAgent prompts the user to select files. The selected file content is added to the context for the next turn.
5.  If your response does not end with `[END]` (and didn't use `====== ASK_FOR_FILES`), CodAgent prompts you to continue with the existing context.
6.  Once you provide `[END]`, CodAgent previews the collected operations/commands to the user.
7.  User approves or rejects.
8.  If approved, CodAgent applies changes/executes commands. Terminal output/operation results are added to history as SYSTEM NOTES. *If REPLACE fails, CodAgent may automatically ask you to retry.*
9.  CodAgent waits for the user's next request.

---

Remember to use the provided **CONTEXT SECTIONS** below.
"""
    return system_prompt

# --- Add Parsing Function ---
def parse_ask_for_files(response_text):
    """Parse the response text to extract suggested files from ====== ASK_FOR_FILES tag."""
    # Use re.MULTILINE and re.DOTALL. Match content between the tags.
    ask_pattern = r"^====== ASK_FOR_FILES\s*\n(.*?)\n====== AEND\s*$"
    match = re.search(ask_pattern, response_text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        files = [line.strip() for line in content.splitlines() if line.strip()]
        if files:
            response_without_tag = response_text[:match.start()] + response_text[match.end():]
            return files, response_without_tag.strip()
    return None, response_text # Return None if tag not found or empty
# --- End Add Parsing Function ---

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

    # Find TERMINAL commands using the new format
    # Use re.MULTILINE and re.DOTALL
    terminal_pattern = r"^====== TERMINAL\s*\n(.*?)\n====== TEND\s*$"
    for match in re.finditer(terminal_pattern, response_text, re.DOTALL | re.MULTILINE | re.IGNORECASE):
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

    # **** ADDED: Explicit command line inside the box content ****
    exec_log.append(f"{Style.BRIGHT}Command:{Style.RESET_ALL} {command}")
    exec_log.append(H * visible_len(f"Command: {command}")) # Add a separator line
    # **** END ADDED ****

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
        # Add error *after* the command line inside the box
        exec_log.append(f"{Fore.RED}✗ Error executing command:{Style.RESET_ALL} {e}")
        return_code = -1 # Indicate failure

    # Print execution log in a box
    # Simplified title slightly, command is now clearly inside the box content.
    box_color = Fore.RED if return_code != 0 else Fore.GREEN
    print_boxed(f"Execution Result", "\n".join(exec_log), color=box_color)
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
    """Parse the response text to extract file operations using the new format."""
    # Clean the response text (remove [END] tag if present) before parsing ops
    cleaned_response, _ = parse_end_response(response_text)

    # ***** ADDED: Pre-strip potential outer fences from the cleaned response *****
    cleaned_response = strip_code_fences(cleaned_response)
    # ***** END ADDED *****

    file_operations = []

    # Find CREATE operations using the new format
    # Use re.MULTILINE and re.DOTALL. Group 1: filename, Group 2: content
    create_pattern = r"^====== CREATE\s+([^\n]+)\n(.*?)\n====== CEND\s*$"
    # Now search within the potentially pre-stripped cleaned_response
    for match in re.finditer(create_pattern, cleaned_response, re.DOTALL | re.MULTILINE | re.IGNORECASE):
        filename = match.group(1).strip()
        raw_content = match.group(2) # Content is group 2
        # Remove potential leading/trailing empty lines from capture
        # Apply strip_code_fences *again* here in case fences were inside the content block itself
        content = strip_code_fences(raw_content.strip())
        if content: # Add operation only if content is not empty after stripping
            file_operations.append({
                "type": "create",
                "filename": filename,
                "content": content
            })
        else:
             print(f"{Fore.YELLOW}Warning: Skipping CREATE operation for '{filename}' because content was empty after stripping.{Style.RESET_ALL}")


    # Find REPLACE operations using the new format
    # Use re.MULTILINE and re.DOTALL. Group 1: filename, Group 2: old_content, Group 3: new_content
    replace_pattern = r"^====== REPLACE\s+([^\n]+)\n(.*?)\n====== TO\n(.*?)\n====== REND\s*$"
     # Now search within the potentially pre-stripped cleaned_response
    for match in re.finditer(replace_pattern, cleaned_response, re.DOTALL | re.MULTILINE | re.IGNORECASE):
        filename = match.group(1).strip()
        raw_old_content = match.group(2) # Old content is group 2
        raw_new_content = match.group(3) # New content is group 3

        # Remove potential leading/trailing empty lines from capture before stripping fences
        # Apply strip_code_fences *again* here to handle fences inside old/new blocks
        old_content = strip_code_fences(raw_old_content.strip())
        new_content = strip_code_fences(raw_new_content.strip())

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

def retry_failed_replacements(failed_ops, model, file_history, conversation_history, max_retries=2):
    """Attempts to automatically retry failed REPLACE operations."""
    retry_attempt = 1
    remaining_failed = list(failed_ops) # Work on a copy
    newly_successful = []
    final_failed = []

    while remaining_failed and retry_attempt <= max_retries:
        print(f"\n{Fore.YELLOW}--- Attempting Auto-Retry {retry_attempt}/{max_retries} for Failed REPLACES ---{Style.RESET_ALL}")

        # --- Construct Retry Prompt ---
        retry_prompt_parts = []
        retry_prompt_parts.append(get_system_prompt()) # Use the latest system prompt rules
        retry_prompt_parts.append(generate_file_context(file_history)) # CRITICAL: Provide CURRENT context

        # Add recent history for context (optional, might make prompt too long)
        # Limit history more aggressively during retry to reduce confusion?
        history_to_include = min(3, len(conversation_history)) # Reduced history length
        if history_to_include > 0:
             prompt_history_lines = []
             start_index = len(conversation_history) - history_to_include
             for i in range(start_index, len(conversation_history)):
                 entry = conversation_history[i]
                 # Skip previous failed retry attempts in this mini-history? Maybe not necessary.
                 role_prefix = f"{entry['role'].upper()}: "
                 prompt_history_lines.append(role_prefix + entry['content'])
             retry_prompt_parts.append(f"{Style.BRIGHT}{Fore.MAGENTA}--- RECENT CONVERSATION HISTORY ---{Style.RESET_ALL}\n" + "\n\n".join(prompt_history_lines))
             retry_prompt_parts.append(f"\n{Style.BRIGHT}{Fore.MAGENTA}--- END HISTORY ---{Style.RESET_ALL}")


        # --- ***** VERY EXPLICIT RETRY MESSAGE ***** ---
        retry_message = f"**Retry Request (Attempt {retry_attempt}):** Your previous attempt to use `====== REPLACE` for the file(s) below **FAILED**. The 'old content' block you provided (between `====== REPLACE ...` and `====== TO`) was **INCORRECT** and did not match the actual file content.\n\n"
        retry_message += f"**CRITICAL INSTRUCTION - READ CAREFULLY:**\n"
        retry_message += f"1.  **IGNORE YOUR PREVIOUS ATTEMPT.** The 'old content' you used was **WRONG**. Do not use it again.\n"
        retry_message += f"2.  **LOOK CAREFULLY** at the `--- FILE CONTEXT ---` section provided *above* this message. It contains the **CURRENT and CORRECT** content of the files.\n"
        retry_message += f"3.  For **each file** listed below, find the **EXACT BLOCK OF TEXT** within its `--- FILE CONTEXT ---` that you want to replace.\n"
        retry_message += f"4.  Construct the **ENTIRE** `====== REPLACE` tag again using that **EXACT TEXT** (including all original indentation and whitespace) between `====== REPLACE filename` and `====== TO`.\n\n"

        files_to_retry = {op['filename'] for op in remaining_failed}
        for filename in files_to_retry:
            retry_message += f"**File to Correct:** `{filename}`\n"
            retry_message += f"  *Find the correct content block for `{filename}` in the `--- FILE CONTEXT ---` section above.*\n\n"

        retry_message += "**Provide ONLY the corrected `====== REPLACE` tag(s) below. Ensure the 'old content' block perfectly matches the text from the `--- FILE CONTEXT ---`. Do NOT include explanations or `[END]` tags.**"
        # --- ***** END VERY EXPLICIT RETRY MESSAGE ***** ---


        retry_prompt_parts.append(f"{Fore.RED}{Style.BRIGHT}{retry_message}{Style.RESET_ALL}") # Make retry request very prominent

        full_retry_prompt = "\n\n".join(retry_prompt_parts)

        # Add a system message about the retry attempt to history
        conversation_history.append({"role": "system", "content": f"Initiating auto-retry {retry_attempt}/{max_retries} for {len(remaining_failed)} failed REPLACE operations. Emphasizing use of FILE CONTEXT."}) # Added note

        # --- Call Model ---
        print(f"{Style.DIM}--- Asking AI for corrected REPLACE tags... ---{Style.RESET_ALL}")
        retry_response_text = ""
        try:
            # Use non-streaming for simpler retry logic for now
            retry_response = model.generate_content(full_retry_prompt)
            # Access text safely, handling potential None or missing attribute
            retry_response_text = getattr(retry_response, 'text', '')
            if not retry_response_text:
                # Check parts if text is empty (common with Gemini API structure)
                if hasattr(retry_response, 'parts'):
                    retry_response_text = "".join(part.text for part in retry_response.parts if hasattr(part, 'text'))

            print(f"{Fore.CYAN}AI Retry Response:\n{retry_response_text}{Style.RESET_ALL}")
             # Add AI's retry response to history
            conversation_history.append({"role": "model", "content": f"[Retry {retry_attempt} Response]\n{retry_response_text}"})
        except Exception as e:
            print(f"{Back.RED}{Fore.WHITE} ERROR during retry generation: {e} {Style.RESET_ALL}")
            # Add error note to history
            conversation_history.append({"role": "system", "content": f"Error during retry attempt {retry_attempt}: {e}"})
            # Stop retrying if the model fails
            final_failed.extend(remaining_failed) # Assume all remaining ops failed
            remaining_failed = []
            break

        # --- Process Retry Response ---
        # Parse *only* replace operations from the retry response
        retry_file_ops = [op for op in parse_file_operations(retry_response_text) if op['type'] == 'replace']

        if not retry_file_ops:
            print(f"{Fore.YELLOW}No valid REPLACE tags found in AI's retry response.{Style.RESET_ALL}")
            # Keep remaining_failed as they are, try again next loop if attempts remain
        else:
            print(f"{Fore.CYAN}Applying corrections from retry attempt {retry_attempt}...{Style.RESET_ALL}")
            # Preview is skipped for auto-retry for simplicity
            retry_apply_result = apply_changes(retry_file_ops) # Apply only the new ops

            # Update history with retry apply results
            op_summary_lines = [f"Auto-Retry {retry_attempt} Apply Results:"]
            successful_filenames_this_retry = {op['filename'] for op in retry_apply_result.get('successful', [])}
            if retry_apply_result['successful']:
                 op_summary_lines.append(f"  {Fore.GREEN}Successful ({len(retry_apply_result['successful'])}):{Style.RESET_ALL} {', '.join(successful_filenames_this_retry)}")
                 newly_successful.extend(retry_apply_result['successful'])
            if retry_apply_result['failed']:
                 op_summary_lines.append(f"  {Fore.RED}Failed ({len(retry_apply_result['failed'])}):{Style.RESET_ALL} {', '.join([op['filename'] for op in retry_apply_result['failed']])}")
            conversation_history.append({"role": "system", "content": "\n".join(op_summary_lines)})

            # Update remaining_failed: remove ops for files that succeeded in this retry attempt
            current_remaining = []
            for op in remaining_failed:
                 if op['filename'] not in successful_filenames_this_retry:
                      current_remaining.append(op)
            remaining_failed = current_remaining

        retry_attempt += 1
        # Small delay before next retry if needed
        if remaining_failed and retry_attempt <= max_retries:
             time.sleep(1)

    # After loop: any ops still in remaining_failed are now finally failed
    final_failed.extend(remaining_failed)

    print(f"{Fore.YELLOW}--- Auto-Retry Finished ---{Style.RESET_ALL}")
    if newly_successful:
         print(f"{Fore.GREEN}Successfully applied corrections for: {', '.join(list({op['filename'] for op in newly_successful}))}{Style.RESET_ALL}")
    if final_failed:
         print(f"{Fore.RED}Still failed after retries: {', '.join(list({op['filename'] for op in final_failed}))}{Style.RESET_ALL}")


    # Return lists of operations that were newly successful or ultimately failed after retries
    return {"newly_successful": newly_successful, "final_failed": final_failed}

def process_mentions(user_input):
    """
    Find @path/to/file mentions and @codebase in user input, read files/get structure,
    and prepend their content to the input string for the model.
    Returns the processed input (with prepended content) and the original input
    with mentions removed (for history).
    """
    mention_pattern = r"(@[\w\/\.\-\_]+)" # Regex to find @ followed by path chars or 'codebase'
    mentions = re.findall(mention_pattern, user_input)

    prepended_content = ""
    mentioned_files = set() # Keep track to avoid duplicates
    processed_mention_spans = [] # Store (start, end) of processed mentions

    if not mentions:
        return user_input, user_input # No mentions found

    print(f"{Style.DIM}--- Processing Mentions ---{Style.RESET_ALL}")

    # Use finditer to get match objects with positions
    for match in re.finditer(mention_pattern, user_input):
        mention_text = match.group(1) # The full mention, e.g., "@path/to/file"
        raw_target = mention_text[1:] # Remove the leading '@'

        if raw_target == "codebase":
            if "codebase" not in mentioned_files: # Process only once
                print(f"{Fore.CYAN}  Injecting codebase structure...{Style.RESET_ALL}")
                codebase_structure = get_codebase_structure() # Add params if needed
                prepended_content += f"\n{Style.BRIGHT}{Fore.MAGENTA}--- CODEBASE STRUCTURE ---{Style.RESET_ALL}\n"
                prepended_content += f"```\n{codebase_structure}\n```\n"
                prepended_content += f"{Style.BRIGHT}{Fore.MAGENTA}--- END CODEBASE STRUCTURE ---{Style.RESET_ALL}\n\n"
                mentioned_files.add("codebase")
                processed_mention_spans.append(match.span())
            continue # Move to next mention

        # Process file paths
        filepath = raw_target
        if filepath in mentioned_files:
            # Check if this specific mention span has already been processed
            # This handles cases like "@file1 @file1" where the second one should still be processed
            # if its span is different. However, simpler to just skip if filepath seen.
            continue # Skip already processed file paths

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
                processed_mention_spans.append(match.span())

            except UnicodeDecodeError:
                print(f"{Fore.YELLOW}  Warning: Cannot read mentioned binary file: {filepath}{Style.RESET_ALL}")
                prepended_content += f"\n{Fore.YELLOW}[CodAgent Note: Mentioned file '{filepath}' is likely binary and could not be read.]{Style.RESET_ALL}\n\n"
                mentioned_files.add(filepath)
                processed_mention_spans.append(match.span())
            except Exception as e:
                print(f"{Fore.RED}  Error reading mentioned file {filepath}: {e}{Style.RESET_ALL}")
                prepended_content += f"\n{Fore.RED}[CodAgent Note: Error reading mentioned file '{filepath}'.]{Style.RESET_ALL}\n\n"
                mentioned_files.add(filepath)
                processed_mention_spans.append(match.span())
        else:
            print(f"{Fore.YELLOW}  Warning: Mentioned file not found or is not a file: {filepath}{Style.RESET_ALL}")
            # Optionally inform the model the file wasn't found
            prepended_content += f"\n{Fore.YELLOW}[CodAgent Note: Mentioned file '{filepath}' not found.]{Style.RESET_ALL}\n\n"
            mentioned_files.add(filepath) # Add even if not found to avoid reprocessing

    # --- Clean the original input by removing processed mentions ---
    cleaned_user_input = ""
    last_end = 0
    # Sort spans to process them in order
    processed_mention_spans.sort(key=lambda x: x[0])
    for start, end in processed_mention_spans:
        cleaned_user_input += user_input[last_end:start]
        last_end = end
    cleaned_user_input += user_input[last_end:]
    # Remove potential leftover whitespace after cleaning
    cleaned_user_input = ' '.join(cleaned_user_input.split())


    if prepended_content:
         print(f"{Style.DIM}--- End Processing Mentions ---{Style.RESET_ALL}")
         # Return the prepended content followed by the CLEANED user input
         return prepended_content + cleaned_user_input, cleaned_user_input
    else:
        # Return original input if no valid mentions were processed
        return user_input, user_input

# Custom Completer for '@' mentions
class MentionCompleter(Completer):
    def __init__(self):
        # Combine path completer and specific '@codebase' word completer
        self.path_completer = PathCompleter(expanduser=True, get_paths=lambda: ['.'])
        self.codebase_completer = WordCompleter(['@codebase'], ignore_case=True)

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        text_before_cursor = document.text_before_cursor
        at_pos = text_before_cursor.rfind('@')
        space_pos = text_before_cursor.rfind(' ')

        if at_pos > -1 and (at_pos == len(text_before_cursor) - 1 or at_pos > space_pos):
            # --- Complete @codebase ---
            word_fragment_codebase = text_before_cursor[at_pos:]
            codebase_doc = Document(word_fragment_codebase, cursor_position=len(word_fragment_codebase))
            for completion in self.codebase_completer.get_completions(codebase_doc, complete_event):
                 start_pos_relative_to_cursor = completion.start_position - len(word_fragment_codebase)
                 # Ensure display is a string
                 display_text = str(completion.display) if completion.display else completion.text
                 yield Completion(completion.text, start_position=start_pos_relative_to_cursor, display=display_text, style=completion.style)


            # --- Complete file paths ---
            path_prefix = text_before_cursor[at_pos+1:]
            path_doc = Document(path_prefix, cursor_position=len(path_prefix))

            for completion in self.path_completer.get_completions(path_doc, complete_event):
                # Ensure display is a string before prepending '@'
                display_text = str(completion.display) if completion.display else completion.text
                yield Completion(
                    f"@{completion.text}",
                    start_position=completion.start_position,
                    display=f"@{display_text}", # Use the string version
                    style=completion.style
                )
        # else: # No need for explicit else pass


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
        'completion-menu.completion.current': 'bg:#00aaaa #000000',
        'completion-menu.completion': 'bg:#008888 #ffffff',
        # Potentially add styling for the separator line if desired
    })
    
    print(f"\n{Style.BRIGHT}{Fore.CYAN}=== CodAgent Interactive Session ==={Style.RESET_ALL}\n")
    print(f"{Fore.GREEN}Type '/help' for commands, '@' for file/codebase completion, 'exit' or Ctrl+C to quit.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Working Directory: {Fore.WHITE}{os.getcwd()}{Style.RESET_ALL}")
    print("-" * 40) # Separator
    
    # Keep track of conversation to maintain context
    conversation_history = []
    
    # --- Initialize the custom completer ---
    mention_completer = MentionCompleter()
    # Combine with history auto-suggest (will handle non-mention cases)
    # Note: We don't need a separate completer for history; AutoSuggestFromHistory works alongside custom completers.

    # Keep track of conversation to maintain context
    conversation_history = []
    # Store context that needs to be prepended *outside* the AI's direct turn
    pending_context_injection = ""

    while True:
        try:
            # --- Prepend any pending context (e.g., from user file selection) ---
            current_context_for_model = pending_context_injection
            pending_context_injection = "" # Clear after use

            # Add separator
            print(f"\n{Fore.BLUE}{H * (min(os.get_terminal_size().columns, 80))}{Style.RESET_ALL}")

            # --- Get User Input ---
            # (Only prompt user if there isn't context waiting for the AI)
            if not current_context_for_model:
                rprompt_text = f"[{Fore.CYAN}{os.path.basename(os.getcwd())}{Style.RESET_ALL}]"
                raw_user_input = prompt(
                    "CodAgent >>> ",
                    history=FileHistory(history_file),
                    auto_suggest=AutoSuggestFromHistory(),
                    completer=mention_completer,
                    style=style,
                    rprompt=ANSI(rprompt_text)
                )

                if raw_user_input.lower().strip() in ['exit', 'quit', 'q']:
                    print(f"{Fore.YELLOW}Exiting CodAgent session.{Style.RESET_ALL}")
                    break
                if not raw_user_input.strip(): continue

                # Process mentions
                user_input_for_model, user_input_for_history = process_mentions(raw_user_input)
                if user_input_for_history.strip():
                     conversation_history.append({"role": "user", "content": user_input_for_history})

                # --- Initialize command_processed ---
                command_processed = False
                # --- Process commands ---
                if raw_user_input.startswith('/'):
                     # We assume a command IS processed unless it's unknown or needs AI
                     command_processed = True
                     # ... (existing command handling logic) ...
                     # Ensure local commands like /list, /help `continue` the outer loop
                     is_local_only_command = raw_user_input.startswith(('/list', '/files', '/help'))
                     if command_processed and is_local_only_command:
                         # History pop handled within command logic
                         continue
                     # If command needs AI (/refresh) or is unknown, let it fall through

                # Add user input (after mentions processed) to the context for this turn
                current_context_for_model += user_input_for_model

            # --- Prepare Prompt for AI ---
            system_prompt = get_system_prompt()
            file_context_for_prompt = generate_file_context(file_history)

            # Construct the prompt: Sys Prompt + File Context + Pending/New Context + History
            model_prompt_parts = [system_prompt, file_context_for_prompt]

            # Add the context for this turn (user input or selected files)
            if current_context_for_model:
                 model_prompt_parts.append(current_context_for_model)

            # Add relevant conversation history
            # ... (existing history construction logic) ...
            # Make sure to only add USER/MODEL roles from history here, system notes added separately?
            history_to_include = min(10, len(conversation_history))
            prompt_history_formatted = []
            start_index = len(conversation_history) - history_to_include
            for i in range(start_index, len(conversation_history)):
                entry = conversation_history[i]
                role = entry['role'] # Keep original case for filtering maybe
                # Include User, Model, and specific System notes if desired, skip internal ones?
                # Let's include all for now for simplicity in history view
                prefix = f"{role.upper()}: "
                if role == 'system': prefix = f"{Fore.YELLOW}SYSTEM NOTE:{Style.RESET_ALL} "
                elif role == 'user': prefix = f"{Fore.GREEN}USER:{Style.RESET_ALL} "
                elif role == 'model': prefix = f"{Fore.CYAN}MODEL:{Style.RESET_ALL} "
                prompt_history_formatted.append(prefix + entry['content'])

            if prompt_history_formatted:
                 # Decide whether to place history before or after current context injection
                 # Let's place it *before* the current turn's specific context
                 model_prompt_parts.insert(2, f"{Style.BRIGHT}{Fore.MAGENTA}--- CONVERSATION HISTORY (Last {history_to_include}) ---{Style.RESET_ALL}\n" + "\n\n".join(prompt_history_formatted))
                 model_prompt_parts.insert(3, f"\n{Style.BRIGHT}{Fore.MAGENTA}--- END HISTORY ---{Style.RESET_ALL}")


            final_prompt_string = "\n\n".join(model_prompt_parts)

            # --- Model Interaction Loop (Multi-Turn) ---
            all_responses_this_turn = []
            is_end_of_turn = False
            ask_for_files_detected = False
            files_to_ask_user_for = []

            print(f"\n{Style.DIM}--- CodAgent Thinking ---{Style.RESET_ALL}")
            current_prompt_for_model_segment = final_prompt_string # Start with the full prompt

            while not is_end_of_turn and not ask_for_files_detected:
                print(f"\n{Style.BRIGHT}{Fore.GREEN}>>> AI Response Segment >>>{Style.RESET_ALL}")
                current_segment_text = ""
                stream_error_occurred = False
                try:
                    # ... (existing streaming generation logic) ...
                    response_stream = model.generate_content(current_prompt_for_model_segment, stream=True)
                    # ... (loop through chunks, print, append to current_segment_text) ...
                    for chunk in response_stream:
                        try:
                            chunk_text = chunk.text
                            print(chunk_text, end='', flush=True)
                            current_segment_text += chunk_text
                        except ValueError:
                            pass # Ignore non-text chunks
                        except Exception as e_text_access:
                            print(f"\n{Fore.RED}Error processing stream chunk text: {e_text_access}{Style.RESET_ALL}", flush=True)
                    print() # Newline after segment stream

                except Exception as model_error:
                     # ... (existing error handling) ...
                     print(f"\n{Back.RED}{Fore.WHITE} ERROR during model generation request: {model_error} {Style.RESET_ALL}")
                     current_segment_text = "[CodAgent Error: Generation failed]"
                     is_end_of_turn = True # Assume end on error
                     stream_error_occurred = True

                # --- Check for Tags AFTER getting full segment ---
                segment_for_parsing = current_segment_text # Use the complete segment text

                # 1. Check for ====== ASK_FOR_FILES
                if not stream_error_occurred:
                    extracted_files, segment_without_ask_tag = parse_ask_for_files(segment_for_parsing)
                    if extracted_files:
                        print(f"\n{Fore.YELLOW}[CodAgent needs files...]{Style.RESET_ALL}")
                        ask_for_files_detected = True
                        files_to_ask_user_for = extracted_files
                        # Use the response text *without* the ASK tag for history/display
                        current_segment_text = segment_without_ask_tag
                        # This stops the multi-turn loop for this AI response cycle
                        # We will handle user interaction outside this inner loop

                # 2. Check for [END] tag (only if ASK_FOR_FILES wasn't detected)
                if not stream_error_occurred and not ask_for_files_detected:
                     parsed_segment, is_end_of_turn_from_tag = parse_end_response(segment_for_parsing)
                     if is_end_of_turn_from_tag:
                          current_segment_text = parsed_segment # Use text without tag
                          is_end_of_turn = True

                # Store the (potentially modified) segment text
                all_responses_this_turn.append(current_segment_text)

                # Prepare for next segment if needed (only if no END and no ASK)
                if not is_end_of_turn and not ask_for_files_detected:
                    print(f"\n{Style.DIM}--- CodAgent Continuing ---{Style.RESET_ALL}")
                    current_prompt_for_model_segment += f"\n\n{Fore.CYAN}MODEL:{Style.RESET_ALL} {current_segment_text}\n\n{Fore.GREEN}CONTINUE:{Style.RESET_ALL}"
                elif is_end_of_turn:
                     print(f"\n{Style.DIM}--- CodAgent Finished Turn ---{Style.RESET_ALL}")
                # No explicit message needed if ask_for_files_detected, handled below

            # --- End of AI Turn / Segment Loop ---

            # Combine all raw responses from this turn for history/parsing file ops
            full_raw_response = "\n\n".join(all_responses_this_turn)

            # Add final AI response to history (even if asking for files)
            if full_raw_response.strip() and not stream_error_occurred:
                 conversation_history.append({"role": "model", "content": full_raw_response})
            elif stream_error_occurred:
                 conversation_history.append({"role": "system", "content": "Model generation failed."})

            # --- Handle ASK_FOR_FILES Interaction ---
            if ask_for_files_detected:
                print(f"\n{Fore.CYAN}CodAgent is asking for the following files:{Style.RESET_ALL}")
                for idx, filepath in enumerate(files_to_ask_user_for):
                    print(f"  {Fore.YELLOW}[{idx+1}]{Style.RESET_ALL} {filepath}")

                selected_indices_str = input(f"{Fore.CYAN}Enter numbers of files to provide (comma-separated, e.g., 1,3), or press Enter to skip: {Style.RESET_ALL}").strip()
                selected_files_content = ""
                selected_filenames_for_note = []

                if selected_indices_str:
                    try:
                        selected_indices = [int(i.strip()) - 1 for i in selected_indices_str.split(',')]
                        temp_selected_context = "" # Build context incrementally

                        for idx in selected_indices:
                            if 0 <= idx < len(files_to_ask_user_for):
                                filepath = files_to_ask_user_for[idx]
                                full_path = os.path.abspath(filepath)
                                if os.path.exists(full_path) and os.path.isfile(full_path):
                                    try:
                                        with open(full_path, 'r', encoding='utf-8') as f:
                                            file_content = f.read()
                                        print(f"{Fore.GREEN}  ✓ Reading: {filepath}{Style.RESET_ALL}")
                                        temp_selected_context += f"\n{Style.BRIGHT}{Fore.MAGENTA}--- SELECTED FILE: {filepath} ---{Style.RESET_ALL}\n"
                                        temp_selected_context += f"```\n{file_content}\n```\n"
                                        temp_selected_context += f"{Style.BRIGHT}{Fore.MAGENTA}--- END SELECTED FILE: {filepath} ---{Style.RESET_ALL}\n\n"
                                        selected_filenames_for_note.append(filepath)
                                    except Exception as e:
                                        print(f"{Fore.RED}  ✗ Error reading {filepath}: {e}{Style.RESET_ALL}")
                                else:
                                    print(f"{Fore.RED}  ✗ File not found or not a file: {filepath}{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.RED}  ✗ Invalid number skipped: {idx+1}{Style.RESET_ALL}")

                        if temp_selected_context:
                            # Add a header indicating this context follows the AI's request
                            selected_files_content = f"{Style.BRIGHT}{Fore.GREEN}--- Providing Content for User-Selected Files ---{Style.RESET_ALL}\n" + temp_selected_context
                        else:
                             print(f"{Fore.YELLOW}No valid files selected or read.{Style.RESET_ALL}")

                    except ValueError:
                        print(f"{Fore.RED}Invalid input format. Please enter numbers separated by commas.{Style.RESET_ALL}")

                if selected_files_content:
                    # Store this context to be prepended at the start of the *next* loop iteration
                    pending_context_injection = selected_files_content
                    conversation_history.append({"role": "system", "content": f"User selected and provided content for: {', '.join(selected_filenames_for_note)}"})
                    # Continue the loop immediately to send this context back to the AI
                    continue
                else:
                    # User skipped or selection failed
                    print(f"{Fore.YELLOW}Skipping file provision. Asking AI to proceed without them.{Style.RESET_ALL}")
                    pending_context_injection = f"{Fore.YELLOW}[SYSTEM NOTE: User did not provide the requested files. Proceed based on existing context or ask again if necessary.]{Style.RESET_ALL}"
                    conversation_history.append({"role": "system", "content": "User skipped providing requested files."})
                    # Continue the loop immediately
                    continue

            # --- If NOT asking for files, proceed with normal post-response processing ---
            if not ask_for_files_detected:

                # ***** TERMINAL COMMAND HANDLING *****
                terminal_commands = parse_terminal_commands(full_raw_response)
                executed_command_results = [] # Store results for context

                if terminal_commands:
                    print("\n" + "="*5 + " Terminal Commands Proposed " + "="*5) # Header
                    print_boxed(
                        "Terminal Commands Preview",
                        "\n".join([f"- {cmd}" for cmd in terminal_commands]),
                        color=Fore.YELLOW
                    )
                    print("-" * 30)
                    confirm_terminal = input(f"{Style.BRIGHT}{Fore.CYAN}Execute these terminal commands? (y/n): {Style.RESET_ALL}").lower().strip()

                    if confirm_terminal.startswith('y'):
                        print(f"{Fore.YELLOW}Executing commands...{Style.RESET_ALL}")
                        for command in terminal_commands:
                            result = execute_terminal_command(command)
                            executed_command_results.append({"command": command, "result": result})
                        # Add results to conversation history
                        cmd_summary_lines = ["Terminal Command Execution Results:"]
                        for res in executed_command_results:
                            # **** MODIFIED: History message format ****
                            status = f"{Fore.GREEN}✓ SUCCESS{Style.RESET_ALL}" if res['result']['returncode'] == 0 else f"{Fore.RED}✗ FAILED (Code: {res['result']['returncode']}){Style.RESET_ALL}"
                            # Prepend status to the command itself in the summary line
                            cmd_summary_lines.append(f"`{res['command']}`: {status}")
                            if res['result']['stdout']:
                                # Indent output/errors under the command status line
                                cmd_summary_lines.append(f"  Output: {res['result']['stdout'][:100]}{'...' if len(res['result']['stdout']) > 100 else ''}")
                            if res['result']['stderr']:
                                cmd_summary_lines.append(f"  {Fore.RED}Errors:{Style.RESET_ALL} {res['result']['stderr'][:100]}{'...' if len(res['result']['stderr']) > 100 else ''}")
                        conversation_history.append({"role": "system", "content": "\n".join(cmd_summary_lines)})
                        print(f"{Fore.GREEN}Finished executing commands.{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.YELLOW}✗ Terminal commands skipped by user.{Style.RESET_ALL}")
                        conversation_history.append({"role": "system", "content": "User skipped proposed terminal commands."})
                # ***** END TERMINAL COMMAND HANDLING *****


                # --- File Operation Handling (remains the same) ---
                initial_file_operations = parse_file_operations(full_raw_response) # Parse from the full response text
                applied_file_ops = False
                aggregated_apply_result = {"successful": [], "failed": []}

                if initial_file_operations:
                    print("\n" + "="*5 + " File Operations Proposed " + "="*5) # Header
                    if preview_changes(initial_file_operations):
                        initial_apply_result = apply_changes(initial_file_operations)
                        applied_file_ops = True

                        aggregated_apply_result["successful"].extend(initial_apply_result.get("successful", []))
                        aggregated_apply_result["failed"].extend(initial_apply_result.get("failed", []))

                        # Auto-Retry Logic
                        initial_failed_replaces = [op for op in initial_apply_result.get("failed", []) if op["type"] == "replace"]
                        if initial_failed_replaces:
                            aggregated_apply_result["failed"] = [op for op in aggregated_apply_result["failed"] if op["type"] != "replace" or op not in initial_failed_replaces]
                            retry_results = retry_failed_replacements(initial_failed_replaces, model, file_history, conversation_history)
                            aggregated_apply_result["successful"].extend(retry_results.get("newly_successful", []))
                            aggregated_apply_result["failed"].extend(retry_results.get("final_failed", []))

                        # Post-Retry Summary
                        op_summary_lines = [f"Final File Operations Status:"]
                        # ... (append successful/failed lines) ...
                        if aggregated_apply_result['successful']: op_summary_lines.append(f"  {Fore.GREEN}Successful ({len(aggregated_apply_result['successful'])}):{Style.RESET_ALL} {', '.join(list({op['filename'] for op in aggregated_apply_result['successful']}))}")
                        if aggregated_apply_result['failed']: op_summary_lines.append(f"  {Fore.RED}Failed ({len(aggregated_apply_result['failed'])}):{Style.RESET_ALL} {', '.join(list({op['filename'] for op in aggregated_apply_result['failed']}))}")
                        conversation_history.append({"role": "system", "content": "\n".join(op_summary_lines)})

                        if not aggregated_apply_result['failed']: print(f"\n{Fore.GREEN}✓ All file operations completed successfully (including retries)!{Style.RESET_ALL}")
                        else: print(f"\n{Fore.RED}✗ Some file operations failed.{Style.RESET_ALL}")

                    else:
                         print(f"{Fore.YELLOW}✗ File operations skipped by user.{Style.RESET_ALL}")
                         conversation_history.append({"role": "system", "content": "User skipped proposed file operations."})

                # Update file history based on final successful ops
                if applied_file_ops:
                     # ... (existing file history update logic) ...
                     for op in aggregated_apply_result.get("successful", []):
                          # ... (update created/modified lists) ...
                          norm_filename = os.path.normpath(op["filename"])
                          # ... (rest of update logic)
                          if op["type"] == "create":
                              if norm_filename not in file_history["created"]:
                                  file_history["created"].append(norm_filename)
                              if norm_filename not in file_history["current_workspace"]:
                                  file_history["current_workspace"].append(norm_filename)
                              if norm_filename in file_history["modified"]:
                                  file_history["modified"].remove(norm_filename)
                          elif op["type"] == "replace":
                              if norm_filename not in file_history["modified"] and norm_filename not in file_history["created"]:
                                  file_history["modified"].append(norm_filename)
                              if norm_filename not in file_history["current_workspace"]:
                                   file_history["current_workspace"].append(norm_filename)


            # Separator moved down, only print if not asking for files? Or always? Let's keep it always.
            # print("\n" + H * (min(os.get_terminal_size().columns, 80))) # End of turn separator moved from here

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupt received. Exiting...{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n{Back.RED}{Fore.WHITE} UNEXPECTED ERROR: {e} {Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            # Add error to history for context
            conversation_history.append({"role": "system", "content": f"An unexpected error occurred: {e}\n{traceback.format_exc()}"})

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
    parser.add_argument("--model", default="gemini-2.5-pro-exp-03-25", help="Google Gemini model to use")

    args = parser.parse_args()

    print(f"{Fore.CYAN}Initializing CodAgent with model: {Fore.GREEN}{args.model}{Style.RESET_ALL}")
    model = initialize_genai(args.model)
    chat_with_model(model)

if __name__ == "__main__":
    main() 