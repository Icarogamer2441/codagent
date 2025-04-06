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
from openai import OpenAI # Added for OpenRouter
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

# --- Added: Check OpenRouter API Key ---
def check_openrouter_api_key():
    """Check if OpenRouter API key is set in environment variables."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(f"{Fore.YELLOW}OpenRouter API key (OPENROUTER_API_KEY) not found in environment variables.{Style.RESET_ALL}")
        print("You can get one from https://openrouter.ai")
        if os.name == "nt":  # Windows
            print("\nYou can set it temporarily with:")
            print(f"{Fore.CYAN}set OPENROUTER_API_KEY=your_api_key{Style.RESET_ALL}")
            print("\nOr permanently via System Properties > Advanced > Environment Variables.")
        else:  # Linux/Unix
            print("\nYou can set it temporarily with:")
            print(f"{Fore.CYAN}export OPENROUTER_API_KEY=your_api_key{Style.RESET_ALL}")
            print("\nOr add it to your ~/.bashrc or ~/.zshrc file.")

        print(Style.RESET_ALL, end='') # Explicitly reset colors before input
        api_key = input("\nEnter your OpenRouter API key (leave blank to skip OpenRouter): ").strip()
        if not api_key:
            print(f"{Fore.YELLOW}No OpenRouter API key provided.{Style.RESET_ALL}")
            return None # Allow skipping
        os.environ["OPENROUTER_API_KEY"] = api_key
        print(f"{Fore.GREEN}OpenRouter API key set for this session.{Style.RESET_ALL}")
    return api_key
# --- End Added ---

def initialize_model(args):
    """Initialize the AI model based on command-line arguments."""
    if args.omodel:
        # --- Initialize OpenRouter ---
        print(f"{Fore.CYAN}Attempting to initialize OpenRouter model: {Fore.GREEN}{args.omodel}{Style.RESET_ALL}")
        api_key = check_openrouter_api_key()
        if not api_key:
             print(f"{Fore.RED}OpenRouter API key is required to use --omodel. Exiting.{Style.RESET_ALL}")
             sys.exit(1)
        try:
            # Point OpenAI client to OpenRouter endpoint
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            # You might want to add a check here to see if the model exists
            # client.models.retrieve(args.omodel) # This would verify the model ID
            print(f"{Fore.GREEN}Successfully configured OpenRouter client.{Style.RESET_ALL}")
            return client, "openrouter", args.omodel # Return client, provider name, model name
        except Exception as e: # Ensure this except aligns with the try
            print(f"{Fore.RED}Error initializing OpenRouter model '{args.omodel}': {e}{Style.RESET_ALL}")
            sys.exit(1)

    else: # Ensure this else aligns with the initial if args.omodel
        # --- Initialize Google Gemini ---
        print(f"{Fore.CYAN}Initializing Google Gemini model: {Fore.GREEN}{args.model}{Style.RESET_ALL}")
        api_key = check_api_key() # Uses existing GOOGLE_API_KEY check
        if not api_key:
             print(f"{Fore.RED}Google API key is required if not using --omodel. Exiting.{Style.RESET_ALL}")
             sys.exit(1)
        genai.configure(api_key=api_key)
        try:
            model = genai.GenerativeModel(args.model)
            # Perform a simple test call? Optional.
            # model.generate_content("test", generation_config=genai.types.GenerationConfig(max_output_tokens=1))
            print(f"{Fore.GREEN}Successfully initialized Google Gemini model.{Style.RESET_ALL}")
            return model, "google", args.model # Return model, provider name, model name
        except Exception as e:
            print(f"{Fore.RED}Error initializing Google Gemini model '{args.model}': {e}{Style.RESET_ALL}")
            sys.exit(1) # Corrected indentation

def get_system_prompt():
    """Generate a system prompt detailing capabilities and the auto-fix loop."""
    current_dir = os.getcwd()
    # --- System Prompt with Auto-Fix Instructions --- 
    system_prompt = f"""**You are CodAgent:** An AI assistant performing file operations and terminal commands in `{current_dir}`.

**IMPORTANT: FILE TRACKING**
- When a file appears in `--- FILE CONTEXT ---` section, it has been loaded and you can access its content
- DO NOT ask for files that are already in the context
- Check the `--- FILE CONTEXT ---` section before asking for any file
- If a file is mentioned in the context, ASSUME you have full access to it

**IMPORTANT: USE THE ASK_TO_USER TAG**
When you need ANY information from the user, you MUST use the ASK_TO_USER tag. NEVER ask questions in regular text.

**Commands:**
- `====== CREATE path/file.ext`\n...content...\n`====== CEND`
- `======= REPLACE path/file.ext`\n`old_code_block`\n`======= TO`\n`new_code_block`\n`======= REND` (Block-based replacement that maintains exact indentation)
- `====== TERMINAL`\n...command...\n`====== TEND`
- `====== ASK_FOR_FILES`\npath/file1\npath/file2\n`====== AEND`
- `====== ASK_TO_USER format:type`\n...question...\n`====== QEND`

**Block-Based REPLACE Example:**
```
======= REPLACE path/file.ext
def hello_world():
    print("Hello, world!")
    return None
======= TO
def hello_world():
    print("Hello, beautiful world!")
    return True
======= REND
```

**WRONG Block-Based REPLACE Example (DO NOT use line numbers):**
```
======= REPLACE path/file.ext
10 | def hello_world():
11 |     print("Hello, world!")
12 |     return None
======= TO
10 | def hello_world():
11 |     print("Hello, beautiful world!")
12 |     return True
======= REND
```

**NOTE on BLOCK-BASED REPLACE:** 
- The old code block must match EXACTLY with code in the file (including whitespace and indentation)
- If there's no exact match, the system will attempt to find the closest match and show what doesn't match
- This is the preferred method for replacing multi-line code segments where exact indentation is important
- DO NOT include line numbers in the code blocks (like "10 |" or "42 |") - this will always cause the match to fail
- Code blocks should ONLY contain the exact code as it appears in the file without any line prefixes

**TIPS for using BLOCK-BASED REPLACE:**
- Use seven equal signs (=======) for block-based REPLACE, not six equal signs
- Ensure the old code block includes proper indentation, empty lines, and comments exactly as in the file
- If a replacement fails due to mismatch, check for whitespace differences and try again
- For replacing entire functions or code blocks, copy the exact text first (using ASK_FOR_FILES if needed)
- Block-based replacement is safer than line-based replacement for complex code changes
- Use "======= TO" to separate the old and new code blocks
- NEVER include line numbers (like "10 |" or "42 |") in the code blocks - they will cause the match to fail
- ONLY include the exact code content without any line number prefixes

**ASK_TO_USER formats:**
- `format:normal` - Ask a free-form question
- `format:options` - Present numbered options (1, 2, 3, ...)
- `format:yesno` - Ask a yes/no question

**ASK_TO_USER Examples:**
```
# Example of normal question
====== ASK_TO_USER format:normal
What would you like to name this file?
====== QEND

# Example of options (each option on its own line)
====== ASK_TO_USER format:options
CLI Task Manager
Weather Info Fetcher
Simple Chatbot
====== QEND

# Example of yes/no question
====== ASK_TO_USER format:yesno
Would you like to add error handling to this function?
====== QEND
```

**CRITICAL: ALWAYS USE ASK_TO_USER**
- If you need ANY clarification from the user, use the ASK_TO_USER tag
- If user instructions are unclear, use ASK_TO_USER format:normal to ask what they want
- If user needs to choose between options, use ASK_TO_USER format:options
- If user needs to make a yes/no decision, use ASK_TO_USER format:yesno
- NEVER ask questions in regular text responses - ALWAYS use the ASK_TO_USER tag

**CRITICAL: `[END]` Tag Usage**
- Use `[END]` ONLY after the *final* action that *completely* fulfills the user's entire request for the current turn.
- DO NOT use `[END]` if more steps are needed, if asking for files, or if initiating an error fix (see below).
- DO NOT use `[END]` in the same segment where you use `====== ASK_TO_USER` to ask a question. Wait for the user's response first.

**Workflow & Auto-Fix Loop:**
1.  **User Request:** Understand the user's goal.
2.  **Unclear Instructions?** If the user's request is unclear, use `====== ASK_TO_USER format:normal` to ask for clarification.
3.  **Plan & Execute:** Generate commands (`CREATE`, `REPLACE`, `TERMINAL`) to fulfill the request. This might take multiple responses if the plan has multiple steps.
4.  **Need User Input?** Use `====== ASK_TO_USER` with the appropriate format type whenever you need the user to make a decision.
5.  **Context Update:** After your file changes (`REPLACE`, `CREATE`) are applied, the system will automatically provide you with the updated content of the modified files in the `--- FILE CONTEXT ---` section.
6.  **Terminal Output:** When you execute terminal commands via `====== TERMINAL`, you'll receive detailed execution results including stdout, stderr, and exit codes in the conversation history. Review these to understand command results before proceeding.
7.  **User Interaction:** When you need specific information from the user, use the `====== ASK_TO_USER` tag with the appropriate format to ask questions directly.
8.  **Error Check (Self-Correction):**
    *   **IMPERATIVE:** Carefully review the updated `--- FILE CONTEXT ---`, specifically the code you *just* modified or created.
    *   **FILE CONTENT:** Read the updated file content carefully using the `====== ASK_FOR_FILES` tag.
    *   **ERRORS:** Look for syntax errors, logical flaws, incorrect variable names, missing imports, or inconsistencies based on the surrounding code and the user's original request.
    *   **If you find an error** caused by your *previous* action: 
        *   **DO NOT use `[END]`**. 
        *   Explain the error you found briefly.
        *   Provide a `======= REPLACE` command targeting the code that needs to be fixed.
        *   The loop will repeat (Steps 3-4) after the fix is applied.
    *   **If you find NO errors** in your *previous* action:
        *   Proceed to the next step in your plan (if any).
        *   If all steps are done and the user's request is fully met, explain briefly and use the `[END]` tag.
9.  **Completion:** The process ends when you confirm no errors in your last action AND the user's overall goal is achieved, signaled by you using `[END]`.

**Other Key Rules:**
*   **Context is King:** Base ALL actions on the most recent `--- FILE CONTEXT ---` and `@mentions`.
*   **Error-Driven Fixes:** If the `SYSTEM CHECK` provides `stderr` output from a syntax check, **prioritize fixing the specific errors reported in `stderr`**. Use the line numbers and error messages provided.
*   **Indentation:** Maintain EXACT indentation for all code, in all languages.
*   **Clarity:** Briefly explain plans, results, and discovered errors/fixes.
*   **Safety:** Be cautious with `TERMINAL` commands.

Use the provided **CONTEXT SECTIONS** below.
"""
    return system_prompt

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

def parse_ask_to_user(response_text):
    """Parse the response text to extract user questions from ====== ASK_TO_USER tag."""
    # Use re.MULTILINE and re.DOTALL to match content between the tags
    ask_pattern = r"^====== ASK_TO_USER format:(\w+)\s*\n(.*?)\n====== QEND\s*$"
    match = re.search(ask_pattern, response_text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    if match:
        question_format = match.group(1).strip().lower()
        content = match.group(2).strip()
        
        # Validate format type
        if question_format not in ['normal', 'options', 'yesno']:
            question_format = 'normal'  # Default to normal if invalid format
        
        # For options format, parse the content as a list of options
        if question_format == 'options':
            # Extract options as separate lines, ignoring empty lines and comments
            options = [line.strip() for line in content.splitlines() 
                      if line.strip() and not line.strip().startswith('#')]
            # Return question_data with the options list
            question_data = {
                "format": question_format, 
                "options": options
            }
        else:
            # For normal and yesno, just return the content as question
            question_data = {
                "format": question_format,
                "question": content
            }
            
        response_without_tag = response_text[:match.start()] + response_text[match.end():]
        return question_data, response_without_tag.strip()
    
    return None, response_text  # Return None if tag not found or empty

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
    ai_response_log = [] # New log specifically formatted for AI consumption

    # **** ADDED: Explicit command line inside the box content ****
    exec_log.append(f"{Style.BRIGHT}Command:{Style.RESET_ALL} {command}")
    exec_log.append(H * visible_len(f"Command: {command}")) # Add a separator line
    # **** END ADDED ****

    # **** ADDED: AI-specific log format ****
    ai_response_log.append(f"Command: {command}")
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
            
            # **** ADDED: AI-specific log format for stdout ****
            ai_response_log.append(f"--- STDOUT ---")
            ai_response_log.extend(output.splitlines())
            ai_response_log.append("-------------")
            # **** END ADDED ****

        if errors:
            exec_log.append(f"{Fore.RED}--- Errors ---{Style.RESET_ALL}")
            exec_log.extend(errors.splitlines())
            exec_log.append(f"{Fore.RED}------------{Style.RESET_ALL}")
            
            # **** ADDED: AI-specific log format for stderr ****
            ai_response_log.append(f"--- STDERR ---")
            ai_response_log.extend(errors.splitlines())
            ai_response_log.append("-------------")
            # **** END ADDED ****

        if return_code == 0:
             exec_log.append(f"{Fore.GREEN}✓ Command finished successfully (Exit Code: 0){Style.RESET_ALL}")
             # **** ADDED: AI-specific log format for exit code ****
             ai_response_log.append(f"Exit Code: 0 (Success)")
             # **** END ADDED ****
        else:
             exec_log.append(f"{Fore.RED}✗ Command failed (Exit Code: {return_code}){Style.RESET_ALL}")
             # **** ADDED: AI-specific log format for exit code ****
             ai_response_log.append(f"Exit Code: {return_code} (Error)")
             # **** END ADDED ****

    except Exception as e:
        errors = str(e)
        # Add error *after* the command line inside the box
        exec_log.append(f"{Fore.RED}✗ Error executing command:{Style.RESET_ALL} {e}")
        return_code = -1 # Indicate failure
        
        # **** ADDED: AI-specific log format for exception ****
        ai_response_log.append(f"Error: Failed to execute command: {e}")
        ai_response_log.append(f"Exit Code: {return_code} (Error)")
        # **** END ADDED ****

    # Print execution log in a box
    box_color = Fore.RED if return_code != 0 else Fore.GREEN
    print_boxed(f"Execution Result", "\n".join(exec_log), color=box_color)
    print("-" * 30) # Separator after box

    # Return both the standard result and the AI-formatted log
    return {
        "stdout": output, 
        "stderr": errors, 
        "returncode": return_code,
        "ai_log": "\n".join(ai_response_log)  # Add the AI-specific formatted log
    }

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
    cleaned_response, _ = parse_end_response(response_text)
    cleaned_response = strip_code_fences(cleaned_response) # Pre-strip outer fences
    
    file_operations = []
    
    # --- CREATE Operation ---
    create_pattern = r"^====== CREATE\s+([^\n]+)\n(.*?)\n====== CEND\s*$"
    for match in re.finditer(create_pattern, cleaned_response, re.DOTALL | re.MULTILINE | re.IGNORECASE):
        filename = match.group(1).strip()
        raw_content = match.group(2)
        content = strip_code_fences(raw_content.strip())
        if content:
            file_operations.append({
                "type": "create",
                "filename": filename,
                "content": content
            })
        else:
             print(f"{Fore.YELLOW}Warning: Skipping CREATE operation for '{filename}' because content was empty after stripping.{Style.RESET_ALL}")
    
    # --- Simplified Line-Based REPLACE Operation (old format - keeping for backward compatibility) ---
    old_replace_pattern = r"^====== REPLACE\s+([^\n]+)\n(.*?)\n====== REND\s*$"
    # Modified to handle +/- line prefixes and ~ for after-line insertion
    line_content_pattern = re.compile(r"^\s*([\+\-\~]\d+)\s*\|(.*?)$", re.DOTALL)
    
    for match in re.finditer(old_replace_pattern, cleaned_response, re.DOTALL | re.MULTILINE | re.IGNORECASE):
        filename = match.group(1).strip()
        raw_lines_block = match.group(2)  # Don't strip - preserve whitespace

        parsed_replacements = [] # List of {"line": N, "new_content": "...", "delete": bool, "insert": bool}

        # --- Parse the block for "N | new_content" lines ---
        block_lines = raw_lines_block.splitlines()
        for line in block_lines:
            line_match = line_content_pattern.match(line)
            if line_match:
                line_num_str = line_match.group(1)
                is_delete = line_num_str.startswith('-')
                is_exact = line_num_str.startswith('+')  # Changed: + now means exact line
                is_insert = line_num_str.startswith('~')  # Changed: ~ now means insert after
                
                # Get the actual line number
                line_num = int(line_num_str[1:])  # Always remove the first character (+, - or ~)
                
                # Get content after pipe and TRIM EXACTLY ONE SPACE if it exists
                raw_content = line_match.group(2)
                new_content = raw_content[1:] if raw_content.startswith(' ') else raw_content
                
                if is_delete:
                    parsed_replacements.append({
                        "line": line_num,
                        "new_content": "",  # Empty content for deletion
                        "delete": True,
                        "insert": False,
                        "exact": False
                    })
                elif is_insert:  # ~ for insert after
                    parsed_replacements.append({
                        "line": line_num,
                        "new_content": new_content,  # Insert after line
                        "delete": False,
                        "insert": True,
                        "exact": False
                    })
                elif is_exact:  # + for exact line
                    parsed_replacements.append({
                        "line": line_num,
                        "new_content": new_content,  # Insert at exact line number
                        "delete": False,
                        "insert": False,
                        "exact": True
                    })
            elif line.strip(): # Non-empty line that doesn't match expected format
                 print(f"{Fore.YELLOW}Warning: Skipping unrecognized line in ====== REPLACE block for '{filename}': {line}{Style.RESET_ALL}")

        if parsed_replacements: # Only add if we successfully parsed some line changes
            file_operations.append({
                "type": "replace_lines",
                "filename": filename,
                "replacements": parsed_replacements # Store the parsed line replacements
            })
        elif filename: # If tag was present but no valid lines found
             print(f"{Fore.YELLOW}Warning: REPLACE for '{filename}' contained no valid '+N | content', '-N |', or '~N |' lines.{Style.RESET_ALL}")
    
    # --- New Block-Based REPLACE Operation ---
    new_replace_pattern = r"^======= REPLACE\s+([^\n]+)\n(.*?)\n======= TO\n(.*?)\n======= REND\s*$"
    
    for match in re.finditer(new_replace_pattern, cleaned_response, re.DOTALL | re.MULTILINE | re.IGNORECASE):
        filename = match.group(1).strip()
        old_code_block = match.group(2)
        new_code_block = match.group(3)
        
        # Skip empty replacements
        if not old_code_block.strip() or not new_code_block.strip():
            print(f"{Fore.YELLOW}Warning: Skipping block REPLACE for '{filename}' because old or new code block was empty.{Style.RESET_ALL}")
            continue
            
        if not os.path.exists(filename):
            print(f"{Fore.YELLOW}Warning: File '{filename}' does not exist for block REPLACE operation.{Style.RESET_ALL}")
            continue
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
            # We need to verify that the old code block exists exactly in the file
            old_code_lines = old_code_block.splitlines()
            file_lines = file_content.splitlines()
            
            # Create a special operation for block-based replacement
            file_operations.append({
                "type": "replace_block",
                "filename": filename,
                "old_code": old_code_block,
                "new_code": new_code_block,
                "verified": False  # Will be set to True during apply phase if matched
            })
                
        except Exception as e:
            print(f"{Fore.RED}Error reading file '{filename}' for block REPLACE: {str(e)}{Style.RESET_ALL}")
    
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
                 preview_content.append(f"{Fore.GREEN}  {line}{Style.RESET_ALL}")
            if len(content_lines) > 5:
                 preview_content.append(f"{Fore.GREEN}  ...{Style.RESET_ALL}")
            preview_content.append("") # Add empty line for spacing
        
        # Preview for replace_lines - Properly showing indentation
        elif op["type"] == "replace_lines":
            preview_content.append(f"{Style.BRIGHT}{Fore.YELLOW}REPLACE in File (Lines):{Style.RESET_ALL} {Fore.WHITE}{op['filename']}{Style.RESET_ALL}")
            preview_content.append(f"{Fore.YELLOW}Proposed Line Changes (indentation preserved):{Style.RESET_ALL}")
            
            # Sort replacements by line number for preview
            for change in sorted(op.get('replacements', []), key=lambda x: x['line']):
                line_num = change['line']
                if change.get('delete', False):
                    # Show deletion
                    preview_content.append(f"{Fore.RED}- -{line_num} | (Deleted line){Style.RESET_ALL}")
                elif change.get('insert', False):
                    # Show insertion after line
                    preview_content.append(f"{Fore.BLUE}+ ~{line_num} | {change['new_content']}{Style.RESET_ALL}")
                elif change.get('exact', False):
                    # Show exact line insertion
                    preview_content.append(f"{Fore.MAGENTA}= +{line_num} | {change['new_content']}{Style.RESET_ALL}")
                else:
                    # This case should never happen with the updated regex, but keep for safety
                    preview_content.append(f"{Fore.YELLOW}? {line_num} | {change['new_content']} (Unsupported - use +/- prefix){Style.RESET_ALL}")
            
            preview_content.append("") # Add empty line for spacing
            
        # Preview for block-based replacement
        elif op["type"] == "replace_block":
            preview_content.append(f"{Style.BRIGHT}{Fore.CYAN}REPLACE CODE BLOCK in File:{Style.RESET_ALL} {Fore.WHITE}{op['filename']}{Style.RESET_ALL}")
            
            # Show old code that will be replaced
            preview_content.append(f"{Fore.YELLOW}Code to be replaced:{Style.RESET_ALL}")
            old_code_lines = op["old_code"].splitlines()
            
            # Show first few lines of old code
            max_preview_lines = min(5, len(old_code_lines))
            for line in old_code_lines[:max_preview_lines]:
                preview_content.append(f"{Fore.RED}- {line}{Style.RESET_ALL}")
            if len(old_code_lines) > max_preview_lines:
                preview_content.append(f"{Fore.RED}- ...{Style.RESET_ALL}")
            
            preview_content.append(f"{Fore.YELLOW}Will be replaced with:{Style.RESET_ALL}")
            
            # Show new code that will replace the old
            new_code_lines = op["new_code"].splitlines()
            max_preview_lines = min(5, len(new_code_lines))
            for line in new_code_lines[:max_preview_lines]:
                preview_content.append(f"{Fore.GREEN}+ {line}{Style.RESET_ALL}")
            if len(new_code_lines) > max_preview_lines:
                preview_content.append(f"{Fore.GREEN}+ ...{Style.RESET_ALL}")
                
            # Show line counts for reference
            preview_content.append(f"{Fore.CYAN}({len(old_code_lines)} lines replaced with {len(new_code_lines)} lines){Style.RESET_ALL}")
            preview_content.append("") # Add empty line for spacing

    if not operations_present:
         preview_content.append(f"{Fore.YELLOW}No file operations were parsed from the response.{Style.RESET_ALL}")
         print_boxed("File Operations Preview", "\n".join(preview_content), color=Fore.YELLOW)
         return True

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
        filename = op['filename']
        
        # Create file operation - existing logic
        if op["type"] == "create":
            # ... existing code ...
            try:
                # Create parent directories if needed
                os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
                # Write the file
                with open(filename, "w", newline='\n') as f:
                    f.write(op["content"])
                apply_log.append(f"{Fore.GREEN}✓ SUCCESS:{Style.RESET_ALL} Created file {Fore.WHITE}{filename}{Style.RESET_ALL}")
                successful_ops.append(op)
            except Exception as e:
                apply_log.append(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} Error creating file {Fore.WHITE}{filename}{Style.RESET_ALL}: {e}")
                import traceback
                apply_log.append(f"  {Fore.RED}{traceback.format_exc().splitlines()[-1]}{Style.RESET_ALL}")
                failed_ops.append(op)
                
        # Line-based replace operation - existing logic
        elif op["type"] == "replace_lines":
            # ... existing code ...
            try:
                # Check if file exists first
                if not os.path.exists(filename):
                    raise FileNotFoundError(f"File {filename} not found")
                
                # Open and read the file
                with open(filename, 'r', encoding='utf-8') as f:
                    original_lines_with_endings = f.readlines()  # Keep line endings
                    original_content = f.read()
                
                # Strip line endings to work with the content
                original_lines = [line.rstrip('\r\n') for line in original_lines_with_endings]
                num_original_lines = len(original_lines)
                
                # Maps for tracking operations - key is line number (1-based)
                replacements_map = {}  # Line replacements
                insertion_map = {}     # Line insertions (after specified line)
                deleted_lines = set()  # Set of deleted line numbers
                
                highest_line_num = 0   # Track highest accessed line for appends
                line_validity_passed = True  # Flag for overall validation
                error_details = []     # Collect specific error details
                
                # First pass: check line numbers for validity and find highest line
                for r in sorted(op.get('replacements', []), key=lambda x: x['line']):
                    # ... existing code for line-based replacements ...
                    pass
                
                # Special case: handle exact line insertions beyond file end
                has_exact_insertions = any(r.get('exact', False) for r in op.get('replacements', []))
                
                # Application Step
                if line_validity_passed:
                    # ... existing code for applying line-based changes ...
                    
                    # Success message handling...
                    success_message = f"{Fore.GREEN}✓ SUCCESS:{Style.RESET_ALL} Applied changes to {Fore.WHITE}{filename}{Style.RESET_ALL}"
                    apply_log.append(success_message)
                    successful_ops.append(op)
                else:
                    # Line number verification failed
                    apply_log.append(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} Invalid line number(s) specified for {Fore.WHITE}{filename}{Style.RESET_ALL}.")
                    for detail in error_details:
                        apply_log.append(f"  {Fore.RED}{detail}{Style.RESET_ALL}")
                    failed_ops.append(op)

            except FileNotFoundError:
                apply_log.append(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} File {Fore.WHITE}{filename}{Style.RESET_ALL} not found for REPLACE LINES.")
                failed_ops.append(op)
            except Exception as e:
                apply_log.append(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} Error processing REPLACE LINES for {Fore.WHITE}{filename}{Style.RESET_ALL}: {e}")
                import traceback
                apply_log.append(f"  {Fore.RED}{traceback.format_exc().splitlines()[-1]}{Style.RESET_ALL}")
                failed_ops.append(op)
                
        # --- Block-based replace operation - new logic ---
        elif op["type"] == "replace_block":
            try:
                if not os.path.exists(filename):
                    raise FileNotFoundError(f"File {filename} not found")
                    
                # Read the file content
                with open(filename, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Get the old code and new code from the operation
                old_code = op["old_code"]
                new_code = op["new_code"]
                
                # Check if old_code exists exactly in the file (with indentation)
                if old_code in file_content:
                    # Perfect match found - replace directly
                    new_content = file_content.replace(old_code, new_code)
                    
                    # Write the modified content back to the file
                    with open(filename, 'w', newline='\n') as f:
                        f.write(new_content)
                        
                    apply_log.append(f"{Fore.GREEN}✓ SUCCESS:{Style.RESET_ALL} Replaced code block in {Fore.WHITE}{filename}{Style.RESET_ALL}")
                    op["verified"] = True
                    successful_ops.append(op)
                else:
                    # No exact match - need to analyze what doesn't match
                    old_code_lines = old_code.splitlines()
                    file_lines = file_content.splitlines()
                    
                    # Try to find where the block should be
                    # First, find all potential starting points by matching the first line
                    potential_matches = []
                    
                    if old_code_lines:
                        for i in range(len(file_lines) - len(old_code_lines) + 1):
                            if file_lines[i].rstrip() == old_code_lines[0].rstrip():
                                potential_matches.append(i)
                    
                    # For each potential match, check the whole block
                    best_match = None
                    best_match_score = 0
                    best_mismatches = []
                    
                    for start_idx in potential_matches:
                        match_score = 0
                        current_mismatches = []
                        
                        for j in range(len(old_code_lines)):
                            if start_idx + j < len(file_lines):
                                if file_lines[start_idx + j].rstrip() == old_code_lines[j].rstrip():
                                    match_score += 1
                                else:
                                    current_mismatches.append({
                                        'file_line_num': start_idx + j + 1,  # 1-indexed line number
                                        'file_line': file_lines[start_idx + j],
                                        'old_code_line_num': j + 1,  # 1-indexed line number
                                        'old_code_line': old_code_lines[j]
                                    })
                        
                        if match_score > best_match_score:
                            best_match_score = match_score
                            best_match = start_idx
                            best_mismatches = current_mismatches
                    
                    # If we found a reasonable match (>50% matching)
                    if best_match is not None and best_match_score > len(old_code_lines) / 2:
                        match_percentage = (best_match_score / len(old_code_lines)) * 100
                        
                        # Log the mismatch information
                        apply_log.append(f"{Fore.YELLOW}⚠ PARTIAL MATCH:{Style.RESET_ALL} Found {match_percentage:.1f}% match in {Fore.WHITE}{filename}{Style.RESET_ALL} at line {best_match + 1}")
                        apply_log.append(f"{Fore.YELLOW}  The following lines don't match exactly (whitespace/indentation sensitive):{Style.RESET_ALL}")
                        
                        for mismatch in best_mismatches:
                            apply_log.append(f"{Fore.RED}  - File (L{mismatch['file_line_num']}):{Style.RESET_ALL} {repr(mismatch['file_line'])}")
                            apply_log.append(f"{Fore.RED}  - Old Code (L{mismatch['old_code_line_num']}):{Style.RESET_ALL} {repr(mismatch['old_code_line'])}")
                            apply_log.append("")  # Empty line for readability
                    else:
                        apply_log.append(f"{Fore.RED}✗ NO MATCH:{Style.RESET_ALL} Could not find matching code block in {Fore.WHITE}{filename}{Style.RESET_ALL}")
                        
                    # Mark as failed with detailed information for retry
                    op["match_details"] = {
                        "has_match": best_match is not None,
                        "match_line": best_match + 1 if best_match is not None else None,
                        "match_score": best_match_score,
                        "total_lines": len(old_code_lines),
                        "mismatches": best_mismatches
                    }
                    failed_ops.append(op)
            
            except FileNotFoundError:
                apply_log.append(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} File {Fore.WHITE}{filename}{Style.RESET_ALL} not found for REPLACE BLOCK.")
                failed_ops.append(op)
            except Exception as e:
                apply_log.append(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} Error processing REPLACE BLOCK for {Fore.WHITE}{filename}{Style.RESET_ALL}: {e}")
                import traceback
                apply_log.append(f"  {Fore.RED}{traceback.format_exc().splitlines()[-1]}{Style.RESET_ALL}")
                failed_ops.append(op)

    # Print the apply log inside a box
    box_color = Fore.RED if failed_ops else Fore.GREEN
    print_boxed("Applying File Operations Results", "\n".join(apply_log), color=box_color)

    return {"successful": successful_ops, "failed": failed_ops}

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

def _format_content_with_lines(content):
    """Helper function to prepend line numbers to content."""
    lines = content.splitlines()
    numbered_lines = [f"{i+1} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)

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
    
    # Add information about all files in workspace
    context_lines.append(f"\n{Fore.BLUE}Files AVAILABLE in workspace:{Style.RESET_ALL}")
    files_available = sorted(list(set(file_history["current_workspace"])))
    if files_available:
        for file in files_available:
            context_lines.append(f"- {Fore.WHITE}{file}{Style.RESET_ALL}")
    else:
        context_lines.append(f"{Fore.CYAN}No files in workspace yet.{Style.RESET_ALL}")
    
    # Add CRITICAL warning about using correct filenames and file availability
    context_lines.append(f"\n{Fore.RED}⚠️ CRITICAL:{Style.RESET_ALL} Always use exact filenames from context.")
    context_lines.append(f"{Fore.RED}⚠️ IMPORTANT:{Style.RESET_ALL} All files listed above are fully AVAILABLE for you to use.")
    
    # Add the CURRENT CONTENT of all created/modified files with line numbers
    context_lines.append(f"\n{Style.BRIGHT}{Fore.MAGENTA}--- CURRENT FILE CONTENTS (Line Numbered) ---{Style.RESET_ALL}") # Updated title
    all_files = sorted(list(set(file_history["created"] + file_history["modified"] + file_history["current_workspace"])))
    
    if not all_files:
         context_lines.append(f"{Fore.CYAN}No files to show content for yet.{Style.RESET_ALL}")
    else:
        for filename in all_files:
            try:
                if os.path.exists(filename):
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Add file delimiters and NUMBERED content
                        context_lines.append(f"\n{Fore.CYAN}=== START: {filename} ==={Style.RESET_ALL}")
                        context_lines.append(f"```")
                        # Use helper to add line numbers
                        context_lines.append(_format_content_with_lines(content))
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
    context_header = f"{Style.BRIGHT}{Fore.MAGENTA}--- FILE CONTEXT (ALL FILES LISTED HERE ARE AVAILABLE TO YOU) ---{Style.RESET_ALL}\n"
    context_footer = f"\n{Style.BRIGHT}{Fore.MAGENTA}--- END FILE CONTEXT (DO NOT ASK FOR FILES ALREADY LISTED ABOVE) ---{Style.RESET_ALL}"
    return context_header + "\n".join(context_lines) + context_footer

def retry_failed_replacements(failed_ops, client_or_model, provider, model_name, file_history, conversation_history, max_retries=2): # Updated signature
    """Attempts to automatically retry failed REPLACE operations."""
    retry_attempt = 1
    # Filter for retryable failures - now includes both replace_lines and replace_block types
    remaining_failed = [op for op in failed_ops if op.get('type') in ['replace_lines', 'replace_block']]
    newly_successful = []
    final_failed = [op for op in failed_ops if op.get('type') not in ['replace_lines', 'replace_block']] # Pass non-retryable failures through

    # For block replacements, limit retries to 10 attempts
    max_block_retries = 10
    
    while remaining_failed and retry_attempt <= max(max_retries, max_block_retries):
        print(f"\n{Fore.YELLOW}--- Attempting Auto-Retry {retry_attempt}/{max(max_retries, max_block_retries)} for Failed Replacements ---{Style.RESET_ALL}")

        # --- Construct Retry Prompt ---
        retry_prompt_parts = []
        retry_prompt_parts.append(get_system_prompt())
        retry_prompt_parts.append(generate_file_context(file_history))
        # ... (include history if needed) ...
        history_to_include = min(3, len(conversation_history))
        if history_to_include > 0:
             prompt_history_lines = []
             start_index = len(conversation_history) - history_to_include
             for i in range(start_index, len(conversation_history)):
                 entry = conversation_history[i]
                 role_prefix = f"{entry['role'].upper()}: "
                 prompt_history_lines.append(role_prefix + entry['content'])
             retry_prompt_parts.append(f"{Style.BRIGHT}{Fore.MAGENTA}--- RECENT CONVERSATION HISTORY ---{Style.RESET_ALL}\n" + "\n\n".join(prompt_history_lines))
             retry_prompt_parts.append(f"\n{Style.BRIGHT}{Fore.MAGENTA}--- END HISTORY ---{Style.RESET_ALL}")

        # --- Construct a tailored retry message based on operation types ---
        line_replace_ops = [op for op in remaining_failed if op.get('type') == 'replace_lines']
        block_replace_ops = [op for op in remaining_failed if op.get('type') == 'replace_block']
        
        retry_message = f"**Retry Request (Attempt {retry_attempt}):** Your previous attempt to replace content in the file(s) below **FAILED**.\n\n"

        # Handle line-based replace retries (original functionality)
        if line_replace_ops:
            retry_message += f"**For LINE-BASED REPLACEMENTS:**\n"
            retry_message += f"1. **CAREFULLY CHECK** the `--- FILE CONTEXT ---` section provided *above* to find the correct line numbers for your changes.\n"
            retry_message += f"2. For **each file** listed below with line issues, construct the `====== REPLACE` tag again using the format `NUMBER | NEW_CONTENT` for each line you want to replace.\n"
            retry_message += f"3. **ENSURE ALL `NUMBER`s are valid line numbers** present in the file's context.\n\n"
            
            files_with_line_issues = {op['filename'] for op in line_replace_ops}
            for filename in files_with_line_issues:
                retry_message += f"**Line-Based Replace File:** `{filename}`\n"
                retry_message += f"  *Verify line numbers against the line-numbered context above.*\n\n"
        
        # Handle block-based replace retries (new functionality)
        if block_replace_ops:
            retry_message += f"**For BLOCK-BASED REPLACEMENTS:**\n"
            retry_message += f"1. Your attempted replacements using `======= REPLACE` with exact code blocks did not match.\n"
            retry_message += f"2. Below are details about what didn't match correctly. Please fix your code block to **EXACTLY** match the file contents.\n"
            retry_message += f"3. Pay special attention to whitespace, indentation, and line breaks.\n\n"
            
            for op in block_replace_ops:
                filename = op['filename']
                retry_message += f"**Block Replace File:** `{filename}`\n"
                
                if op.get('match_details') and op['match_details'].get('has_match'):
                    # Partial match - show specific mismatches
                    match_line = op['match_details']['match_line']
                    match_score = op['match_details']['match_score']
                    total_lines = op['match_details']['total_lines']
                    match_percent = (match_score / total_lines) * 100
                    
                    retry_message += f"  * Found partial match ({match_percent:.1f}% matching) at line {match_line}\n"
                    retry_message += f"  * The following lines didn't match (showing file content vs. what you provided):\n\n"
                    
                    for mismatch in op['match_details'].get('mismatches', []):
                        retry_message += f"    - **File Line {mismatch['file_line_num']}:** `{mismatch['file_line']}`\n"
                        retry_message += f"    - **Your Code Line {mismatch['old_code_line_num']}:** `{mismatch['old_code_line']}`\n\n"
                else:
                    # No match found
                    retry_message += f"  * **NO MATCHING BLOCK FOUND** - Please check the file content carefully and ensure your 'old code' block matches exactly.\n\n"
                
                retry_message += f"  * When fixing, use this format:\n"
                retry_message += f"```\n======= REPLACE {filename}\n<exact old code with correct indentation>\n======= TO\n<new code>\n======= REND\n```\n\n"
        
        retry_message += "**Provide ONLY the corrected replacement tag(s) below using valid content. Do NOT include explanations or `[END]` tags.**"
        
        retry_prompt_parts.append(f"{Fore.RED}{Style.BRIGHT}{retry_message}{Style.RESET_ALL}")
        full_retry_prompt = "\n\n".join(retry_prompt_parts)
        conversation_history.append({"role": "system", "content": f"Initiating auto-retry {retry_attempt}/{max(max_retries, max_block_retries)} for {len(remaining_failed)} failed replacement ops."})

        # --- Call Model ---
        # ... (call model, get response) ...
        print(f"{Style.DIM}--- Asking AI for corrected replacement operations... ---{Style.RESET_ALL}")
        retry_response_text = ""
        try:
            # --- Use correct API based on provider --- Start
            if provider == "google":
                 # Construct Google request content (similar to main chat loop, but simpler history?)
                 # Let's just use the full_retry_prompt directly for Google for simplicity in retry
                 retry_response = client_or_model.generate_content(full_retry_prompt)
                 retry_response_text = getattr(retry_response, 'text', '')
                 if not retry_response_text and hasattr(retry_response, 'parts'):
                     retry_response_text = "".join(part.text for part in retry_response.parts if hasattr(part, 'text'))
            elif provider == "openrouter":
                 # Construct OpenAI messages (System + User retry message)
                 retry_messages = [
                     {"role": "system", "content": get_system_prompt()}, # Add base system prompt
                     # Maybe add file context again?
                     {"role": "user", "content": full_retry_prompt} # Pass the constructed retry prompt
                 ]
                 retry_response = client_or_model.chat.completions.create(
                     model=model_name,
                     messages=retry_messages,
                     # No streaming needed for retry, just get the full response
                 )
                 if retry_response.choices:
                     retry_response_text = retry_response.choices[0].message.content
            # --- Use correct API based on provider --- End

            # Correctly indented block
            print(f"{Fore.CYAN}AI Retry Response:\n{retry_response_text}{Style.RESET_ALL}")
            conversation_history.append({"role": "model", "content": f"[Retry {retry_attempt} Response]\n{retry_response_text}"})
        except Exception as e:
            # Correctly indented block
            print(f"{Back.RED}{Fore.WHITE} ERROR during retry generation: {e} {Style.RESET_ALL}")
            conversation_history.append({"role": "system", "content": f"Error during retry attempt {retry_attempt}: {e}"})
            final_failed.extend(remaining_failed)
            remaining_failed = []
            break

        # --- Process Retry Response ---
        retry_file_ops = parse_file_operations(retry_response_text)
        
        # Sort the operations by type - we want to process block replacements first as they're more targeted
        ops_to_apply_this_retry = []
        
        # First add block replacements
        block_ops = [op for op in retry_file_ops if op.get('type') == 'replace_block' and 
                     any(failed_op['filename'] == op['filename'] for failed_op in block_replace_ops)]
        ops_to_apply_this_retry.extend(block_ops)
        
        # Then add line replacements
        line_ops = [op for op in retry_file_ops if op.get('type') == 'replace_lines' and 
                    any(failed_op['filename'] == op['filename'] for failed_op in line_replace_ops)]
        ops_to_apply_this_retry.extend(line_ops)
        
        if not ops_to_apply_this_retry:
            print(f"{Fore.YELLOW}No valid replacement tags found in AI's retry response for the failed files.{Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}Applying corrections from retry attempt {retry_attempt}...{Style.RESET_ALL}")
            retry_apply_result = apply_changes(ops_to_apply_this_retry) # Apply the parsed ops

            # Update history
            # ... (update history logic - unchanged) ...
            op_summary_lines = [f"Auto-Retry {retry_attempt} Apply Results:"]
            successful_filenames_this_retry = {op['filename'] for op in retry_apply_result.get('successful', [])}
            if retry_apply_result['successful']:
                 op_summary_lines.append(f"  {Fore.GREEN}Successful ({len(retry_apply_result['successful'])}):{Style.RESET_ALL} {', '.join(successful_filenames_this_retry)}")
                 newly_successful.extend(retry_apply_result['successful'])
            if retry_apply_result['failed']:
                 op_summary_lines.append(f"  {Fore.RED}Failed ({len(retry_apply_result['failed'])}):{Style.RESET_ALL} {', '.join([op['filename'] for op in retry_apply_result['failed']])}")
            conversation_history.append({"role": "system", "content": "\n".join(op_summary_lines)})

            # Update remaining_failed - keep track of operation types separately
            current_remaining = []
            
            # Check which operations are still failing
            for op in remaining_failed:
                # Check if this operation was successfully applied in this retry
                if op['type'] == 'replace_lines':
                    if not any(success_op['filename'] == op['filename'] and success_op['type'] == 'replace_lines' 
                              for success_op in retry_apply_result.get('successful', [])):
                        current_remaining.append(op)
                elif op['type'] == 'replace_block':
                    if not any(success_op['filename'] == op['filename'] and success_op['type'] == 'replace_block' 
                              for success_op in retry_apply_result.get('successful', [])):
                        # Check max retries for block operations
                        if op['type'] == 'replace_block' and retry_attempt >= max_block_retries:
                            final_failed.append(op)
                        else:
                            current_remaining.append(op)
                            
            remaining_failed = current_remaining

        # Stop line-based retries early if we've reached the standard max_retries
        if retry_attempt >= max_retries:
            # Move all line-based operations to final_failed
            line_ops_to_move = [op for op in remaining_failed if op['type'] == 'replace_lines']
            final_failed.extend(line_ops_to_move)
            # Keep only block-based operations for further retries
            remaining_failed = [op for op in remaining_failed if op['type'] == 'replace_block']

        retry_attempt += 1
        if remaining_failed and retry_attempt <= max(max_retries, max_block_retries):
             time.sleep(1)

    # ... (final logging and return) ...
    final_failed.extend(remaining_failed) # Add any ops that still failed after retries
    print(f"{Fore.YELLOW}--- Auto-Retry Finished ---{Style.RESET_ALL}")
    if newly_successful:
         print(f"{Fore.GREEN}Successfully applied corrections for: {', '.join(list({op['filename'] for op in newly_successful}))}{Style.RESET_ALL}")
    if final_failed:
         print(f"{Fore.RED}Still failed after retries: {', '.join(list({op['filename'] for op in final_failed}))}{Style.RESET_ALL}")
    return {"newly_successful": newly_successful, "final_failed": final_failed}

def process_mentions(user_input):
    """
    Find @path/to/file mentions and @codebase in user input, read files/get structure,
    and prepend their content (with line numbers) to the input string for the model.
    Returns the processed input and the original input with mentions removed.
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

                print(f"{Fore.CYAN}  Injecting content from: {filepath} (with line numbers){Style.RESET_ALL}") # Updated print

                # Format with line numbers
                numbered_content = _format_content_with_lines(file_content)

                prepended_content += f"\n{Style.BRIGHT}{Fore.MAGENTA}--- MENTIONED FILE: {filepath} (Line Numbered) ---{Style.RESET_ALL}\n" # Updated title
                prepended_content += f"```\n{numbered_content}\n```\n"
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


def chat_with_model(client_or_model, provider, model_name): # Modified signature
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
    conversation_history = [] # Reset history for each run for simplicity now
    # If you want persistent history across runs, load it here based on provider/model?
    
    # --- Initialize the custom completer ---
    mention_completer = MentionCompleter()
    # Combine with history auto-suggest (will handle non-mention cases)
    # Note: We don't need a separate completer for history; AutoSuggestFromHistory works alongside custom completers.

    # Keep track of conversation to maintain context
    conversation_history = []
    # Store context that needs to be prepended *outside* the AI's direct turn
    pending_context_injection = ""

    # --- Add Custom Exception Class --- Start
    class AutoFixRequired(Exception):
        """Custom exception to signal that the auto-fix loop needs to continue."""
        pass
    # --- Add Custom Exception Class --- End

    while True:
        try:
            # --- Prepend any pending context (e.g., from user file selection or auto-fix) ---
            current_context_for_model = pending_context_injection
            pending_context_injection = "" # Clear after use

            # Add separator
            print(f"\n{Fore.BLUE}{H * (min(os.get_terminal_size().columns, 80))}{Style.RESET_ALL}")

            # --- Get User Input --- 
            # (Only prompt user if there isn't context waiting from auto-fix/file selection)
            if not current_context_for_model:
                # ... (Existing user input prompt logic remains here) ...
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
                user_input_for_model, user_input_for_history = process_mentions(raw_user_input)
                if user_input_for_history.strip():
                     conversation_history.append({"role": "user", "content": user_input_for_history})
                # Add user input (after mentions processed) to the context for this turn
                current_context_for_model += user_input_for_model

            # --- Prepare Prompt for AI ---
            system_prompt = get_system_prompt()
            file_context_for_prompt = generate_file_context(file_history)

            # --- Format History for Model --- 
            # ... (Existing history formatting logic for Google/OpenRouter) ...
            history_for_model = []
            if provider == "google":
                 combined_history = []
                 temp_history = conversation_history[-10:] 
                 for entry in temp_history:
                      role = entry['role']
                      if role in ['user', 'model']:
                          combined_history.append({"role": role, "parts": [entry['content']]})
                 history_for_model = combined_history 
            elif provider == "openrouter":
                 openai_messages = [{"role": "system", "content": system_prompt}]
                 temp_history = conversation_history[-10:] 
                 for entry in temp_history:
                      role = entry['role']
                      if role == 'user':
                          openai_messages.append({"role": "user", "content": entry['content']})
                      elif role == 'model':
                           openai_messages.append({"role": "assistant", "content": entry['content']})
                 history_for_model = openai_messages 
            
            # Display history in the console (unified format)
            history_to_display = min(10, len(conversation_history))
            # ... (Existing history display formatting) ...
            prompt_history_formatted = []
            start_index = len(conversation_history) - history_to_display
            for i in range(start_index, len(conversation_history)):
                 entry = conversation_history[i]
                 role = entry['role']
                 prefix = f"{role.upper()}: "
                 if role == 'system': prefix = f"{Fore.YELLOW}SYSTEM NOTE:{Style.RESET_ALL} "
                 elif role == 'user': prefix = f"{Fore.GREEN}USER:{Style.RESET_ALL} "
                 elif role == 'model': prefix = f"{Fore.CYAN}MODEL:{Style.RESET_ALL} "
                 prompt_history_formatted.append(prefix + entry['content'])
            if prompt_history_formatted:
                 print(f"{Style.BRIGHT}{Fore.MAGENTA}--- CONVERSATION HISTORY (Last {history_to_display}) ---{Style.RESET_ALL}\n" + "\n\n".join(prompt_history_formatted))
                 print(f"\n{Style.BRIGHT}{Fore.MAGENTA}--- END HISTORY ---{Style.RESET_ALL}")

            # --- Construct Final Prompt/Messages --- 
            # ... (Existing provider-specific prompt/message construction) ...
            generation_request_content = [] # For google
            if provider == "google":
                 prompt_string_for_google = "\n\n".join([system_prompt, file_context_for_prompt, current_context_for_model if current_context_for_model else ""])
                 generation_request_content = history_for_model + [{"role": "user", "parts": [current_context_for_model if current_context_for_model else "Continue."]}]
            elif provider == "openrouter":
                 full_user_content = f"{file_context_for_prompt}\n\n{current_context_for_model}"
                 history_for_model.append({"role": "user", "content": full_user_content})

            # --- Model Interaction Loop (Multi-Turn) ---
            all_responses_this_turn = []
            is_end_of_turn = False
            ask_for_files_detected = False
            ask_to_user_detected = False
            user_question = None
            files_to_ask_user_for = []

            print(f"\n{Style.DIM}--- CodAgent Thinking ---{Style.RESET_ALL}")

            while not is_end_of_turn and not ask_for_files_detected:
                print(f"\n{Style.BRIGHT}{Fore.GREEN}>>> AI Response Segment {len(all_responses_this_turn) + 1} >>>{Style.RESET_ALL}")
                current_segment_text = ""
                stream_error_occurred = False
                segment_apply_result = None # Reset apply result for this segment
                executed_command_results = [] # Reset command results for this segment

                try:
                    # --- Call Correct API --- Start
                    if provider == "google":
                        # Pass the potentially updated generation_request_content
                        response_stream = client_or_model.generate_content(generation_request_content, stream=True)
                        for chunk in response_stream:
                             try:
                                 chunk_text = chunk.text
                                 # ... (Logic to hide [END] tag during print) ...
                                 text_to_print = chunk_text
                                 temp_full_segment = current_segment_text + chunk_text
                                 if temp_full_segment.rstrip().endswith("[END]"):
                                     tag_start_index = temp_full_segment.rstrip().rfind("[END]")
                                     prev_segment_len = len(current_segment_text)
                                     if tag_start_index >= prev_segment_len:
                                         tag_start_in_chunk = tag_start_index - prev_segment_len
                                         text_to_print = chunk_text[:tag_start_in_chunk]
                                 print(text_to_print, end='', flush=True) 
                                 current_segment_text += chunk_text 
                             except ValueError: pass
                             except Exception as e_text_access: print(f"\n{Fore.RED}Error processing Google stream chunk text: {e_text_access}{Style.RESET_ALL}", flush=True)
                    elif provider == "openrouter":
                         # Pass the potentially updated history_for_model
                         response_stream = client_or_model.chat.completions.create(model=model_name, messages=history_for_model, stream=True)
                         for chunk in response_stream:
                              if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                                   chunk_text = chunk.choices[0].delta.content
                                   # ... (Logic to hide [END] tag during print) ...
                                   text_to_print = chunk_text
                                   temp_full_segment = current_segment_text + chunk_text
                                   if temp_full_segment.rstrip().endswith("[END]"):
                                       tag_start_index = temp_full_segment.rstrip().rfind("[END]")
                                       prev_segment_len = len(current_segment_text)
                                       if tag_start_index >= prev_segment_len:
                                           tag_start_in_chunk = tag_start_index - prev_segment_len
                                           text_to_print = chunk_text[:tag_start_in_chunk]
                                   print(text_to_print, end='', flush=True) 
                                   current_segment_text += chunk_text 
                    # --- Call Correct API --- End
                    print() # Newline after segment stream
                except Exception as model_error:
                     print(f"\n{Back.RED}{Fore.WHITE} ERROR during model generation request: {model_error} {Style.RESET_ALL}")
                     current_segment_text = "[CodAgent Error: Generation failed]"
                     is_end_of_turn = True
                     stream_error_occurred = True

                # --- Process Tags and Execute Commands PER SEGMENT --- Start
                segment_for_processing = current_segment_text 
                segment_to_log = current_segment_text # Store original segment for logging

                # 1. Check for ====== ASK_FOR_FILES first
                if not stream_error_occurred:
                    extracted_files, segment_without_ask_tag = parse_ask_for_files(segment_for_processing)
                    if extracted_files:
                        print(f"\n{Fore.YELLOW}[CodAgent needs files... Processing request.]){Style.RESET_ALL}")
                        ask_for_files_detected = True
                        files_to_ask_user_for = extracted_files
                        segment_to_log = segment_without_ask_tag # Log text without the tag
                        all_responses_this_turn.append(segment_to_log)
                        conversation_history.append({"role": "model", "content": segment_to_log}) # Add AI msg asking for files
                        break # Break inner loop immediately to handle ASK prompt
                
                # 1.5 Check for ====== ASK_TO_USER
                if not stream_error_occurred and not ask_for_files_detected:
                    extracted_question, segment_without_ask_tag = parse_ask_to_user(segment_for_processing)
                    if extracted_question:
                        print(f"\n{Fore.YELLOW}[CodAgent is asking you a question... ({extracted_question['format']} format)]{Style.RESET_ALL}")
                        ask_to_user_detected = True
                        user_question = extracted_question
                        segment_to_log = segment_without_ask_tag # Log text without the tag
                        all_responses_this_turn.append(segment_to_log)
                        conversation_history.append({"role": "model", "content": segment_to_log}) # Add AI msg asking question
                        break # Break inner loop immediately to handle question

                # 2. Check for [END] tag 
                if not stream_error_occurred and not ask_for_files_detected and not ask_to_user_detected:
                     parsed_segment, is_end_from_tag = parse_end_response(segment_for_processing)
                     if is_end_from_tag:
                          segment_for_processing = parsed_segment # Use text without tag for actions
                          segment_to_log = parsed_segment # Log text without the tag
                          is_end_of_turn = True # Mark to terminate loop after this segment's actions

                # Log the AI's segment response (potentially without tags)
                if segment_to_log.strip() and not stream_error_occurred:
                    all_responses_this_turn.append(segment_to_log)
                    # Add model response to history *before* execution results for clarity
                    conversation_history.append({"role": "model", "content": segment_to_log})
                elif stream_error_occurred:
                    # Already logged error, don't add failed segment text
                    pass 

                # --- Execute File Operations for this Segment --- 
                if not stream_error_occurred and not ask_for_files_detected:
                    segment_file_ops = parse_file_operations(segment_for_processing)
                    if segment_file_ops:
                        print("\n" + "="*5 + f" File Operations Proposed (Segment {len(all_responses_this_turn)}) " + "="*5)
                        if preview_changes(segment_file_ops):
                            segment_apply_result = apply_changes(segment_file_ops)
                            # Update system history with results
                            op_summary_lines_segment = [f"File Operations Status (Segment {len(all_responses_this_turn)}):"]
                            if segment_apply_result.get('successful'): op_summary_lines_segment.append(f"  {Fore.GREEN}Successful ({len(segment_apply_result['successful'])}):{Style.RESET_ALL} {', '.join(list({op['filename'] for op in segment_apply_result['successful']}))}")
                            if segment_apply_result.get('failed'): op_summary_lines_segment.append(f"  {Fore.RED}Failed ({len(segment_apply_result['failed'])}):{Style.RESET_ALL} {', '.join(list({op['filename'] for op in segment_apply_result['failed']}))}")
                            conversation_history.append({"role": "system", "content": "\n".join(op_summary_lines_segment)})
                            # Update internal file state tracker
                            for op in segment_apply_result.get("successful", []):
                                norm_filename = os.path.normpath(op["filename"])
                                # ... (file_history update logic remains same) ...
                                if op["type"] == "create":
                                    if norm_filename not in file_history["created"]: file_history["created"].append(norm_filename)
                                    if norm_filename not in file_history["current_workspace"]: file_history["current_workspace"].append(norm_filename)
                                    if norm_filename in file_history["modified"]: file_history["modified"].remove(norm_filename)
                                elif op["type"] == "replace_lines": 
                                    if norm_filename not in file_history["modified"] and norm_filename not in file_history["created"]: file_history["modified"].append(norm_filename)
                                    if norm_filename not in file_history["current_workspace"]: file_history["current_workspace"].append(norm_filename)

                            # --- Run Syntax Check / Auto-Fix --- 
                            successful_ops_this_segment = segment_apply_result.get("successful", [])
                            if successful_ops_this_segment:
                                # ... (Existing syntax check logic) ...
                                python_files_changed = [op['filename'] for op in successful_ops_this_segment if op['filename'].endswith('.py')]
                                syntax_errors_found = {} 
                                if python_files_changed:
                                    print(f"\n{Fore.CYAN}--- Running Syntax Checks on: {', '.join(python_files_changed)} ---{Style.RESET_ALL}")
                                    for filename in python_files_changed:
                                         if os.path.exists(filename):
                                            try:
                                                syntax_check_result = subprocess.run([sys.executable, "-m", "py_compile", filename], capture_output=True, text=True, check=False)
                                                if syntax_check_result.returncode != 0 and syntax_check_result.stderr:
                                                    error_output = syntax_check_result.stderr.strip()
                                                    syntax_errors_found[filename] = error_output
                                                    print(f"{Fore.RED}✗ Syntax Error detected in {filename}:{Style.RESET_ALL}\n{error_output}")
                                                else: print(f"{Fore.GREEN}✓ Syntax OK for {filename}{Style.RESET_ALL}")
                                            except Exception as e: print(f"{Fore.RED}Error running syntax check on {filename}: {e}{Style.RESET_ALL}")
                                         else: print(f"{Fore.YELLOW}Skipping syntax check for {filename} (file not found after apply?){Style.RESET_ALL}")

                                if syntax_errors_found:
                                    print(f"\n{Fore.YELLOW}--- Initiating Auto-Fix Check (Syntax Errors Detected) ---{Style.RESET_ALL}")
                                    updated_file_context = generate_file_context(file_history)
                                    error_details = "\n".join([f"File: `{fname}`\nError:\n```\n{err}\n```" for fname, err in syntax_errors_found.items()])
                                    affected_filenames = list(syntax_errors_found.keys())
                                    auto_fix_prompt = f"""{updated_file_context}\n\n{Style.BRIGHT}{Fore.RED}SYSTEM CHECK - SYNTAX ERROR:{Style.RESET_ALL} The following syntax error(s) were detected in the file(s) you just modified:\n\n{error_details}\n\n**Your Task:** Review the errors and the code context above. Provide `======= REPLACE` command(s) to fix **only these specific errors**. Do NOT use `[END]`."""
                                    pending_context_injection = auto_fix_prompt
                                    conversation_history.append({"role": "system", "content": f"Auto-fix check initiated due to syntax errors in: {', '.join(affected_filenames)}"})                 
                                    raise AutoFixRequired() # Use exception to break inner loop and continue outer
                        else:
                            print(f"{Fore.YELLOW}✗ File operations skipped by user (segment {len(all_responses_this_turn)}).{Style.RESET_ALL}")
                            conversation_history.append({"role": "system", "content": f"User skipped proposed file operations (segment {len(all_responses_this_turn)})."})
                
                # --- Execute Terminal Commands for this Segment --- 
                if not stream_error_occurred and not ask_for_files_detected:
                    segment_terminal_commands = parse_terminal_commands(segment_for_processing)
                    if segment_terminal_commands:
                        # ... (Existing terminal command preview, confirmation, execution logic) ...
                        print("\n" + "="*5 + f" Terminal Commands Proposed (Segment {len(all_responses_this_turn)}) " + "="*5)
                        print_boxed(f"Terminal Commands Preview (Segment {len(all_responses_this_turn)})", "\n".join([f"- {cmd}" for cmd in segment_terminal_commands]), color=Fore.YELLOW)
                        print("-" * 30)
                        confirm_terminal = input(f"{Style.BRIGHT}{Fore.CYAN}Execute these commands? (y/n): {Style.RESET_ALL}").lower().strip()
                        if confirm_terminal.startswith('y'):
                            print(f"{Fore.YELLOW}Executing commands...{Style.RESET_ALL}")
                            for command in segment_terminal_commands:
                                result = execute_terminal_command(command)
                                executed_command_results.append({"command": command, "result": result})
                            # Update system history with results
                            cmd_summary_lines_segment = [f"Terminal Execution Results (Segment {len(all_responses_this_turn)}):"]
                            for res in executed_command_results:
                                # ... (status formatting) ...
                                status = f"{Fore.GREEN}✓ SUCCESS{Style.RESET_ALL}" if res['result']['returncode'] == 0 else f"{Fore.RED}✗ FAILED (Code: {res['result']['returncode']}){Style.RESET_ALL}"
                                cmd_summary_lines_segment.append(f"`{res['command']}`: {status}")
                                if res['result']['stdout']: cmd_summary_lines_segment.append(f"  Output: {res['result']['stdout'][:100]}{'...' if len(res['result']['stdout']) > 100 else ''}")
                                if res['result']['stderr']: cmd_summary_lines_segment.append(f"  {Fore.RED}Errors:{Style.RESET_ALL} {res['result']['stderr'][:100]}{'...' if len(res['result']['stderr']) > 100 else ''}")
                            
                            # Add the original summary to conversation history for display purposes
                            conversation_history.append({"role": "system", "content": "\n".join(cmd_summary_lines_segment)})
                            
                            # ADDED: Also add the AI-specific detailed log directly to the model for better context
                            for res in executed_command_results:
                                if 'ai_log' in res['result']:
                                    # Add a separate entry for each command's detailed output in clean format
                                    # This makes it clear to the AI what each command returned
                                    ai_log_entry = f"```\n{res['result']['ai_log']}\n```"
                                    # Change from system role to user role to make it more visible to the AI
                                    conversation_history.append({"role": "user", "content": f"Terminal command output:\n{ai_log_entry}"})
                                    # Add model acknowledgment to force recognition of the output
                                    conversation_history.append({"role": "model", "content": "I've received the terminal output above. I'll analyze it now."})
                            
                            print(f"{Fore.GREEN}Finished executing segment commands.{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.YELLOW}✗ Terminal commands skipped by user (segment {len(all_responses_this_turn)}).{Style.RESET_ALL}")
                            conversation_history.append({"role": "system", "content": f"User skipped proposed terminal commands (segment {len(all_responses_this_turn)})."})
                
                # --- Process Tags and Execute Commands PER SEGMENT --- End

                # Check if inner loop should terminate based on END tag
                if is_end_of_turn:
                     print(f"\n{Style.DIM}--- CodAgent Finished Turn ([END] detected in segment {len(all_responses_this_turn)}) ---{Style.RESET_ALL}")
                     break # Break the inner while loop

                # Prepare for next segment (if loop wasn't broken by END, ASK, or AutoFix)
                print(f"\n{Style.DIM}--- CodAgent Continuing (More segments expected) ---{Style.RESET_ALL}")
                # Update history structure for the next AI call with a CONTINUE prompt
                if provider == "google":
                    # generation_request_content was used for the *last* call, 
                    # need to append the *result* (segment_to_log) and the new user prompt
                    generation_request_content.append({"role": "model", "parts": [segment_to_log]}) 
                    generation_request_content.append({"role": "user", "parts": ["CONTINUE."]})
                elif provider == "openrouter":
                    # history_for_model was used for the *last* call
                    # need to append the *result* (segment_to_log) and the new user prompt
                    history_for_model.append({"role": "assistant", "content": segment_to_log})
                    history_for_model.append({"role": "user", "content": "CONTINUE."})

            # --- End of AI Turn / Inner Segment Loop (`while not is_end_of_turn...`) ---

            # --- Handle ASK_FOR_FILES Interaction (Happens if inner loop broken by ask_for_files_detected) ---
            if ask_for_files_detected:
                print(f"\n{Fore.CYAN}CodAgent is asking for the following files:{Style.RESET_ALL}")
                # ... (Existing user file selection logic) ...
                files_found = []
                file_options = []
                for i, filepath in enumerate(files_to_ask_user_for):
                    if os.path.isfile(filepath):
                        print(f"{Fore.GREEN}  {i+1}. {filepath} (Found){Style.RESET_ALL}")
                        files_found.append(filepath)
                        file_options.append(filepath)
                    else:
                        print(f"{Fore.RED}  {i+1}. {filepath} (Not found){Style.RESET_ALL}")
                selected_files_content = ""
                selected_filenames_for_note = []
                if files_found:
                    print(f"\n{Fore.YELLOW}Select files by number (comma-separated) or enter to skip:{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}Example: 1,3 to select first and third files{Style.RESET_ALL}")
                    selection_input = input(f"{Style.BRIGHT}{Fore.CYAN}Selection: {Style.RESET_ALL}").strip()
                    if selection_input.strip():
                        try:
                            selected_indices = [int(idx.strip()) - 1 for idx in selection_input.split(',') if idx.strip()]
                            temp_selected_context = ""
                            for idx in selected_indices:
                                if 0 <= idx < len(file_options):
                                    selected_filepath = file_options[idx]
                                    selected_filenames_for_note.append(selected_filepath)
                                    try:
                                        with open(selected_filepath, 'r', encoding='utf-8') as f:
                                            file_content = f.read()
                                            numbered_content = _format_content_with_lines(file_content)
                                            temp_selected_context += f"\n{Fore.CYAN}=== {selected_filepath} (Line Numbered) ==={Style.RESET_ALL}\n"
                                            temp_selected_context += f"```\n{numbered_content}\n```\n"
                                            print(f"{Fore.GREEN}  ✓ Added {selected_filepath}{Style.RESET_ALL}")
                                            
                                            # Add file to tracking lists if not already there
                                            norm_filepath = os.path.normpath(selected_filepath)
                                            if norm_filepath not in file_history["current_workspace"]:
                                                file_history["current_workspace"].append(norm_filepath)
                                    except Exception as e:
                                        print(f"{Fore.RED}  ✗ Error reading {selected_filepath}: {e}{Style.RESET_ALL}")
                                else:
                                    print(f"{Fore.RED}  ✗ Invalid number skipped: {idx+1}{Style.RESET_ALL}")
                            if temp_selected_context:
                                selected_files_content = f"{Style.BRIGHT}{Fore.GREEN}--- Providing Content for User-Selected Files ---{Style.RESET_ALL}\n" + temp_selected_context
                            else: print(f"{Fore.YELLOW}No valid files selected or read.{Style.RESET_ALL}")
                        except ValueError: print(f"{Fore.RED}Invalid input format. Please enter numbers separated by commas.{Style.RESET_ALL}")
                if selected_files_content:
                    pending_context_injection = selected_files_content
                    # Update message to be more explicit about file availability
                    file_list_formatted = ", ".join([f"`{f}`" for f in selected_filenames_for_note])
                    conversation_history.append({"role": "system", "content": f"User selected and provided content for: {file_list_formatted}"})
                    # Add an explicit model acknowledgment that it has access to the files
                    conversation_history.append({"role": "model", "content": f"I now have access to the following files: {file_list_formatted}. I'll analyze them and proceed with your request."})
                    continue # Continue outer loop immediately
                else:
                    print(f"{Fore.YELLOW}Skipping file provision. Asking AI to proceed without them.{Style.RESET_ALL}")
                    pending_context_injection = f"{Fore.YELLOW}[SYSTEM NOTE: User did not provide the requested files. Proceed based on existing context or ask again if necessary.]{Style.RESET_ALL}"
                    conversation_history.append({"role": "system", "content": "User skipped providing requested files."})
                    continue # Continue outer loop immediately

            # --- Handle ASK_TO_USER Interaction ---
            elif ask_to_user_detected and user_question:
                print(f"\n{Fore.CYAN}CodAgent is asking you a question:{Style.RESET_ALL}")
                
                # Format the question based on format type
                question_format = user_question["format"]
                
                # Display the question with appropriate formatting
                if question_format == "normal":
                    question_text = user_question["question"]
                    print(f"\n{Fore.GREEN}{question_text}{Style.RESET_ALL}")
                    
                elif question_format == "options":
                    # Handle options format using the options list
                    options = user_question["options"]
                    print(f"\n{Fore.GREEN}Please select an option:{Style.RESET_ALL}")
                    for i, option in enumerate(options, 1):
                        print(f"{Fore.GREEN}{i}.{Style.RESET_ALL} {option}")
                    
                elif question_format == "yesno":
                    question_text = user_question["question"]
                    print(f"\n{Fore.GREEN}{question_text} (yes/no){Style.RESET_ALL}")
                
                # Get user response
                print(f"{Fore.YELLOW}Enter your response:{Style.RESET_ALL}")
                user_response = input(f"{Style.BRIGHT}{Fore.CYAN}Response: {Style.RESET_ALL}").strip()
                
                if user_response:
                    # Add the user's response to the conversation
                    conversation_history.append({"role": "user", "content": f"[Response to question] {user_response}"})
                    # Continue with AI interaction
                    continue # Continue outer loop immediately
                else:
                    print(f"{Fore.YELLOW}No response provided. Asking AI to proceed.{Style.RESET_ALL}")
                    conversation_history.append({"role": "user", "content": "[No response provided to question]"})
                    continue # Continue outer loop immediately

            # --- Auto-Fix was Triggered --- 
            # Handled by catching AutoFixRequired exception below

            # --- Redundant code blocks after the loop are now removed/commented --- 

        except AutoFixRequired: # Catch the custom exception
             print(f"{Fore.CYAN}--- Auto-fix required, continuing loop ---{Style.RESET_ALL}")
             continue # Continue the main `while True` loop immediately
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupt received. Exiting...{Style.RESET_ALL}")
            break # Exit main loop on Ctrl+C
        except Exception as e:
            print(f"\n{Back.RED}{Fore.WHITE} UNEXPECTED ERROR: {e} {Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            conversation_history.append({"role": "system", "content": f"An unexpected error occurred: {e}\n{traceback.format_exc()}"})

# --- Add Custom Exception Class (Moved before chat_with_model) ---
# class AutoFixRequired(Exception):
#     ...

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
    # Keep Google model as default if --omodel is not used
    parser.add_argument("--model", default="gemini-2.5-pro-exp-03-25", help="Google Gemini model to use (ignored if --omodel is set)")
    # Add OpenRouter model argument
    parser.add_argument("--omodel", default=None, help="OpenRouter model to use (e.g., 'mistralai/mistral-7b-instruct', 'google/gemini-pro'). Overrides --model.")
    # Default OpenRouter model if --omodel is present but without a value? No, let user specify.

    args = parser.parse_args()

    # --- Initialize Model ---
    # Pass all args to the initializer
    client_or_model, provider, model_name_used = initialize_model(args)

    # Start chat with the initialized model/client
    chat_with_model(client_or_model, provider, model_name_used)


if __name__ == "__main__":
    main() 
