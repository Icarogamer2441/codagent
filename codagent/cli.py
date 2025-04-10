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
import difflib
import threading # Need to import threading
import signal # Needed for sending signals in interrupt handler

# Initialize colorama
colorama.init(autoreset=True)

# --- Version --- 
from . import __version__

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

def get_system_prompt(is_reminder=False): # Added is_reminder argument
    """Generate a system prompt detailing capabilities and the auto-fix loop."""
    current_dir = os.getcwd()
    reminder_prefix = "**Reminder of Operating Instructions:**\n\n" if is_reminder else "" # Added reminder prefix
    # --- Revised System Prompt ---
    system_prompt = f"""{reminder_prefix}**You are CodAgent:** An AI assistant operating in `{current_dir}`. Your goal is to fulfill user requests by modifying files and running terminal commands ACCURATELY.

--- [END] TAG - CRITICAL USAGE ---
**Use `[END]` ONLY when:**
1. You have **FULLY COMPLETED ALL** planned steps for the user's current request.
2. Your response is a **single, complete answer** to a question that doesn't require further action from you.
3. You have just used `ASK_TO_USER` or `ASK_FOR_FILES` and are **waiting for the user's input**.

**DO NOT use `[END]` if:**
- You still have more steps in your plan to execute.
- Your response requires follow-up actions from you (unless it's asking the user).

**Think:** "Is my work for *this specific request* finished?" If yes, use `[END]`. If no, continue without it.
--- END [END] TAG ---

--- FILE HANDLING REMINDER ---
*   File content is **NOT** automatically available. Use `@mention` or `ASK_FOR_FILES`.
*   **CRITICAL:** If the conversation history shows the user JUST provided content for a file (via `@mention` or you used `ASK_FOR_FILES` and received it), **DO NOT immediately ask for that SAME file again**. Proceed with analyzing or modifying the content you received. Check the history first!
--- END FILE HANDLING REMINDER ---

tags: `TERMINAL`, `REPLACE/TO`, `REWRITE`, `CREATE`, `ASK_FOR_FILES`, `ASK_TO_USER`

**CRITICAL:** Use `====== END` to terminate **ALL** command blocks (TERMINAL, CREATE, REPLACE, REWRITE, ASK_FOR_FILES, ASK_TO_USER).

**Examples:**
====== TERMINAL
ls -a
====== END

====== CREATE mynewfile.js
code...
====== END

====== REWRITE myfile.js
newcode...
this replaces the old code of file with a new one you write here
====== END

====== REPLACE myfile.py
    old code line 1
    old code line 2
====== TO
    new code line 1
    new code line 2
    new code line 3
====== END

====== ASK_FOR_FILES
file1
file2
file3
...
====== END

your question
====== ASK_TO_USER format:options
option1
option2
...
====== END

====== ASK_TO_USER format:normal
your question for the user
====== END

====== ASK_TO_USER format:yesno
your yes or no question
====== END

**Example of Multi-Step Plan:**
user: create a pygame snake game

AI (1): Okay, plan: 1. Ask for main.py content, 2. Create player.py, 3. Ask for player.py content, 4. Update main.py. Starting step 1.
====== ASK_FOR_FILES
main.py
====== END
[END] # Waiting for user

AI (2): (After user provides main.py) Step 2: Creating player file.
====== CREATE player.py
# player code
====== END

AI (3): Step 3: Asking for player.py content to verify.
====== ASK_FOR_FILES
player.py
====== END
[END] # Waiting for user

AI (4): (After user provides player.py) Step 4: Updating main.py (using content from step 1).
====== REPLACE main.py
# old import
====== TO
# new import
====== END

AI (5): All steps complete. The snake game files are ready.
[END]

**Example of Single Response:**
user: how do i use @codebase?
AI (1): You can use `@codebase` to show me the project structure.
[END]

**ALWAYS:** use the ASK_TO_USER tag for questions, like with options, normal questions and yes or no questions
**ALWAYS:** use REPLACE/TO tag within big files, because this is more helpful, no one wants to rewrite the code of 500+ lines of code!
**ALWAYS:** use TO in replace, because the logic is simple and makes sence:
**CRITICAL:** **ALWAYS** use [END] if you have finished some kind of plan, steps or response, because if you don't use, you'll be in a loop of CONTINUE and you'll never stop modifying the code or sending responses, and i'll never get the request of the user again.

replace the code selected here
====== REPLACE file.ext
old code
====== TO
to this new one here
====== END
end the REPLACE/TO tag

**IMPORTANT:** To modify **multiple, non-contiguous code blocks** within the **SAME file** in one response, use a **SEPARATE** `====== REPLACE ... ====== TO ... ====== END` block for **EACH** modification. Do not combine unrelated changes into one large block.
**IMPORTANT:** If you want to replace the **whole content** of a file, please, use the `REWRITE` tag. Use `REPLACE`/`TO` only for specific blocks, not the entire file content. Remember to always use `TO` when using `REPLACE`.
**IMPORTANT:** Never use `REWRITE` like `REPLACE/TO`, because it rewrites the whole file content.
**CRITICAL:** Also, never use ``` inside tags, outsite is possible, but inside? NAH, this will detect the ``` as part of the content inside the tag

**VERY IMPORTANT:** the finish of every tag now uses `====== END`. The amount of = is 6.
**GOOD ACTION:** you can also document the project on a README.md for the user read or for other users who are testing the project and want to know what it is, its features or how to use/run them
"""
    return system_prompt

def parse_ask_for_files(response_text):
    """Parse the response text to extract suggested files from ====== ASK_FOR_FILES tag."""
    # Use re.MULTILINE and re.DOTALL. Match content between the tags. Use generic END.
    ask_pattern = r"^====== ASK_FOR_FILES\s*\n(.*?)\n====== END\s*$" # Changed AEND to END
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
    # Use re.MULTILINE and re.DOTALL to match content between the tags. Use generic END.
    ask_pattern = r"^====== ASK_TO_USER format:(\w+)\s*\n(.*?)\n====== END\s*$" # Changed QEND to END
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

    # Find TERMINAL commands using the new format. Use generic END.
    # Use re.MULTILINE and re.DOTALL
    terminal_pattern = r"^====== TERMINAL\s*\n(.*?)\n====== END\s*$" # Changed TEND to END
    for match in re.finditer(terminal_pattern, response_text, re.DOTALL | re.MULTILINE | re.IGNORECASE):
        command = match.group(1).strip()
        terminal_commands.append(command)

    return terminal_commands

def execute_terminal_command(command):
    """Execute a terminal command, capture its output, show live output, and handle Ctrl+C."""
    print("-" * 30)
    print(f"{Style.BRIGHT}{Fore.YELLOW}Executing Command:{Style.RESET_ALL} {Fore.WHITE}{command}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}--- Live Output Start (Press Ctrl+C to interrupt command) ---{Style.RESET_ALL}")

    output_lines = []
    error_lines = []
    return_code = None
    interrupted = False
    process = None # Initialize process variable

    stdout_data = {"lines": [], "lock": threading.Lock()}
    stderr_data = {"lines": [], "lock": threading.Lock()}

    def read_stream(stream, data):
        """Reads lines from a stream and prints them live."""
        try:
            for line in iter(stream.readline, ''):
                line_stripped = line.rstrip() # Keep original line ending for printing? No, strip for consistency.
                print(line_stripped, flush=True) # Print live output
                with data["lock"]:
                    data["lines"].append(line_stripped) # Store for final log
            stream.close()
        except Exception as e:
            # Handle potential errors during stream reading (e.g., decoding errors)
            error_message = f"[Stream reading error: {e}]"
            print(error_message, flush=True)
            with data["lock"]:
                data["lines"].append(error_message)


    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8', # Be explicit about encoding
            errors='replace', # Replace characters that cause decoding errors
            bufsize=1 # Line-buffered
        )

        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_data))
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_data))

        stdout_thread.start()
        stderr_thread.start()

        # Wait for threads to finish (means process streams are closed)
        stdout_thread.join()
        stderr_thread.join()

        # Wait for process to terminate and get return code
        process.wait()
        return_code = process.returncode

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}--- User Interrupt (Ctrl+C) Detected ---{Style.RESET_ALL}")
        interrupted = True
        if process:
            print(f"{Fore.YELLOW}--- Sending interrupt signal to command... ---{Style.RESET_ALL}")
            try:
                # Try terminating gracefully first
                process.terminate()
                try:
                    # Wait briefly for termination
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    print(f"{Fore.YELLOW}--- Command did not terminate gracefully, killing... ---{Style.RESET_ALL}")
                    process.kill() # Force kill if terminate fails
                return_code = -99 # Special code for user interrupt
                print(f"{Fore.YELLOW}--- Command Interrupted ---{Style.RESET_ALL}")
            except Exception as e:
                 print(f"{Fore.RED}Error during command interruption: {e}{Style.RESET_ALL}")
                 return_code = -100 # Indicate error during interruption

    except Exception as e:
        error_lines.append(f"Error executing command: {e}")
        return_code = -1 # Indicate general failure

    finally:
        # Ensure streams are closed if process was started
        if process and process.stdout: process.stdout.close()
        if process and process.stderr: process.stderr.close()
        # Ensure threads are joined if started, even if interrupted
        if 'stdout_thread' in locals() and stdout_thread.is_alive(): stdout_thread.join(timeout=1)
        if 'stderr_thread' in locals() and stderr_thread.is_alive(): stderr_thread.join(timeout=1)


    print(f"{Fore.MAGENTA}--- Live Output End ---{Style.RESET_ALL}")


    # --- Prepare Logs ---
    exec_log = [] # Log for the final box shown to user
    ai_response_log = [] # Log formatted for AI consumption

    exec_log.append(f"{Style.BRIGHT}Command:{Style.RESET_ALL} {command}")
    exec_log.append(H * visible_len(f"Command: {command}"))
    ai_response_log.append(f"Command: {command}")

    # Combine collected lines
    output = "\n".join(stdout_data["lines"])
    errors = "\n".join(stderr_data["lines"])

    if output:
        exec_log.append(f"{Fore.CYAN}--- Final Captured Output ---{Style.RESET_ALL}")
        exec_log.extend(output.splitlines()) # Add each line separately
        exec_log.append(f"{Fore.CYAN}---------------------------{Style.RESET_ALL}")
        ai_response_log.append(f"--- STDOUT ---")
        ai_response_log.extend(output.splitlines())
        ai_response_log.append("-------------")

    if errors:
        exec_log.append(f"{Fore.RED}--- Final Captured Errors ---{Style.RESET_ALL}")
        exec_log.extend(errors.splitlines())
        exec_log.append(f"{Fore.RED}---------------------------{Style.RESET_ALL}")
        ai_response_log.append(f"--- STDERR ---")
        ai_response_log.extend(errors.splitlines())
        ai_response_log.append("-------------")

    # Final status message
    if interrupted:
         status_message_user = f"{Fore.YELLOW}⚠ Command Interrupted by User (Exit Code: {return_code}){Style.RESET_ALL}"
         status_message_ai = f"Exit Code: {return_code} (User Interrupted)"
         box_color = Fore.YELLOW
    elif return_code == 0:
         status_message_user = f"{Fore.GREEN}✓ Command finished successfully (Exit Code: 0){Style.RESET_ALL}"
         status_message_ai = f"Exit Code: 0 (Success)"
         box_color = Fore.GREEN
    elif return_code is None: # Should not happen often with Popen, maybe if Popen failed
         status_message_user = f"{Fore.RED}✗ Command execution failed (Unknown Exit Code){Style.RESET_ALL}"
         status_message_ai = f"Exit Code: Unknown (Error)"
         box_color = Fore.RED
    else:
         status_message_user = f"{Fore.RED}✗ Command failed (Exit Code: {return_code}){Style.RESET_ALL}"
         status_message_ai = f"Exit Code: {return_code} (Error)"
         box_color = Fore.RED

    exec_log.append(status_message_user)
    ai_response_log.append(status_message_ai)

    # Print execution log in a box
    print_boxed(f"Final Execution Result", "\n".join(exec_log), color=box_color)
    print("-" * 30) # Separator after box

    # Return both the standard result and the AI-formatted log
    return {
        "stdout": output,
        "stderr": errors,
        "returncode": return_code,
        "interrupted": interrupted, # Add interrupted flag
        "ai_log": "\n".join(ai_response_log)
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

    # --- CREATE Operation --- Use generic END.
    create_pattern = r"^====== CREATE\s+([^\n]+)\n(.*?)\n====== END\s*$" # Changed CEND to END
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

    # --- New Block-Based REPLACE Operation --- Use generic END.
    # Changed pattern to use six equals (======)
    new_replace_pattern = r"^====== REPLACE\s+([^\n]+)\n(.*?)\n====== TO\n(.*?)\n====== END\s*$" # Changed REND to END

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

    # --- REWRITE Operation --- Use generic END.
    rewrite_pattern = r"^====== REWRITE\s+([^\n]+)\n(.*?)\n====== END\s*$" # Changed WEND to END
    for match in re.finditer(rewrite_pattern, cleaned_response, re.DOTALL | re.MULTILINE | re.IGNORECASE):
        filename = match.group(1).strip()
        raw_content = match.group(2)
        # We don't need to strip code fences here because the instruction is to never use them inside
        content = raw_content.strip()
        if content: # Allow empty file rewrite?
            file_operations.append({
                "type": "rewrite",
                "filename": filename,
                "content": content
            })
        else:
            print(f"{Fore.YELLOW}Warning: Skipping REWRITE operation for '{filename}' because content was empty.{Style.RESET_ALL}")

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

        # Preview for rewrite operation
        elif op["type"] == "rewrite":
            preview_content.append(f"{Style.BRIGHT}{Fore.RED}REWRITE File (Replace Entire Content):{Style.RESET_ALL} {Fore.WHITE}{op['filename']}{Style.RESET_ALL}")
            preview_content.append(f"{Fore.YELLOW}New Content Preview (first 5 lines):{Style.RESET_ALL}")
            content_lines = op["content"].splitlines()
            for line in content_lines[:5]:
                 preview_content.append(f"{Fore.GREEN}  + {line}{Style.RESET_ALL}") # Use + prefix for clarity
            if len(content_lines) > 5:
                 preview_content.append(f"{Fore.GREEN}  + ...{Style.RESET_ALL}")
            preview_content.append(f"{Fore.CYAN}(Total {len(content_lines)} lines){Style.RESET_ALL}")
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
                        
                    # Generate a more detailed diff report if a partial match was found
                    diff_report = ""
                    if best_match is not None:
                        diff_report = generate_diff_report(file_lines, old_code_lines, best_match)
                    
                    # Mark as failed with detailed information for retry
                    op["match_details"] = {
                        "has_match": best_match is not None,
                        "match_line": best_match + 1 if best_match is not None else None,
                        "match_score": best_match_score,
                        "total_lines": len(old_code_lines),
                        "mismatches": best_mismatches,
                        "diff_report": diff_report # Add the detailed diff
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

        # --- REWRITE operation - new logic ---
        elif op["type"] == "rewrite":
            try:
                # Create parent directories if they don't exist
                os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
                # Overwrite the file completely
                with open(filename, "w", newline='\n') as f:
                    f.write(op["content"])
                apply_log.append(f"{Fore.GREEN}✓ SUCCESS:{Style.RESET_ALL} Rewrote file {Fore.WHITE}{filename}{Style.RESET_ALL}")
                successful_ops.append(op)
            except Exception as e:
                apply_log.append(f"{Fore.RED}✗ FAILED:{Style.RESET_ALL} Error rewriting file {Fore.WHITE}{filename}{Style.RESET_ALL}: {e}")
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
    # REMOVED: Line numbering logic
    # lines = content.splitlines()
    # numbered_lines = [f"{i+1} | {line}" for i, line in enumerate(lines)]
    # return "\n".join(numbered_lines)
    # RETURN: Original content without modification
    return content

def generate_file_context(file_history):
    """Generate a context string listing files in the session and workspace."""
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
    context_lines.append(f"\n{Fore.BLUE}Files AVAILABLE in workspace (content NOT loaded):{Style.RESET_ALL}") # Updated title
    files_available = sorted(list(set(file_history["current_workspace"])))
    if files_available:
        for file in files_available:
            context_lines.append(f"- {Fore.WHITE}{file}{Style.RESET_ALL}")
    else:
        context_lines.append(f"{Fore.CYAN}No files detected in workspace.{Style.RESET_ALL}") # Updated message

    context_lines.append(f"\n{Fore.YELLOW}Use @mention or ASK_FOR_FILES to load specific file content.{Style.RESET_ALL}") # Emphasize loading

    # Add CRITICAL warning about using correct filenames and file availability
    context_lines.append(f"\n{Fore.RED}⚠️ CRITICAL:{Style.RESET_ALL} Always use exact filenames from the lists above.")
    context_lines.append(f"{Fore.RED}⚠️ IMPORTANT:{Style.RESET_ALL} File content is NOT included below. You MUST use @mention or ASK_FOR_FILES to view content before modifying.") # Updated warning

    # --- REMOVED SECTION THAT READ AND INCLUDED FILE CONTENTS ---

    # Return the formatted context string
    context_header = f"{Style.BRIGHT}{Fore.MAGENTA}--- FILE CONTEXT (File List Only) ---{Style.RESET_ALL}\n" # Updated header
    context_footer = f"\n{Style.BRIGHT}{Fore.MAGENTA}--- END FILE CONTEXT (Use @mention or ASK_FOR_FILES to load content) ---{Style.RESET_ALL}" # Updated footer
    return context_header + "\n".join(context_lines) + context_footer

def generate_diff_report(file_lines, ai_old_code_lines, best_match_start_line):
    """Generates a diff-like report comparing AI's old code and actual file lines."""
    report = []
    file_slice = file_lines[best_match_start_line : best_match_start_line + len(ai_old_code_lines)]
    diff = difflib.ndiff(file_slice, ai_old_code_lines)
    
    report.append("--- Diff Report (File vs. Your Attempted Old Code) ---")
    for i, line in enumerate(diff):
        file_line_num = best_match_start_line + i + 1
        prefix = line[:2]
        content = line[2:]
        if prefix == '+ ': # Lines only in AI's code (shouldn't happen if copied?)
            report.append(f"AI Only (?): {repr(content)}") 
        elif prefix == '- ': # Lines only in File code
            report.append(f"File (L{file_line_num}):  {repr(content)}")
        elif prefix == '  ': # Lines matching
            report.append(f"Match (L{file_line_num}): {repr(content)}")
        elif prefix == '? ': # Difference hints
            # report.append(f"Hint:         {content}") # Optional: Show hints 
            pass
    report.append("------------------------------------------------------")
    return "\n".join(report)

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
        block_replace_ops = [op for op in remaining_failed if op.get('type') == 'replace_block']
        
        retry_message_parts = [] # Build prompt piece by piece
        retry_message_parts.append(f"**Retry Request (Attempt {retry_attempt}):** Your previous attempt to replace content in the file(s) below **FAILED**.")

        # Handle line-based replace retries (original functionality - less critical now)
        # ... (keep existing line-based retry message construction)
            
        # Handle block-based replace retries (new functionality)
        if block_replace_ops:
            retry_message_parts.append(f"\n**For FAILED BLOCK-BASED REPLACEMENTS:**")
            retry_message_parts.append(f"1. Your previous `====== REPLACE` command's `old_code` block **DID NOT MATCH** the actual file content.")
            
            # Pre-fetch actual code and add to prompt
            fetched_code_details = {} # Store filename -> actual_code
            for op in block_replace_ops:
                filename = op['filename']
                actual_code_segment = ""
                try:
                    with open(filename, 'r') as f:
                        file_lines = f.read().splitlines()
                    
                    start_line_idx = -1
                    # Try finding based on the partial match line first
                    match_line = op.get('match_details', {}).get('match_line')
                    if match_line: 
                        start_line_idx = match_line - 1
                    # Fallback: try finding the first line of the AI's original bad old_code
                    else:
                        original_old_lines = op.get('old_code', '').splitlines()
                        if original_old_lines:
                             first_line = original_old_lines[0].rstrip()
                             for i, line in enumerate(file_lines):
                                 if line.rstrip() == first_line:
                                      start_line_idx = i
                                      break
                                      
                    # Extract the segment if found
                    if start_line_idx != -1:
                        num_lines = op.get('match_details', {}).get('total_lines', len(op.get('old_code', '').splitlines()))
                        actual_code_lines = file_lines[start_line_idx : start_line_idx + num_lines]
                        actual_code_segment = "\n".join(actual_code_lines)
                        fetched_code_details[filename] = actual_code_segment
                    else:
                         print(f"{Fore.RED}ERROR: Could not re-locate target code block in {filename} for retry prompt.{Style.RESET_ALL}")
                         # Proceed without fetched code for this file if lookup fails
                except Exception as e:
                    print(f"{Fore.RED}ERROR: Failed to read {filename} to fetch code for retry prompt: {e}{Style.RESET_ALL}")
            
            # Add fetched code section to prompt
            if fetched_code_details:
                retry_message_parts.append(f"\n**--- ACTUAL CODE FROM FILE (Use this for old_code!) ---**")
                for fname, code in fetched_code_details.items():
                     retry_message_parts.append(f"**File: `{fname}`**\n```\n{code}\n```")
                retry_message_parts.append(f"**-------------------------------------------------------**")

            retry_message_parts.append(f"\n2. **CRITICAL:** Look at the `--- FILE CONTEXT ---` section provided *above*. **FIND the code block** you intended to replace.")
            retry_message_parts.append(f"3. **CRITICAL:** **COPY THE CODE *EXACTLY*** from the file context (including all indentation, whitespace, and newlines) to create the `old_code` block...")
            retry_message_parts.append(f"4. **CRITICAL:** **USE THE EXACT COPIED CODE** in the example format provided below:")
            retry_message_parts.append(f"\n```\n====== REPLACE {filename}\n<EXACTLY COPIED CODE FROM FILE CONTEXT>\n====== TO\n<your new code>\n====== END\n```")

            for op in block_replace_ops:
                filename = op['filename']
                retry_message_parts.append(f"\n**Fix Required For:** `{filename}`")
                
                # Add diff report if available 
                diff_report = op.get('match_details', {}).get('diff_report')
                if diff_report:
                    retry_message_parts.append(f"  * Diff Report (showing mismatch):\n```diff\n{diff_report}\n```")
                # Add old instructions as context if needed, but emphasize using the fetched code
                # retry_message_parts.append(f"  * Use this format **using the ACTUAL code provided above**:\n```\n====== REPLACE {filename}\n<ACTUAL code from ACTUAL CODE FROM FILE section>\n====== TO\n<your new code>\n====== END\n```")
        
        retry_message_parts.append("\n\n**--- FINAL COMMAND ---**\nProvide ONLY the corrected `====== REPLACE ... END` command(s) below, using the ACTUAL file code provided above for the `old_code` part.\n**DO NOT** ask for files.\n**DO NOT** include explanations.\n**DO NOT** use `[END]`.")
        
        # --- Simplified and Focused Retry Prompt --- 
        retry_message = "\n".join(retry_message_parts)
        simplified_retry_prompt_parts = [
            generate_file_context(file_history), # Keep file context for overall reference
            f"{Fore.RED}{Style.BRIGHT}{retry_message}{Style.RESET_ALL}"
        ]
        full_retry_prompt = "\n\n".join(simplified_retry_prompt_parts)

        conversation_history.append({"role": "system", "content": f"Initiating auto-retry {retry_attempt}/{max(max_retries, max_block_retries)} for {len(remaining_failed)} failed replacement ops."})

        # --- Call Model ---
        # ... (call model, get response) ...
        print(f"{Style.DIM}--- Asking AI for corrected replacement operations... ---{Style.RESET_ALL}")
        retry_response_text = ""
        try:
            # --- Use correct API based on provider --- Start
            if provider == "google":
                 # Send the simplified prompt directly
                 retry_response = client_or_model.generate_content(full_retry_prompt)
                 retry_response_text = getattr(retry_response, 'text', '')
                 if not retry_response_text and hasattr(retry_response, 'parts'):
                     retry_response_text = "".join(part.text for part in retry_response.parts if hasattr(part, 'text'))
            elif provider == "openrouter":
                 # Construct minimal messages for OpenRouter focused on the task
                 retry_messages = [
                     {"role": "system", "content": "You are an AI assistant helping fix a failed code replacement. Focus ONLY on the user request."}, 
                     {"role": "user", "content": full_retry_prompt} # Pass the simplified prompt
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
        ops_to_apply_this_retry = []
        invalid_retry_ops = []
        
        # Filter and validate retry operations
        for retry_op in retry_file_ops:
            # Check block replaces for empty old_code (which caused the hallucination)
            if retry_op.get('type') == 'replace_block':
                original_failed_op = next((failed_op for failed_op in block_replace_ops if failed_op['filename'] == retry_op['filename']), None)
                if original_failed_op: # Make sure it corresponds to a failed op we asked about
                    if not retry_op.get('old_code', '').strip():
                        print(f"{Fore.RED}✗ Invalid Retry: AI provided empty old_code block for {retry_op['filename']}. Skipping this attempt.{Style.RESET_ALL}")
                        invalid_retry_ops.append(retry_op) # Track invalid attempt
                        continue # Skip this invalid operation
                    else:
                        ops_to_apply_this_retry.append(retry_op)
            # Include line replacements if they are valid (assuming parse_file_operations handles basic structure)
            elif retry_op.get('type') == 'replace_lines':
                 line_replace_ops = [op for op in remaining_failed if op.get('type') == 'replace_lines']
                 original_failed_op = next((failed_op for failed_op in line_replace_ops if failed_op['filename'] == retry_op['filename']), None)
                 if original_failed_op: 
                     ops_to_apply_this_retry.append(retry_op)

        # Sort the operations by type - we want to process block replacements first as they're more targeted
        ops_to_apply_this_retry.sort(key=lambda x: 0 if x.get('type') == 'replace_block' else 1)

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
                print(f"{Fore.CYAN}  Processing @codebase mention...{Style.RESET_ALL}") # Updated print message
                # Generate the codebase structure tree
                codebase_structure = get_codebase_structure() # Add params if needed

                # --- ADDED: Print the structure for the user --- 
                print_boxed("Codebase Structure Preview", codebase_structure, color=Fore.MAGENTA)
                # --- END ADDED ---

                # Prepare content for the AI prompt (ONLY the structure)
                prepended_content += f"\n{Style.BRIGHT}{Fore.MAGENTA}--- CODEBASE STRUCTURE ---{Style.RESET_ALL}\n"
                prepended_content += f"```\n{codebase_structure}\n```\n" # Inject the structure
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

                print(f"{Fore.CYAN}  Injecting content from: {filepath}{Style.RESET_ALL}") # Removed mention of line numbers

                # Format with line numbers (Now just returns content)
                # numbered_content = _format_content_with_lines(file_content)

                prepended_content += f"\n{Style.BRIGHT}{Fore.MAGENTA}--- MENTIONED FILE: {filepath} ---{Style.RESET_ALL}\n" # Removed (Line Numbered)
                prepended_content += f"```\n{file_content}\n```\n" # Use raw file_content
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
    
    # --- Load Initial and Reminder System Prompts --- Start
    initial_system_prompt = get_system_prompt(is_reminder=False)
    reminder_system_prompt = get_system_prompt(is_reminder=True)
    # --- Load Initial and Reminder System Prompts --- End

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
    # conversation_history = [] # Already initialized above
    # Store context that needs to be prepended *outside* the AI's direct turn
    pending_context_injection = ""

    # --- Add Custom Exception Class --- Start
    class AutoFixRequired(Exception):
        """Custom exception to signal that the auto-fix loop needs to continue."""
        pass
    # --- Add Custom Exception Class --- End

    turn_count = 0 # Keep track of turns for system prompt logic

    while True:
        main_loop_interrupted = False # Flag to indicate if main loop caught interrupt
        try: # Outer try for main loop KeyboardInterrupt
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
                turn_count += 1 # Increment turn count on new user input

            # --- Determine Which System Prompt to Use --- Start
            # Use initial prompt on first turn, reminder prompt otherwise
            active_system_prompt = reminder_system_prompt if turn_count > 1 else initial_system_prompt
            # --- Determine Which System Prompt to Use --- End

            # --- Prepare Prompt for AI ---
            # system_prompt = get_system_prompt() # Replaced by active_system_prompt
            file_context_for_prompt = generate_file_context(file_history)

            # --- Format History for Model ---
            # ... (Existing history formatting logic for Google/OpenRouter) ...
            history_for_model = []
            if provider == "google":
                 combined_history = []
                 temp_history = conversation_history[-10:] # Limit history context
                 for entry in temp_history:
                      role = entry['role']
                      if role in ['user', 'model']:
                          # Combine role and content
                          combined_history.append({"role": role, "parts": [entry['content']]})
                 history_for_model = combined_history 
            elif provider == "openrouter":
                 # Always start with the active system prompt for OpenRouter
                 openai_messages = [{"role": "system", "content": active_system_prompt}]
                 temp_history = conversation_history[-10:] # Limit history context
                 for entry in temp_history:
                      role = entry['role']
                      if role == 'user':
                          openai_messages.append({"role": "user", "content": entry['content']})
                      elif role == 'model':
                           # Map 'model' role from history to 'assistant' for OpenAI API
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
                 # Prepend the system prompt + file context to the current user input/context
                 # Note: Google API prefers system instructions within the user message or history, not a separate system role usually
                 prompt_string_for_google = "\n\n".join([active_system_prompt, file_context_for_prompt, current_context_for_model if current_context_for_model else "Continue."])

                 # Construct content for generate_content API call
                 generation_request_content = history_for_model + [{"role": "user", "parts": [prompt_string_for_google]}] # Include system/file context here
            elif provider == "openrouter":
                 # System prompt is already the first message in history_for_model
                 # Append the file context and current user input/context as the latest user message
                 full_user_content = f"{file_context_for_prompt}\n\n{current_context_for_model if current_context_for_model else 'Continue.'}"
                 history_for_model.append({"role": "user", "content": full_user_content})


            # --- Model Interaction Loop (Multi-Turn) ---
            all_responses_this_turn = []
            is_end_of_turn = False
            ask_for_files_detected = False
            ask_to_user_detected = False
            user_question = None
            files_to_ask_user_for = []

            print(f"\n{Style.DIM}--- CodAgent Thinking ---{Style.RESET_ALL}")

            # --- Inner loop needs adjustment for system prompt injection ---
            # The prompt is constructed *before* this loop now.
            # We might need to reconstruct parts if it continues.

            while not is_end_of_turn and not ask_for_files_detected and not ask_to_user_detected: # Added ask_to_user check
                print(f"\n{Style.BRIGHT}{Fore.GREEN}>>> AI Response Segment {len(all_responses_this_turn) + 1} >>>{Style.RESET_ALL}")
                current_segment_text = ""
                stream_error_occurred = False
                segment_apply_result = None # Reset apply result for this segment
                executed_command_results = [] # Reset command results for this segment

                try:
                    # --- Call Correct API --- Start
                    if provider == "google":
                        # Pass the potentially updated generation_request_content
                        # Ensure generation_request_content is correctly formed for subsequent calls in the loop
                        # If the loop continues, the last model response needs adding, and a "CONTINUE" prompt
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
                         # Ensure history_for_model is correctly formed for subsequent calls in the loop
                         # If the loop continues, the last model response needs adding, and a "CONTINUE" prompt
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
                        # Ensure model response is added to history before breaking
                        if segment_to_log.strip():
                            conversation_history.append({"role": "model", "content": segment_to_log})
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
                        # Ensure model response is added to history before breaking
                        if segment_to_log.strip():
                            conversation_history.append({"role": "model", "content": segment_to_log})
                        break # Break inner loop immediately to handle question

                # 2. Check for [END] tag (Final end-of-turn signal)
                if not stream_error_occurred and not ask_for_files_detected and not ask_to_user_detected:
                     parsed_segment, is_end_from_tag = parse_end_response(segment_for_processing)
                     if is_end_from_tag:
                          segment_for_processing = parsed_segment # Use text without tag for actions
                          segment_to_log = parsed_segment # Log text without the tag
                          is_end_of_turn = True # Mark to terminate loop after this segment's actions

                # Log the AI's segment response (potentially without tags) BEFORE actions
                if segment_to_log.strip() and not stream_error_occurred:
                    all_responses_this_turn.append(segment_to_log)
                    conversation_history.append({"role": "model", "content": segment_to_log})
                elif stream_error_occurred:
                    # Already logged error, don't add failed segment text
                    pass

                # --- Execute File Operations for this Segment ---
                if not stream_error_occurred and not ask_for_files_detected and not ask_to_user_detected: # Added ask_to_user check
                    segment_file_ops = parse_file_operations(segment_for_processing)
                    if segment_file_ops:
                        print("\n" + "="*5 + f" File Operations Proposed (Segment {len(all_responses_this_turn)}) " + "="*5)
                        if preview_changes(segment_file_ops):
                            segment_apply_result = apply_changes(segment_file_ops)
                            # Update system history with results
                            op_summary_lines_segment = [f"File Operations Status (Segment {len(all_responses_this_turn)})]:"]
                            if segment_apply_result.get('successful'): op_summary_lines_segment.append(f"  {Fore.GREEN}Successful ({len(segment_apply_result['successful'])}):{Style.RESET_ALL} {', '.join(list({op['filename'] for op in segment_apply_result['successful']}))}")
                            if segment_apply_result.get('failed'): op_summary_lines_segment.append(f"  {Fore.RED}Failed ({len(segment_apply_result['failed'])}):{Style.RESET_ALL} {', '.join(list({op['filename'] for op in segment_apply_result['failed']}))}")
                            # Don't add to conversation_history if it's empty
                            if len(op_summary_lines_segment) > 1:
                                conversation_history.append({"role": "system", "content": "\n".join(op_summary_lines_segment)})
                            # Update internal file state tracker
                            for op in segment_apply_result.get("successful", []):
                                norm_filename = os.path.normpath(op["filename"])
                                # ... (file_history update logic remains same) ...
                                if op["type"] == "create":
                                    if norm_filename not in file_history["created"]: file_history["created"].append(norm_filename)
                                    if norm_filename not in file_history["current_workspace"]: file_history["current_workspace"].append(norm_filename)
                                    if norm_filename in file_history["modified"]: file_history["modified"].remove(norm_filename)
                                elif op["type"] in ["rewrite", "replace_block"]: # Added rewrite and replace_block
                                    if norm_filename not in file_history["modified"] and norm_filename not in file_history["created"]: file_history["modified"].append(norm_filename)
                                    if norm_filename not in file_history["current_workspace"]: file_history["current_workspace"].append(norm_filename)

                            # --- Initialize list of successful operations for this segment ---
                            # Make a copy initially. This list will be extended if retries are successful.
                            successful_ops_this_segment = segment_apply_result.get("successful", [])[:]

                            # --- Check for and Trigger BLOCK REPLACE Retries ---
                            failed_block_ops = [op for op in segment_apply_result.get("failed", []) if op.get('type') == 'replace_block']
                            if failed_block_ops:
                                print(f"\n{Fore.YELLOW}--- Initiating Auto-Retry for {len(failed_block_ops)} Failed Block Replacements ---{Style.RESET_ALL}")
                                retry_result = retry_failed_replacements(
                                    failed_block_ops,
                                    client_or_model,
                                    provider,
                                    model_name,
                                    file_history,
                                    conversation_history # Pass current history
                                )
                                # Update successful ops list and file history with newly successful retries
                                if retry_result.get('newly_successful'):
                                    # Update file history for newly successful ops first
                                    for op in retry_result['newly_successful']:
                                        norm_filename = os.path.normpath(op["filename"])
                                        if norm_filename not in file_history["modified"] and norm_filename not in file_history["created"]:
                                             file_history["modified"].append(norm_filename)
                                        if norm_filename not in file_history["current_workspace"]:
                                             file_history["current_workspace"].append(norm_filename)
                                    # NOW extend the list used for subsequent steps
                                    successful_ops_this_segment.extend(retry_result['newly_successful'])
                                # Note: retry_failed_replacements logs its own results to history

                            # --- Add Explicit Review Instruction (Uses the potentially updated successful_ops_this_segment list) ---
                            if successful_ops_this_segment:
                                modified_filenames = list({op['filename'] for op in successful_ops_this_segment})
                                review_instruction = f"**SYSTEM CHECK:** Files {', '.join([f'`{f}`' for f in modified_filenames])} were modified. Please carefully review their full content in the `--- FILE CONTEXT ---` above for correctness (syntax and logic) based on the original request before proceeding. If you find errors, provide fixes. If not, continue or use `[END]` if the task is complete."
                                conversation_history.append({"role": "system", "content": review_instruction})

                            # --- Run Syntax Check / Auto-Fix (Uses the potentially updated successful_ops_this_segment list) ---
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
                                    # Use the reminder prompt structure for auto-fix
                                    auto_fix_prompt = f"""{reminder_system_prompt}\n\n{updated_file_context}\n\n{Style.BRIGHT}{Fore.RED}SYSTEM CHECK - SYNTAX ERROR:{Style.RESET_ALL} The following syntax error(s) were detected in the file(s) you just modified:\n\n{error_details}\n\n**Your Task:** Review the errors and the code context above. Provide `======= REPLACE ... END` command(s) to fix **only these specific errors**. Do NOT use `[END]`."""
                                    pending_context_injection = auto_fix_prompt
                                    conversation_history.append({"role": "system", "content": f"Auto-fix check initiated due to syntax errors in: {', '.join(affected_filenames)}"})
                                    raise AutoFixRequired() # Use exception to break inner loop and continue outer
                        else:
                            print(f"{Fore.YELLOW}✗ File operations skipped by user (segment {len(all_responses_this_turn)}).{Style.RESET_ALL}")
                            conversation_history.append({"role": "system", "content": f"User skipped proposed file operations (segment {len(all_responses_this_turn)})."})

                # --- Execute Terminal Commands for this Segment ---
                if not stream_error_occurred and not ask_for_files_detected and not ask_to_user_detected: # Added ask_to_user check
                    segment_terminal_commands = parse_terminal_commands(segment_for_processing)
                    if segment_terminal_commands:
                        # ... (Existing terminal command preview, confirmation, execution logic) ...
                        print("\n" + "="*5 + f" Terminal Commands Proposed (Segment {len(all_responses_this_turn)}) " + "="*5)
                        print_boxed(f"Terminal Commands Preview (Segment {len(all_responses_this_turn)})", "\n".join([f"- {cmd}" for cmd in segment_terminal_commands]), color=Fore.YELLOW)
                        print("-" * 30)
                        confirm_terminal = input(f"{Style.BRIGHT}{Fore.CYAN}Execute these commands? (y/n): {Style.RESET_ALL}").lower().strip()
                        if confirm_terminal.startswith('y'):
                            # Execute commands one by one
                            for command in segment_terminal_commands:
                                # execute_terminal_command now handles live output and its own Ctrl+C
                                result = execute_terminal_command(command)
                                executed_command_results.append({"command": command, "result": result})
                                # Check if the command execution itself was interrupted
                                if result.get("interrupted"):
                                    print(f"{Fore.YELLOW}Command '{command}' was interrupted. Stopping further commands in this segment.{Style.RESET_ALL}")
                                    # Add a note to history about the interruption stopping the sequence
                                    conversation_history.append({"role": "system", "content": f"User interrupted command '{command}'. Subsequent commands in this segment were skipped."})
                                    break # Stop executing commands in this segment

                            # --- Update system history with results --- Start
                            cmd_summary_lines_segment = [f"Terminal Execution Results (Segment {len(all_responses_this_turn)})]:"]
                            ai_logs_for_model = [] # Collect detailed logs for AI

                            for res in executed_command_results:
                                cmd = res['command']
                                cmd_result = res['result']
                                status = ""
                                # Format status based on return code and interrupted flag
                                if cmd_result.get('interrupted'):
                                     status = f"{Fore.YELLOW}⚠ INTERRUPTED (Code: {cmd_result['returncode']}){Style.RESET_ALL}"
                                elif cmd_result['returncode'] == 0:
                                     status = f"{Fore.GREEN}✓ SUCCESS (Code: 0){Style.RESET_ALL}"
                                else:
                                     status = f"{Fore.RED}✗ FAILED (Code: {cmd_result['returncode']}){Style.RESET_ALL}"

                                cmd_summary_lines_segment.append(f"`{cmd}`: {status}")
                                # Only show output/error snippets if not interrupted (live output was already shown)
                                if not cmd_result.get('interrupted'):
                                     if cmd_result['stdout']: cmd_summary_lines_segment.append(f"  Output: {cmd_result['stdout'][:100]}{'...' if len(cmd_result['stdout']) > 100 else ''}")
                                     if cmd_result['stderr']: cmd_summary_lines_segment.append(f"  {Fore.RED}Errors:{Style.RESET_ALL} {cmd_result['stderr'][:100]}{'...' if len(cmd_result['stderr']) > 100 else ''}")

                                # Store the detailed AI log for sending back to the model
                                if 'ai_log' in cmd_result:
                                    ai_logs_for_model.append(cmd_result['ai_log'])

                            # Add summary to conversation history (for system tracking/user visibility)
                            if len(cmd_summary_lines_segment) > 1:
                                conversation_history.append({"role": "system", "content": "\n".join(cmd_summary_lines_segment)})

                            # Add detailed logs back to the AI as user input (so it reacts to them)
                            if ai_logs_for_model:
                                combined_ai_log = "\n\n".join([f"```\n{log}\n```" for log in ai_logs_for_model])
                                conversation_history.append({"role": "user", "content": f"Terminal command output(s):\n{combined_ai_log}"})
                                # Add acknowledgment to confirm receipt
                                conversation_history.append({"role": "model", "content": "Received terminal output(s). Analyzing now."})

                            # --- Update system history with results --- End

                            print(f"{Fore.GREEN}Finished processing segment commands.{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.YELLOW}✗ Terminal commands skipped by user (segment {len(all_responses_this_turn)}).{Style.RESET_ALL}")
                            conversation_history.append({"role": "system", "content": f"User skipped proposed terminal commands (segment {len(all_responses_this_turn)})."})

                # --- Process Tags and Execute Commands PER SEGMENT --- End

                # Check if inner loop should terminate based on END tag
                if is_end_of_turn:
                     print(f"\n{Style.DIM}--- CodAgent Finished Turn ([END] detected in segment {len(all_responses_this_turn)}) ---{Style.RESET_ALL}")
                     break # Break the inner while loop

                # --- Prepare for next segment (if loop wasn't broken by END, ASK, or AutoFix) --- Start
                print(f"\n{Style.DIM}--- CodAgent Continuing (More segments expected) ---{Style.RESET_ALL}")
                # The next model call needs the latest context, including the reminder prompt,
                # file context, conversation history, and the "CONTINUE" instruction.
                # We need to reconstruct the input for the API call.

                # Update history structures before reconstructing the API call input
                # The model's last response (segment_to_log) was already added to conversation_history

                # Re-generate file context as it might have changed due to actions
                file_context_for_prompt = generate_file_context(file_history)

                # Reconstruct the history for the next API call
                if provider == "google":
                    # Rebuild history for Google
                    temp_history = conversation_history[-10:] # Limit history again
                    history_for_google_continue = []
                    for entry in temp_history:
                        role = entry['role']
                        if role in ['user', 'model']:
                            history_for_google_continue.append({"role": role, "parts": [entry['content']]})

                    # Construct the new user prompt string including reminder, files, and "CONTINUE"
                    prompt_string_for_google_continue = "\n\n".join([
                        reminder_system_prompt, # Always use reminder after first turn
                        file_context_for_prompt,
                        "CONTINUE."
                    ])
                    # Update generation_request_content for the *next* iteration
                    generation_request_content = history_for_google_continue + [{"role": "user", "parts": [prompt_string_for_google_continue]}]

                elif provider == "openrouter":
                    # Rebuild history for OpenRouter
                    temp_history = conversation_history[-10:] # Limit history again
                    history_for_openai_continue = [{"role": "system", "content": reminder_system_prompt}] # Start with reminder
                    for entry in temp_history:
                        role = entry['role']
                        if role == 'user':
                            history_for_openai_continue.append({"role": "user", "content": entry['content']})
                        elif role == 'model':
                           history_for_openai_continue.append({"role": "assistant", "content": entry['content']})

                    # Construct the new user prompt string including file context and "CONTINUE"
                    user_content_for_openai_continue = f"{file_context_for_prompt}\n\nCONTINUE."
                    # Update history_for_model for the *next* iteration
                    history_for_model = history_for_openai_continue + [{"role": "user", "content": user_content_for_openai_continue}]
                # --- Prepare for next segment --- End


            # --- End of AI Turn / Inner Segment Loop (`while not is_end_of_turn...`) ---

            # --- Handle ASK_FOR_FILES Interaction (Happens if inner loop broken by ask_for_files_detected) ---
            if ask_for_files_detected:
                print(f"\n{Fore.CYAN}CodAgent is asking for the following files:{Style.RESET_ALL}")
                # ... (Existing user file selection logic - no changes needed here) ...
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
                                            # numbered_content = _format_content_with_lines(file_content) # Removed call
                                            temp_selected_context += f"\n{Fore.CYAN}=== {selected_filepath} ==={Style.RESET_ALL}\n" # Removed (Line Numbered)
                                            temp_selected_context += f"```\n{file_content}\n```\n" # Use raw file_content
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
                user_response = ""
                response_to_log = ""

                # Display the question with appropriate formatting
                if question_format == "normal":
                    question_text = user_question["question"]
                    print(f"\n{Fore.GREEN}{question_text}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Enter your response:{Style.RESET_ALL}")
                    user_response = input(f"{Style.BRIGHT}{Fore.CYAN}Response: {Style.RESET_ALL}").strip()
                    response_to_log = f"[Response to question] {user_response}" if user_response else "[No response provided to question]"

                elif question_format == "options":
                    # Handle options format using the options list
                    options = user_question["options"]
                    print(f"\n{Fore.GREEN}Please select an option by number:{Style.RESET_ALL}") # Clarified prompt
                    for i, option in enumerate(options, 1):
                        print(f"{Fore.GREEN}{i}.{Style.RESET_ALL} {option}")
                    print(f"{Fore.YELLOW}Enter your selection number:{Style.RESET_ALL}") # Clarified prompt
                    user_response_num_str = input(f"{Style.BRIGHT}{Fore.CYAN}Selection: {Style.RESET_ALL}").strip()

                    # --- New Logic: Convert number to text ---
                    try:
                        selected_index = int(user_response_num_str) - 1
                        if 0 <= selected_index < len(options):
                            selected_option_text = options[selected_index]
                            # Format response for AI with both number and text
                            response_to_log = f"User selected option {selected_index + 1}: '{selected_option_text}'"
                            print(f"{Style.DIM}Processing selection: {response_to_log}{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}Invalid selection number. Asking AI to clarify.{Style.RESET_ALL}")
                            response_to_log = f"[User provided invalid option number: {user_response_num_str}. Please clarify selection.]"
                    except ValueError:
                        print(f"{Fore.RED}Invalid input (not a number). Asking AI to clarify.{Style.RESET_ALL}")
                        response_to_log = f"[User provided non-numeric input: '{user_response_num_str}'. Please clarify selection.]"
                    # --- End New Logic ---

                elif question_format == "yesno":
                    question_text = user_question["question"]
                    print(f"\n{Fore.GREEN}{question_text} (yes/no){Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Enter your response:{Style.RESET_ALL}")
                    user_response = input(f"{Style.BRIGHT}{Fore.CYAN}Response: {Style.RESET_ALL}").strip()
                    response_to_log = f"[Response to yes/no question] {user_response}" if user_response else "[No response provided to question]"

                # Add the user's response (or formatted selection) to the conversation history
                conversation_history.append({"role": "user", "content": response_to_log})

                # Store the response to be used as input for the AI's continuation
                pending_context_injection = response_to_log
                continue # Continue the main loop immediately with the user's response

            # --- Auto-Fix was Triggered ---
            # Handled by catching AutoFixRequired exception below

            # --- Redundant code blocks after the loop are now removed/commented ---

        except AutoFixRequired: # Catch the custom exception
             print(f"{Fore.CYAN}--- Auto-fix required, continuing loop ---{Style.RESET_ALL}")
             # Make sure turn_count doesn't increment again if auto-fix triggers immediately
             continue # Continue the main `while True` loop immediately
        except KeyboardInterrupt: # Catch Ctrl+C ONLY if it happens *outside* execute_terminal_command
            print(f"\n{Fore.YELLOW}Interrupt received outside command execution. Exiting...{Style.RESET_ALL}")
            main_loop_interrupted = True # Set flag to break outer loop
        except Exception as e:
            print(f"\n{Back.RED}{Fore.WHITE} UNEXPECTED ERROR: {e} {Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            conversation_history.append({"role": "system", "content": f"An unexpected error occurred: {e}\n{traceback.format_exc()}"})

        if main_loop_interrupted:
             break # Break the main while loop if flag is set

    # ... (end of chat_with_model) ...

def generate_diff_report(file_lines, ai_old_code_lines, best_match_start_line):
    """Generates a diff-like report comparing AI's old code and actual file lines."""
    report = []
    file_slice = file_lines[best_match_start_line : best_match_start_line + len(ai_old_code_lines)]
    diff = difflib.ndiff(file_slice, ai_old_code_lines)
    
    report.append("--- Diff Report (File vs. Your Attempted Old Code) ---")
    for i, line in enumerate(diff):
        file_line_num = best_match_start_line + i + 1
        prefix = line[:2]
        content = line[2:]
        if prefix == '+ ': # Lines only in AI's code (shouldn't happen if copied?)
            report.append(f"AI Only (?): {repr(content)}") 
        elif prefix == '- ': # Lines only in File code
            report.append(f"File (L{file_line_num}):  {repr(content)}")
        elif prefix == '  ': # Lines matching
            report.append(f"Match (L{file_line_num}): {repr(content)}")
        elif prefix == '? ': # Difference hints
            # report.append(f"Hint:         {content}") # Optional: Show hints 
            pass
    report.append("------------------------------------------------------")
    return "\n".join(report)

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
    # Add version argument
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    # --- Initialize Model ---
    # Pass all args to the initializer
    client_or_model, provider, model_name_used = initialize_model(args)

    # Start chat with the initialized model
    chat_with_model(client_or_model, provider, model_name_used)


if __name__ == "__main__":
    main() 
