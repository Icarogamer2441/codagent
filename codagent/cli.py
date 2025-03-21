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

# Initialize colorama
colorama.init()

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
⚠️ ⚠️ ⚠️ CRITICAL INSTRUCTION - READ CAREFULLY ⚠️ ⚠️ ⚠️

You are CodAgent, an advanced coding assistant designed to help with programming tasks.
You can generate and modify code based on user requests using special syntax for file operations.

WORKING DIRECTORY: {current_dir}

⚠️ EXTREMELY IMPORTANT - CODE EXACTNESS REQUIREMENT ⚠️
When generating file operations, you MUST:
1. PRESERVE EXACT INDENTATION of the original code
2. MATCH WHITESPACE PRECISELY (spaces, tabs, blank lines)
3. INCLUDE EXACT SYNTAX (brackets, semicolons, etc.)
4. COPY VARIABLE/FUNCTION NAMES with perfect spelling
5. NEVER strip indentation from code blocks

DO NOT MODIFY INDENTATION IN THE [REPLACE] BLOCKS! Keep it EXACTLY as it appears in the file!
Copy/paste the exact content from the file context provided below.

If you fail to match code exactly, the operations will fail. I will keep sending the request 
until you get it 100% right. Take your time and verify your code matches EXACTLY.

FILE OPERATIONS SYNTAX:

1. To CREATE a new file:
[CREATE=filename.ext]
// Your code or content here
[/CREATE]

Example:
[CREATE=hello.py]
def greet(name):
    return f"Hello, {{name}}!"

if __name__ == "__main__":
    print(greet("World"))
[/CREATE]

2. To REPLACE content in an existing file:
[REPLACE=filename.ext]
// Exact content that will be replaced WITH INDENTATION
[TO]
// New content that will replace the old content
[/REPLACE]

Example with preserved indentation:
[REPLACE=app.js]
    console.log("Hello World");
[TO]
    console.log("Hello, CodAgent!");
[/REPLACE]

Notice that the indentation (spaces before the console.log line) is preserved in both the old and new content!

IMPORTANT GUIDELINES:

- Be precise and explicit about file paths and content
- Always provide complete, functional code
- For replacements, include enough context to uniquely identify the location
- Explain your changes and reasoning clearly
- When appropriate, suggest multiple alternatives
- Always follow best practices for the relevant programming language
- If necessary, provide step-by-step instructions for the user to follow after applying changes
- Remember that the user will see all changes before they're applied
- If you're uncertain about something, ask for clarification

INDENTATION HANDLING - CRITICAL INFORMATION:

1. IMPORTANCE OF INDENTATION:
   - In languages like Python, indentation is SYNTACTICALLY SIGNIFICANT and defines code blocks
   - Even in other languages, proper indentation is essential for readability and maintainability
   - CodAgent has intelligent indentation detection but needs your help to work effectively

2. HOW TO PROPERLY HANDLE INDENTATION:
   - When identifying code to replace, INCLUDE THE EXACT INDENTATION from the original file
   - If you don't know the exact indentation, use the non-indented version of the code and let CodAgent handle it
   - For multi-line replacements, ensure ALL LINES maintain consistent indentation relative to each other
   - When adding new code blocks, follow the indentation style of surrounding code

3. COMMON INDENTATION PATTERNS:
   - Python: 4 spaces per level (PEP 8 standard)
   - JavaScript/TypeScript: 2 or 4 spaces, or tabs
   - HTML/XML: 2 or 4 spaces, nested elements indented one level deeper
   - CSS: 2 or 4 spaces, properties indented within selectors

4. TIPS FOR SUCCESSFUL REPLACEMENTS:
   - Include unique identifiers in your search patterns (variable names, function signatures, comments)
   - For class/function definitions, include the entire signature line
   - If replacing code inside a code block, include enough context (at least the first line of the block)
   - When uncertain, include more context rather than less

EXAMPLES OF INDENTATION-AWARE REPLACEMENTS:

GOOD (includes exact indentation):
[REPLACE=main.py]
    def calculate_total(items):
        total = 0
        for item in items:
            total += item.price
        return total
[TO]
    def calculate_total(items):
        return sum(item.price for item in items)
[/REPLACE]

GOOD (indentation-free for CodAgent to handle):
[REPLACE=utils.py]
def parse_data(input_string):
    parts = input_string.split(',')
    return parts
[TO]
def parse_data(input_string):
    parts = input_string.split(',')
    return [part.strip() for part in parts]
[/REPLACE]

BAD (mismatched indentation):
[REPLACE=app.py]
def process():
    data = load_data()
    result = transform(data)
    return result
[TO]
def process():
data = load_data()
result = transform(data)
return result
[/REPLACE]

USER WORKFLOW:
1. User describes what they need
2. You respond with explanations and file operations enclosed in the special syntax
3. CodAgent will show the user a preview of changes with color-coded diffs
4. User confirms or rejects the changes

Your primary goal is to help the user accomplish their coding tasks efficiently and correctly.

Remember: With proper indentation in your file operations, you'll provide a better and more reliable experience to the user.

NEVER USE ``` in your response.
"""
    return system_prompt

def parse_file_operations(response_text):
    """Parse the response text to extract file operations."""
    file_operations = []
    
    # Find CREATE operations
    create_pattern = r"\[CREATE=([^\]]+)\](.*?)\[/CREATE\]"
    for match in re.finditer(create_pattern, response_text, re.DOTALL):
        filename = match.group(1).strip()
        content = match.group(2)  # Do NOT strip content for creation
        file_operations.append({
            "type": "create",
            "filename": filename,
            "content": content
        })
    
    # Find REPLACE operations
    replace_pattern = r"\[REPLACE=([^\]]+)\](.*?)\[TO\](.*?)\[/REPLACE\]"
    for match in re.finditer(replace_pattern, response_text, re.DOTALL):
        filename = match.group(1).strip()
        old_content = match.group(2)  # DO NOT STRIP - preserve exact whitespace
        new_content = match.group(3)  # DO NOT STRIP - preserve exact whitespace
        file_operations.append({
            "type": "replace",
            "filename": filename,
            "old_content": old_content,
            "new_content": new_content
        })
    
    return file_operations

def show_diff(old_lines, new_lines):
    """Show colored diff between old and new content."""
    import difflib
    
    diff = difflib.ndiff(old_lines, new_lines)
    for line in diff:
        if line.startswith('+ '):
            print(f"{Back.GREEN}{line[2:]}{Style.RESET_ALL}")
        elif line.startswith('- '):
            print(f"{Back.RED}{line[2:]}{Style.RESET_ALL}")
        elif line.startswith('? '):
            continue
        else:
            print(line[2:])

def preview_changes(file_operations):
    """Preview changes to be made to files."""
    print(f"\n{Fore.CYAN}=== File Operations Preview ==={Style.RESET_ALL}\n")
    
    for op in file_operations:
        if op["type"] == "create":
            print(f"{Fore.GREEN}CREATE{Style.RESET_ALL} {op['filename']}")
            print(f"{Fore.YELLOW}Content:{Style.RESET_ALL}")
            for line in op["content"].splitlines()[:5]:
                print(f"{Back.GREEN}{line}{Style.RESET_ALL}")
            if len(op["content"].splitlines()) > 5:
                print("...")
            print()
        
        elif op["type"] == "replace":
            print(f"{Fore.YELLOW}REPLACE{Style.RESET_ALL} in {op['filename']}")
            
            # Show diff - DO NOT STRIP WHITESPACE here to show exact indentation
            print(f"{Fore.YELLOW}Diff:{Style.RESET_ALL}")
            show_diff(op["old_content"].splitlines(), op["new_content"].splitlines())
            print()
    
    return input(f"{Fore.CYAN}Apply these changes? (y/n): {Style.RESET_ALL}").lower().startswith('y')

def apply_changes(file_operations):
    """Apply the file operations."""
    # Keep track of failed and successful operations
    failed_ops = []
    successful_ops = []
    
    for op in file_operations:
        if op["type"] == "create":
            try:
                # Create directory if it doesn't exist
                directory = os.path.dirname(op["filename"])
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                with open(op["filename"], "w") as f:
                    f.write(op["content"])
                print(f"{Fore.GREEN}✓ SUCCESS: Created{Style.RESET_ALL} {op['filename']}")
                successful_ops.append(op)
            except Exception as e:
                print(f"{Fore.RED}× FAILED: Could not create {op['filename']}: {e}{Style.RESET_ALL}")
                failed_ops.append(op)
        
        elif op["type"] == "replace":
            try:
                with open(op["filename"], "r") as f:
                    content = f.read()
                
                # Check for exact match first
                if op["old_content"] in content:
                    new_content = content.replace(op["old_content"], op["new_content"])
                    with open(op["filename"], "w") as f:
                        f.write(new_content)
                    print(f"{Fore.GREEN}✓ SUCCESS: Updated{Style.RESET_ALL} {op['filename']} with exact match")
                    successful_ops.append(op)
                else:
                    # Try normalized whitespace check
                    normalized_old = "\n".join([line.rstrip() for line in op["old_content"].splitlines()])
                    normalized_content = "\n".join([line.rstrip() for line in content.splitlines()])
                    
                    if normalized_old in normalized_content:
                        print(f"{Fore.GREEN}✓ SUCCESS: Found match with normalized whitespace in {op['filename']}{Style.RESET_ALL}")
                        new_content = normalized_content.replace(normalized_old, "\n".join([line.rstrip() for line in op["new_content"].splitlines()]))
                        with open(op["filename"], "w") as f:
                            f.write(new_content)
                        print(f"{Fore.GREEN}✓ SUCCESS: Updated{Style.RESET_ALL} {op['filename']} with normalized whitespace")
                        successful_ops.append(op)
                    else:
                        # Try indentation-aware replacement as a last resort
                        print(f"{Fore.YELLOW}Exact match not found. Trying indentation-aware replacement...{Style.RESET_ALL}")
                        success = handle_indentation_mismatch(op["filename"], op["old_content"], op["new_content"])
                        
                        if success:
                            print(f"{Fore.GREEN}✓ SUCCESS: Applied indentation-aware replacement in {op['filename']}{Style.RESET_ALL}")
                            successful_ops.append(op)
                        else:
                            print(f"{Fore.RED}× FAILED: Could not find matching content in {op['filename']}.{Style.RESET_ALL}")
                            failed_ops.append(op)
            except FileNotFoundError:
                print(f"{Fore.RED}× FAILED: File {op['filename']} not found.{Style.RESET_ALL}")
                failed_ops.append(op)
    
    # Return info about operations so we know which need retrying
    return {"successful": successful_ops, "failed": failed_ops}

def handle_indentation_mismatch(filename, old_content, new_content):
    """Handle replacements with indentation mismatches."""
    try:
        with open(filename, "r") as f:
            file_lines = f.readlines()
            file_content = ''.join(file_lines)
        
        # Remove leading whitespace from each line of the old content to create a pattern
        old_lines = old_content.splitlines()
        stripped_old_lines = [line.lstrip() for line in old_lines if line.strip()]
        
        # If the content is only one line, handle it specially
        if len(stripped_old_lines) == 1:
            return handle_single_line_replacement(filename, old_content, new_content, file_lines)
        
        # If we have multiple non-empty lines, try a more sophisticated matching
        return handle_multiline_replacement(filename, stripped_old_lines, new_content, file_lines)
    
    except Exception as e:
        print(f"{Fore.RED}Error during indentation-aware replacement: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return False

def handle_multiline_replacement(filename, stripped_old_lines, new_content, file_lines):
    """Handle more complex multiline replacements with flexible indentation matching."""
    # For debugging
    print(f"{Fore.CYAN}Searching for {len(stripped_old_lines)} lines of content...{Style.RESET_ALL}")
    
    # First, find all potential starting points (lines that match the first stripped line)
    potential_starts = []
    for i, line in enumerate(file_lines):
        stripped_line = line.lstrip()
        if stripped_line and stripped_line == stripped_old_lines[0]:
            potential_starts.append(i)
    
    if not potential_starts:
        print(f"{Fore.RED}× FAILED: Could not find any matches for the first line.{Style.RESET_ALL}")
        return False
    
    print(f"{Fore.CYAN}Found {len(potential_starts)} potential starting points.{Style.RESET_ALL}")
    
    # Try each potential starting point
    for start_idx in potential_starts:
        match_found = True
        indentation_patterns = []
        matching_lines = []
        
        # Check if this starting point leads to a full match
        for j, pattern_line in enumerate(stripped_old_lines):
            if start_idx + j >= len(file_lines):
                match_found = False
                break
                
            file_line = file_lines[start_idx + j]
            stripped_file_line = file_line.lstrip()
            
            # Skip empty lines in the file when matching
            if not stripped_file_line and j < len(stripped_old_lines) - 1:
                # Adjust for skipped lines
                start_idx -= 1
                continue
                
            if stripped_file_line == pattern_line:
                # Store the indentation for this line
                indentation = file_line[:len(file_line) - len(stripped_file_line)]
                indentation_patterns.append(indentation)
                matching_lines.append(start_idx + j)
            else:
                match_found = False
                break
        
        if match_found and len(matching_lines) == len(stripped_old_lines):
            print(f"{Fore.GREEN}✓ SUCCESS: Found matching content at lines {matching_lines[0]+1}-{matching_lines[-1]+1}{Style.RESET_ALL}")
            
            # Determine the base indentation (from first line)
            base_indentation = indentation_patterns[0]
            
            # Apply the replacement with proper indentation
            new_lines = []
            for new_line in new_content.splitlines():
                if new_line.strip():
                    new_lines.append(f"{base_indentation}{new_line.lstrip()}\n")
                else:
                    new_lines.append("\n")  # Empty line
            
            # Replace the lines in the file
            file_lines[matching_lines[0]:matching_lines[-1] + 1] = new_lines
            
            # Write back to the file
            with open(filename, "w") as f:
                f.writelines(file_lines)
            
            print(f"{Fore.GREEN}✓ SUCCESS: Replaced content with indentation preserved.{Style.RESET_ALL}")
            return True
    
    # If we get here, we couldn't find a complete match
    print(f"{Fore.RED}× FAILED: Could not find a complete match for the content.{Style.RESET_ALL}")
    
    # As a fallback, try a fuzzy match (ignoring indentation and spacing)
    return attempt_fuzzy_match(filename, stripped_old_lines, new_content, file_lines)

def attempt_fuzzy_match(filename, stripped_old_lines, new_content, file_lines):
    """Attempt a fuzzy match by focusing on content rather than exact whitespace."""
    # Normalize content for fuzzy matching 
    normalized_pattern = '\n'.join([line.strip() for line in stripped_old_lines if line.strip()])
    
    # Prepare sliding window to find the best match
    window_size = len(stripped_old_lines)
    best_match_score = 0
    best_match_idx = -1
    
    # Look for the best matching region in the file
    for i in range(len(file_lines) - window_size + 1):
        window_content = ''.join(file_lines[i:i+window_size])
        normalized_window = '\n'.join([line.strip() for line in window_content.splitlines() if line.strip()])
        
        # Calculate a simple similarity score (can be improved with difflib or other algorithms)
        import difflib
        similarity = difflib.SequenceMatcher(None, normalized_pattern, normalized_window).ratio()
        
        if similarity > best_match_score and similarity > 0.7:  # 70% threshold
            best_match_score = similarity
            best_match_idx = i
    
    if best_match_idx >= 0:
        print(f"{Fore.YELLOW}Found fuzzy match with {best_match_score*100:.1f}% similarity at line {best_match_idx+1}.{Style.RESET_ALL}")
        
        # Determine indentation from the first line of the match
        first_line = file_lines[best_match_idx]
        indentation = first_line[:len(first_line) - len(first_line.lstrip())]
        
        # Apply indentation to new content
        new_lines = []
        for new_line in new_content.splitlines():
            if new_line.strip():
                new_lines.append(f"{indentation}{new_line.lstrip()}\n")
            else:
                new_lines.append("\n")
        
        # Replace the lines
        file_lines[best_match_idx:best_match_idx + window_size] = new_lines
        
        # Write back to the file
        with open(filename, "w") as f:
            f.writelines(file_lines)
        
        print(f"{Fore.GREEN}Applied fuzzy matching replacement.{Style.RESET_ALL}")
        return True
    
    return False

def handle_single_line_replacement(filename, old_content, new_content, file_lines):
    """Handle single line replacements with indentation preservation."""
    old_stripped = old_content.strip()
    success = False
    
    print(f"{Fore.CYAN}Looking for single-line match: '{old_stripped}'{Style.RESET_ALL}")
    
    # Try exact content match first
    for i, line in enumerate(file_lines):
        stripped_line = line.strip()
        if stripped_line == old_stripped:
            # Found match - preserve the indentation
            indentation = line[:len(line) - len(stripped_line)]
            
            # Apply indentation to new content (potentially multi-line)
            new_lines = []
            for new_line in new_content.splitlines():
                if new_line.strip():
                    new_lines.append(f"{indentation}{new_line.lstrip()}\n")
                else:
                    new_lines.append("\n")
            
            # Replace the line in the file
            file_lines[i:i+1] = new_lines
            
            # Write back to the file
            with open(filename, "w") as f:
                f.writelines(file_lines)
            
            print(f"{Fore.GREEN}Successfully replaced line with indentation preserved at line {i+1}.{Style.RESET_ALL}")
            success = True
            break
    
    # If no exact match, try fuzzy matching for the single line
    if not success:
        print(f"{Fore.YELLOW}No exact match found. Trying fuzzy match...{Style.RESET_ALL}")
        
        best_match_score = 0
        best_match_idx = -1
        
        for i, line in enumerate(file_lines):
            stripped_line = line.strip()
            if not stripped_line:
                continue
                
            import difflib
            similarity = difflib.SequenceMatcher(None, old_stripped, stripped_line).ratio()
            
            if similarity > best_match_score and similarity > 0.7:
                best_match_score = similarity
                best_match_idx = i
        
        if best_match_idx >= 0:
            # Found a fuzzy match
            line = file_lines[best_match_idx]
            indentation = line[:len(line) - len(line.lstrip())]
            
            # Apply indentation
            new_lines = []
            for new_line in new_content.splitlines():
                if new_line.strip():
                    new_lines.append(f"{indentation}{new_line.lstrip()}\n")
                else:
                    new_lines.append("\n")
            
            # Replace the line
            file_lines[best_match_idx:best_match_idx+1] = new_lines
            
            # Write back
            with open(filename, "w") as f:
                f.writelines(file_lines)
            
            print(f"{Fore.GREEN}Applied fuzzy single-line replacement at line {best_match_idx+1} with {best_match_score*100:.1f}% similarity.{Style.RESET_ALL}")
            success = True
    
    return success

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
    context = "FILE CONTEXT - IMPORTANT FOR CONTINUITY:\n"
    
    # Add information about files created in this session
    if file_history["created"]:
        context += "\nFiles you have CREATED in this session:\n"
        for file in file_history["created"]:
            context += f"- {file}\n"
    
    # Add information about files modified in this session
    if file_history["modified"]:
        context += "\nFiles you have MODIFIED in this session:\n"
        for file in file_history["modified"]:
            context += f"- {file}\n"
    
    if not file_history["created"] and not file_history["modified"]:
        context += "\nNo files have been created or modified yet in this session.\n"
    
    # Add CRITICAL warning about using correct filenames
    context += "\n⚠️ CRITICAL: When referring to files, ALWAYS use the exact filenames from above if they exist.\n"
    context += "NEVER invent new filenames or refer to files that aren't in the list unless creating new ones.\n\n"
    
    # Add the CURRENT CONTENT of all created/modified files to provide accurate context
    context += "CURRENT FILE CONTENTS:\n"
    all_files = set(file_history["created"] + file_history["modified"])
    
    for filename in all_files:
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                context += f"\n--- {filename} ---\n```\n{content}\n```\n"
            else:
                context += f"\n--- {filename} --- [FILE NO LONGER EXISTS]\n"
        except Exception as e:
            context += f"\n--- {filename} --- [ERROR READING FILE: {str(e)}]\n"
    
    return context

def retry_replacements(file_operations, model, user_query, file_history):
    """Retry failed replacements with more context to the model."""
    failed_operations = []
    
    for op in file_operations:
        if op["type"] == "replace":
            try:
                with open(op["filename"], "r") as f:
                    content = f.read()
                
                # Re-check: if the exact content is present, then it's already successful
                if op["old_content"] in content:
                    print(f"{Fore.GREEN}✓ SUCCESS: Operation for {op['filename']} is correct.{Style.RESET_ALL}")
                    continue
                    
                normalized_old = "\n".join([line.rstrip() for line in op["old_content"].splitlines()])
                normalized_content = "\n".join([line.rstrip() for line in content.splitlines()])
                
                if normalized_old in normalized_content:
                    print(f"{Fore.GREEN}✓ SUCCESS: Operation for {op['filename']} checks out with normalized whitespace.{Style.RESET_ALL}")
                    continue
                
                failed_operations.append(op)
            except FileNotFoundError:
                continue
    
    if not failed_operations:
        if hasattr(retry_replacements, 'attempt_count'):
            retry_replacements.attempt_count = 0
        return  # Nothing to retry
    
    print(f"\n{Fore.YELLOW}Some replacements still failed. Asking the model to correct them...{Style.RESET_ALL}")
    
    if hasattr(retry_replacements, 'attempt_count'):
        retry_replacements.attempt_count += 1
    else:
        retry_replacements.attempt_count = 1
        
    if retry_replacements.attempt_count > 3:
        print(f"{Fore.RED}Maximum retry attempts reached. Please check the files manually or try a different approach.{Style.RESET_ALL}")
        retry_replacements.attempt_count = 0
        return
    
    # Build the retry message with file context, etc.
    retry_message = f"⚠️ Attempt {retry_replacements.attempt_count}/3: The following REPLACE operations did not match exactly. Please correct them using EXACT formatting (including indentation and whitespace):\n\n"
    
    for op in failed_operations:
        retry_message += f"In file {op['filename']} the following pattern was not found exactly:\n"
        retry_message += f"```\n{op['old_content']}\n```\n\n"
    
    retry_message += "Please provide updated REPLACE operations with EXACT matching patterns."
    
    system_prompt = get_system_prompt()
    file_context = generate_file_context(file_history)
    
    final_message = f"{system_prompt}\n\n{file_context}\n\nUSER QUERY: {user_query}\n\n{retry_message}"
    
    with tqdm(total=100, desc="Re-thinking", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        for i in range(10):
            time.sleep(0.1)
            pbar.update(10)
        response = model.generate_content(final_message)
        pbar.update(100 - pbar.n)
    
    print(f"\n{Fore.GREEN}=== Corrections ==={Style.RESET_ALL}")
    print(response.text)
    
    new_file_operations = parse_file_operations(response.text)
    
    if new_file_operations:
        if preview_changes(new_file_operations):
            for op in new_file_operations:
                if op["type"] == "create":
                    if op["filename"] not in file_history["created"]:
                        file_history["created"].append(op["filename"])
                elif op["type"] == "replace":
                    if (op["filename"] not in file_history["modified"]) and (op["filename"] not in file_history["created"]):
                        file_history["modified"].append(op["filename"])
            
            apply_changes(new_file_operations)
            retry_replacements(new_file_operations, model, user_query, file_history)

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
    
    # Use plain prompt text without colorama codes
    prompt_text = ">>> "
    
    # Custom styling for the prompt using prompt_toolkit's native styling
    style = PromptStyle.from_dict({
        'prompt': 'ansicyan bold',
    })
    
    print(f"\n{Fore.CYAN}=== CodAgent Chat ===\n{Style.RESET_ALL}")
    print(f"{Fore.GREEN}You are now chatting with the model. Type 'exit' to quit.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Current directory: {os.getcwd()}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Chat history will be saved to: {history_file}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Commands:{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}/add <file_or_dir>{Style.RESET_ALL} - Add file content or directory files to the chat")
    print(f"  {Fore.CYAN}/remove <file_or_dir>{Style.RESET_ALL} - Remove file or directory from the context")
    print(f"  {Fore.CYAN}/list{Style.RESET_ALL} - List all files and directories in the current context")
    print(f"  {Fore.CYAN}/files{Style.RESET_ALL} - List all files created or modified in this session")
    print(f"  {Fore.CYAN}/refresh{Style.RESET_ALL} - Refresh the model's awareness of all file contents")
    print(f"  {Fore.CYAN}/help{Style.RESET_ALL} - Show this help message")
    print()
    
    # Keep track of conversation to maintain context
    conversation_history = []
    
    while True:
        try:
            # Use plain prompt text without colorama formatting
            user_input = prompt(
                prompt_text,
                history=FileHistory(history_file),
                auto_suggest=AutoSuggestFromHistory(),
                style=style
            )
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            # Process commands
            if user_input.startswith('/add '):
                target = user_input[5:].strip()
                content = process_add_command(target)
                if content:
                    # Track the added file/directory for context management
                    added_context[target] = content
                    user_input = f"Here's the content of {target}:\n\n{content}\n\nPlease help me with this."
                    print(f"{Fore.GREEN}Added content of {target} to the conversation.{Style.RESET_ALL}")
                else:
                    continue
            
            elif user_input.startswith('/remove '):
                target = user_input[8:].strip()
                if target in added_context:
                    del added_context[target]
                    print(f"{Fore.YELLOW}Removed {target} from the conversation context.{Style.RESET_ALL}")
                    # Let the model know what's been removed
                    user_input = f"I've removed the file/directory '{target}' from our conversation context. Please disregard it in future responses."
                else:
                    print(f"{Fore.RED}Error: {target} was not found in the current context.{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Current context includes: {', '.join(added_context.keys()) if added_context else 'nothing'}{Style.RESET_ALL}")
                    continue
            
            elif user_input.startswith('/list'):
                if added_context:
                    print(f"\n{Fore.CYAN}Files/directories in current context:{Style.RESET_ALL}")
                    for idx, item in enumerate(added_context.keys(), 1):
                        print(f"{Fore.GREEN}{idx}.{Style.RESET_ALL} {item}")
                    print()
                else:
                    print(f"\n{Fore.YELLOW}No files or directories in current context.{Style.RESET_ALL}\n")
                continue
                
            elif user_input.startswith('/files'):
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
            
            elif user_input.startswith('/refresh'):
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
                user_input = "I've refreshed your awareness of all file contents. Please use the current file contents for any operations."
                print(f"\n{Fore.GREEN}All files refreshed in model's context.{Style.RESET_ALL}\n")
            
            elif user_input.startswith('/help'):
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
            
            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Generate the file context to include with the system prompt
            file_context = generate_file_context(file_history)
            
            # Get the system prompt and combine with file context and user query
            system_prompt = get_system_prompt()
            
            # Create a context-aware prompt with file history
            final_prompt = f"{system_prompt}\n\n{file_context}\n\nCONVERSATION HISTORY:\n"
            
            # Add the last 5 exchanges from conversation history (or fewer if less available)
            history_to_include = min(5, len(conversation_history))
            for i in range(len(conversation_history) - history_to_include, len(conversation_history)):
                entry = conversation_history[i]
                final_prompt += f"{entry['role'].upper()}: {entry['content']}\n\n"
            
            final_prompt += f"USER QUERY: {user_input}"
            
            # Show progress bar while waiting for response
            with tqdm(total=100, desc="Thinking", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
                for i in range(10):
                    time.sleep(0.1)
                    pbar.update(10)
                
                # Send message with system prompt for every query
                response = model.generate_content(final_prompt)
                response_text = response.text
                
                # Complete the progress bar
                pbar.update(100 - pbar.n)
            
            # Add model response to conversation history
            conversation_history.append({"role": "model", "content": response_text})
            
            print(f"\n{Fore.GREEN}=== Response ==={Style.RESET_ALL}")
            print(response_text)
            
            # Parse and process file operations
            file_operations = parse_file_operations(response_text)
            if file_operations:
                if preview_changes(file_operations):
                    # Update file history based on operations
                    for op in file_operations:
                        if op["type"] == "create":
                            if op["filename"] not in file_history["created"]:
                                file_history["created"].append(op["filename"])
                        elif op["type"] == "replace":
                            if op["filename"] not in file_history["modified"] and op["filename"] not in file_history["created"]:
                                file_history["modified"].append(op["filename"])
                    
                    # Apply changes and get results
                    result = apply_changes(file_operations)
                    
                    # Only retry if there were actual failures
                    if result["failed"]:
                        # Try to retry any failed replacements
                        retry_replacements(result["failed"], model, user_input, file_history)
                    else:
                        print(f"{Fore.GREEN}All operations completed successfully!{Style.RESET_ALL}")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            print(f"{Fore.RED}Traceback: {sys.exc_info()}{Style.RESET_ALL}")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="CodAgent - AI-powered code generation tool")
    parser.add_argument("--model", default="gemini-1.5-pro", help="Google Gemini model to use")
    
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}Initializing CodAgent with model: {Fore.GREEN}{args.model}{Style.RESET_ALL}")
    model = initialize_genai(args.model)
    chat_with_model(model)

if __name__ == "__main__":
    main() 