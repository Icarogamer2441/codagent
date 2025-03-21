# CodAgent

A command-line tool for code generation and editing using Google's Gemini models.

## Installation

```bash
pip install .
```

## Prerequisites

You need a Google API key with access to the Gemini models. Set it as an environment variable:

### Linux/Mac
```bash
export GOOGLE_API_KEY=your_api_key_here
```

### Windows
```cmd
set GOOGLE_API_KEY=your_api_key_here
```

## Usage

```bash
coda --model gemini-pro
```

## Features

- Interactive chat with Google's Gemini models
- File generation and editing through chat
- Preview changes before applying them
- Colored diff view for file changes