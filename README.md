# PDF to Markdown Converter

A Python script that converts PDF documents to Markdown format using an OpenAI compatible api endpoint. It features intelligent image extraction, OCR post-processing, and configurable settings.

As a side note, I would suggest to everyone to use [datalab-to/marker](https://github.com/datalab-to/marker) first. It is far superior to this project. I made this script because for some documents, Marker can produce long repeated sections of text makeing the MD output unusable. I use this script to convert these files for now.

---

## Features

- **PDF Parsing**: Extracts text and images from PDF files using PyMuPDF.
- **LLM-Powered Conversion**: Leverages LLMs to convert raw text into clean, formatted Markdown.
- **Smart Text Reflow**: Detects column layouts and reflows text into standard paragraph formats automatically.
- **Heading Deduplication**: Prevents duplicate headings across multiple pages by converting subsequent occurrences to bold text.
- **Intelligent Image Handling**:
  - Saves relevant images (charts, graphs, diagrams) to an `images/` subdirectory.
  - Uses geometric and pixel-density heuristics to filter out irrelevant images (headers, footers, icons).
  - **Vision Model Integration**: Uses a multimodal LLM to classify and filter images (rejecting TOC, indexes, simple logos).
- **Token Management**: Calculates token counts to ensure text chunks fit within the LLM's context window.
- **Flexible Input**: Processes single PDF files or entire directories recursively.
- **Configurable**: Fully customizable via `config.json` and command-line arguments.

---

## Requirements


Install the required libraries:

```bash
pip install openai tiktoken pymupdf tqdm
```

## Installation

1. **Clone or Download** the repository/script.
2. **Install Dependencies**:
   ```bash
   pip install openai tiktoken pymupdf tqdm
   ```
3. **Configure**: Create a `config.json` file (see configuration section below).
4. **Run**: Execute the script (see usage section).

---

## Configuration

Create a `config.json` file in the project root. Below is the **complete default configuration**:

```json
{
    "llm_api": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "you-api-key",
        "model": "llama-3.2"
    },
    "conversion": {
        "max_context_tokens": 4096,
        "chunk_overlap_tokens": 200,
        "save_images": true,
        "filter_images_by_llm": true,
        "image_prefix": "img",
        "temperature": 0.1,
        "max_tokens_response": 4096
    },
    "input_output": {
        "input_path": ".",
        "output_path": "output",
        "recursive": true
    }
}
```

### Configuration Details

####  Settings

| Parameter | Default | Description |
|-------|-----|-------|
| `base_url` | `http://localhost:11434/v1` | The API endpoint. |
| `api_key` | `api-key` | Authentication key. |
| `model` | `llama-3.2` | The model name loaded for text and image processing conversion. |

#### `conversion` (Processing Settings)

| Parameter | Default | Description |
|-------|-----|-------|
| `max_context_tokens` | `4096` | Maximum tokens allowed per LLM request. Truncates text if exceeded. |
| `chunk_overlap_tokens` | `200` | (Reserved for future chunking logic) Overlap between text chunks. |
| `save_images` | `true` | Whether to extract and save images found in PDFs. |
| `filter_images_by_llm` | `true` | If `true`, uses a vision model to decide if an image is relevant (charts/diagrams) or irrelevant (headers/TOC). |
| `image_prefix` | `img` | Prefix for saved image filenames (e.g., `img_1_cover.jpg`). |
| `temperature` | `0.1` | Sampling temperature for the LLM (lower = more deterministic). |
| `max_tokens_response` | `4096` | Max tokens the LLM can generate in a single response. |

#### `input_output` (I/O Settings)

| Parameter | Default | Description |
|-------|-----|-------|
| `input_path` | `.` | Path to a PDF file or a directory containing PDFs. |
| `output_path` | `output` | Directory where converted Markdown files will be saved. |
| `recursive` | `true` | If `input_path` is a directory, recursively search subdirectories for PDFs. |

---

## Usage

The run the script with the config flag pointing to the config file:

```bash
python pdf-to-markdown.py --config config.json
```



---

## Output Structure

When the script runs, it creates the following directory structure in the `output_path`:

```text
output/
├── Document_Name_1/
│   ├── Document_Name_1.md      # Main Markdown file
│   └── images/                 # Extracted relevant images
│       ├── img_1_cover.jpg
│       ├── img_2_0.jpg
│       └── ...
├── Document_Name_2/
│   └── ...
```

---

## Image Filtering Logic

The script uses a multi-stage filtering pipeline to decide whether to save an image:

1.  **Geometric Check**: Skips images that are too small (< 50px width/height) or have extreme aspect ratios (e.g., thin strips).
2.  **Pixel Density Check**: Analyzes pixel data to detect "white" images (likely scanned text pages).
3.  **LLM Relevance Check** (Optional):
    - If `filter_images_by_llm` is `true`, the script sends the image to a vision-capable model.
    - **REJECT**: Table of Contents, Index, page numbers, headers, footers, logos, scanned text.
    - **ACCEPT**: Charts, graphs, diagrams, technical illustrations, cover photos.

---
