# PDF to Markdown Converter

A Python tool that converts PDF documents to clean, structured Markdown using an LLM for text formatting. Features optional AI-powered layout detection via [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) for precise extraction of text, tables, figures, and formulas before OCR and image saving.

As a side note, I would suggest to everyone to use [datalab-to/marker](https://github.com/datalab-to/marker) first. It is far superior to this project. I made this script because for some documents, Marker can produce long repeated sections of text making the MD output virtually unusable. I use this script to convert these files for now.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Layout Detection](#layout-detection)
- [Legacy Mode](#legacy-mode)
- [Output Structure](#output-structure)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

## Features

- **AI Layout Detection** — Uses DocLayout-YOLO to identify text, tables, figures, formulas, titles, headers, footers, captions, and lists before processing
- **Tables as Images** — Detected table regions are saved as high-resolution JPEG images and referenced in the Markdown output
- **Smart Image Filtering** — Two-stage filtering: fast heuristic pre-filter drops decorative elements, optional LLM check classifies remaining images
- **LLM-Powered Formatting** — Converts raw extracted text to clean Markdown with proper headings, paragraphs, and structure
- **OCR Support** — Handles scanned documents via Tesseract OCR with configurable language
- **Page Batching** — Combines small consecutive pages into single LLM calls for efficiency
- **Table of Contents** — Auto-generates a linked Markdown TOC from detected headings
- **Cover Detection** — Identifies and saves cover pages as full-page snapshots
- **Legacy Mode** — Original image extraction pipeline preserved as a fallback when layout detection is disabled

## Requirements

- Python 3.8+
- An OpenAI-compatible LLM API endpoint (local or remote)
- Tesseract OCR engine (optional, for scanned documents)

## Installation

### 1. Install Tesseract OCR (optional, for scanned PDFs)

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Clone Repository and Install Python dependencies

```bash
#Clone or download the repository
git clone https://github.com/proto-deus/Pdf-to-Markdown

#(Recommended) Create venv:
python -m venv pdf-to-markdown

#Navigate to your venv
cd pdf-to-markdown

#Activate venv
scrips/activate.bat

#Install requirements
pip install -r requirements.txt
```

#### `requirements.txt`
```txt
# Core PDF processing
PyMuPDF>=1.24.0

# LLM API
openai>=1.0.0
tiktoken>=0.5.0

# Progress bars
tqdm>=4.65.0

# DocLayout-YOLO (AI layout detection)
doclayout-yolo>=0.0.4

# OCR support (optional but recommended)
pytesseract>=0.3.10
Pillow>=10.0.0

# Required by doclayout-yolo / ultralytics
ultralytics>=8.0.0
torch>=1.8.0
torchvision>=0.9.0
opencv-python>=4.6.0
numpy>=1.23.0
huggingface_hub
```

**CPU-only PyTorch (lighter install):**
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 3. Download the DocLayout-YOLO model

The model file (`doclayout_yolo_docstructbench_imgsz1024.pt`) is automatically downloaded on first run. If it doesn't download automatically get it from [Here](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/blob/main/doclayout_yolo_docstructbench_imgsz1024.pt) and put it in the root directory.

## Configuration

Create or edit `config.json`:

```json
{
  "llm_api": {
    "base_url": "http://localhost:11434/v1",
    "api_key": "your-key-here",
    "model": "llama-3.2-vision"
  },
  "conversion": {
    "max_context_tokens": 4096,
    "chunk_overlap_tokens": 200,
    "save_images": true,
    "filter_images_by_llm": true,
    "batch_pages": true,
    "batch_fill_ratio": 0.6,
    "generate_toc": true,
    "show_page_breaks": false,
    "force_ocr": false,
    "ocr_language": "eng",
    "image_min_width": 50,
    "image_min_height": 50,
    "image_min_aspect_ratio": 0.02,
    "image_max_aspect_ratio": 50,
    "image_white_ratio_threshold": 0.98,
    "image_tiny_boost": true,
    "image_max_dimension": 2500,
    "image_jpeg_quality": 80,
    "image_min_size": 1500,
    "temperature": 0.1,
    "max_tokens_response": 4000,
    "image_prefix": "img",
    "cover_check_pages": 4
  },
  "input_output": {
    "input_path": ".",
    "output_path": "output",
    "recursive": true,
    "log_file": "pdf_converter.log",
    "log_level": "INFO"
  },
  "layout_detection": {
    "enabled": true,
    "model_path": "doclayout_yolo_docstructbench_imgsz1024.pt",
    "confidence_threshold": 0.3,
    "iou_threshold": 0.45,
    "imgsz": 1024,
    "device": "cpu",
    "save_tables_as_images": true,
    "table_render_scale": 3.0,
    "layout_region_ocr": false,
    "min_region_area": 500
  }
}
```

## Usage

### Basic usage

```bash
# Point to your config file
python PDF-to-Markdown.py --config config.json

```

### CLI overrides

```bash
# Force layout detection ON
python PDF-to-Markdown.py --layout

# Force layout detection OFF (legacy mode)
python PDF-to-Markdown.py --no-layout

```

## How It Works

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Page Extraction                                   │
│                                                             │
│  For each page:                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  DocLayout-YOLO (if enabled)                        │    │
│  │  → Doclayout Detects regions: text, table, figure,  │    │
│  │    formula, title, caption, list, header, footer    │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐    │
│  │  Region Processing                                  │    │
│  │  • Text/Title/Section → extract text (PDF + OCR)    │    │
│  │  • Table → extract text + save as image             │    │
│  │  • Figure → save as image (+ LLM relevance check)   │    │
│  │  • Formula → save as image                          │    │
│  │  • Header/Footer → include as comments              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  OR Legacy mode:                                            │
│  → Extract text directly from page (Or OCR if required)     │
│  → Extract embedded images via xref                         │
│  → Pre-filter (heuristic) + LLM relevance check             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Page Batching                                     │
│  → Merge small consecutive pages into LLM-sized batches     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: LLM Conversion                                    │
│  → Each batch sent to LLM for Markdown formatting           │
│  → Template detection and cleanup                           │
│  → Heading hierarchy normalization                          │
│  → Duplicate heading removal                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: Assembly                                          │
│  → Generate Table of Contents (optional)                    │
│  → Combine all batches with page breaks                     │
│  → Write final .md file                                     │
└─────────────────────────────────────────────────────────────┘
```

## Layout Detection

When `layout_detection.enabled` is `true`, each page is rendered as an image and passed through DocLayout-YOLO, which identifies and classifies rectangular regions:

| Detected Class | Behavior |
|---|---|
| `text` | Text extracted from region (PDF text or OCR fallback) |
| `title` | Text extracted and formatted as `# Heading` |
| `section_header` | Text extracted and formatted as `## Heading` |
| `list` | Text extracted and preserved as-is |
| `caption` | Text extracted and formatted as `*italic*` |
| `table` | Text extracted for Markdown + saved as image |
| `figure` | Saved as image (with optional LLM relevance check) |
| `formula` | Saved as image |
| `page_header` | Included as HTML comment |
| `page_footer` | Included as HTML comment |

### Tables as Images

When `save_tables_as_images` is `true`, every detected table region is rendered at the configured `table_render_scale` (default 3× resolution) and saved as a JPEG. The Markdown output includes both the extracted text version and the image reference:

```markdown
| Column A | Column B |
|----------|----------|
| Data 1   | Data 2   |

```

### Model Notes

- The default model is `doclayout_yolo_docstructbench_imgsz1024.pt`
- The model file downloads automatically on first run or download [manually](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/blob/main/doclayout_yolo_docstructbench_imgsz1024.pt)
- Change `device` to `"cuda"` if you have a compatible GPU for faster speeds.
- Lower `confidence_threshold` to detect more regions; raise it for fewer false positives

## Legacy Mode

When layout detection is disabled (`enabled: false` or `--no-layout`), the original extraction pipeline runs:

1. Extract text directly from each PDF page
2. Identify embedded images via PDF xref table
3. Apply heuristic pre-filter (size, aspect ratio, white pixel ratio)
4. Optionally send images to the LLM for relevance classification
5. Save relevant images as JPEG

This mode is the default if `doclayout-yolo` is not installed.

## Output Structure

```
output/
└── document_name/
    ├── document_name.md          # Final Markdown
    └── images/
        ├── img_1_cover.jpg       # Cover page snapshot
        ├── img_3_table_r0_table.jpg   # Table as image
        ├── img_5_fig_r1_fig.jpg       # Figure
        ├── img_5_table_r2_table.jpg   # Table
        └── img_8_formula_r0_formula.jpg # Formula
```

## Configuration Reference

### `llm_api`

| Key | Default | Description |
|---|---|---|
| `base_url` | `http://localhost:11434/v1` | OpenAI-compatible API endpoint |
| `api_key` | `your-key-here` | API key (use any value for local LLMs) |
| `model` | `llama-3.2-vision` | Model name for chat completions |

### `conversion`

| Key | Default | Description |
|---|---|---|
| `max_context_tokens` | `4096` | Max tokens per LLM context window |
| `chunk_overlap_tokens` | `200` | Overlap between chunks |
| `save_images` | `true` | Whether to extract and save images |
| `filter_images_by_llm` | `true` | Use LLM to classify image relevance |
| `batch_pages` | `true` | Combine small pages into batches |
| `batch_fill_ratio` | `0.6` | Target fill ratio for batches |
| `generate_toc` | `true` | Generate table of contents |
| `show_page_breaks` | `false` | Add page break markers in output |
| `force_ocr` | `false` | Force OCR on all pages |
| `ocr_language` | `eng` | Tesseract OCR language code |
| `image_min_width` | `50` | Min bbox width (points) for pre-filter |
| `image_min_height` | `50` | Min bbox height (points) for pre-filter |
| `image_min_aspect_ratio` | `0.02` | Min aspect ratio to consider |
| `image_max_aspect_ratio` | `50` | Max aspect ratio to consider |
| `image_white_ratio_threshold` | `0.98` | White pixel ratio to reject |
| `image_tiny_boost` | `true` | Adjust threshold for small images |
| `image_max_dimension` | `2500` | Max pixel dimension for saved images |
| `image_jpeg_quality` | `80` | JPEG compression quality (1-100) |
| `image_min_size` | `1500` | Min pixel dimension (upscale small images) |
| `temperature` | `0.1` | LLM sampling temperature |
| `max_tokens_response` | `4000` | Max tokens in LLM response |
| `image_prefix` | `img` | Prefix for saved image filenames |
| `cover_check_pages` | `4` | Only check first N pages for covers |

### `input_output`

| Key | Default | Description |
|---|---|---|
| `input_path` | `.` | File or directory of PDFs to process |
| `output_path` | `output` | Output directory |
| `recursive` | `true` | Search subdirectories for PDFs |
| `log_file` | `pdf_converter.log` | Log file path |
| `log_level` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |

### `layout_detection`

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable DocLayout-YOLO layout detection |
| `model_path` | `doclayout_yolo_docstructbench_imgsz1024.pt` | Path to YOLO model weights |
| `confidence_threshold` | `0.3` | Min confidence for detections |
| `iou_threshold` | `0.45` | IoU threshold for NMS |
| `imgsz` | `1024` | Model input image size |
| `device` | `cpu` | Inference device (`cpu`, `cuda`, `0`, etc.) |
| `save_tables_as_images` | `true` | Save detected tables as images |
| `table_render_scale` | `3.0` | Render scale for table snapshots |
| `layout_region_ocr` | `false` | Force OCR on all layout regions |
| `min_region_area` | `500` | Min region area in PDF points² |

## Troubleshooting

### `No module named 'doclayout_yolo'`

Install the package:
```bash
pip install doclayout-yolo
```

The script will automatically fall back to legacy mode if the module is unavailable.

### `No module named 'pytesseract'`

Install for OCR support:
```bash
pip install pytesseract Pillow
```

Also ensure the Tesseract binary is installed (see [Installation](#installation)).

### Layout detection is slow

- Use a GPU by setting `"device": "cuda"` or `"device": "0"` in `layout_detection`
- Lower `"imgsz"` from `1024` to `640` for faster (but less accurate) detection
- Set `"confidence_threshold"` higher to reduce post-processing

### Images are too large or too small

- Adjust `image_max_dimension` (default 2500) to cap image size
- Adjust `image_min_size` (default 1500) to control upscaling of small images
- Adjust `image_jpeg_quality` (default 80) for file size vs quality

### LLM returns template/placeholder text

This is detected and discarded automatically. If persistent:
- Lower `temperature` in `llm_api`
- Try a different model (An instruct variant or larger model)
- Check your API endpoint is responding correctly

### Password-protected PDFs

Set the password in config:
```json
"conversion": {
    "pdf_password": "your_password"
}
```

### Logs

Check `pdf_converter.log` (or your configured `log_file`) for detailed processing information. Set `"log_level": "DEBUG"` for verbose output.
