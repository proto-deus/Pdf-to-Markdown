# PDF to Markdown Converter

Convert PDF documents into clean, structured Markdown using a local or remote
LLM (OpenAI-compatible API). Handles native text PDFs, column layouts, tables,
and scanned documents via OCR.

As a side note, I would suggest to everyone to use [datalab-to/marker](https://github.com/datalab-to/marker) first. It is far superior to this project. I made this script because for some documents, Marker can produce long repeated sections of text making the MD output virtually unusable. I use this script to convert these files for now.

## Features

- **LLM-powered reflow** — fixes broken line breaks, column layouts, and
  fragmented paragraphs
- **Automatic heading detection** — normalises heading hierarchy and removes
  cross-page duplicates
- **Table of Contents generation** — builds a linked TOC from all detected
  headings
- **Smart image extraction** — filters decorative elements (borders, bullets,
  logos) and saves only meaningful images (diagrams, charts, photos) using
  heuristic + LLM checks
- **Scanned PDF support** — OCR via Tesseract with preprocessing (contrast
  enhancement, binarisation) for scanned/image-only documents
- **Page batching** — merges small consecutive pages into single LLM calls to
  reduce API cost and latency
- **Configurable OCR languages** — supports any Tesseract language pack
  (including combined packs like `eng+fra`)
- **Resume support** — skips already-processed files on re-run
- **Structured logging** — dual output to console and log file with
  configurable verbosity
- **Password-protected PDFs** — configurable password for encrypted documents

## Prerequisites

| Dependency | Required | Install |
|---|---|---|
| Python 3.9+ | Yes | [python.org](https://python.org) |
| PyMuPDF | Yes | `pip install PyMuPDF` |
| openai | Yes | `pip install openai` |
| tiktoken | Yes | `pip install tiktoken` |
| tqdm | Yes | `pip install tqdm` |
| pytesseract | OCR only | `pip install pytesseract` |
| Pillow | OCR only | `pip install Pillow` |
| numpy | OCR preprocessing | `pip install numpy` |
| Tesseract engine | OCR only | See below |

### Installing Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu / Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download the installer from
[github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
and add it to your PATH.

To install additional language packs (e.g. French):
```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr-fra

# macOS
brew install tesseract-lang
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/pdf-to-markdown.git

# Install Python dependencies
pip install PyMuPDF openai tiktoken tqdm

# (Optional) Install OCR support
pip install pytesseract Pillow numpy
```

## Configuration

Create a `config.json` in the project root (or copy from the example below):

```json
{
    "llm_api": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "your-key-here",
        "model": "llama-3.2"
    },
    "conversion": {
        "max_context_tokens": 4096,
        "chunk_overlap_tokens": 200,
        "temperature": 0.1,
        "max_tokens_response": 4000,
        "save_images": true,
        "filter_images_by_llm": true,
        "image_prefix": "img",
        "batch_pages": true,
        "batch_fill_ratio": 0.6,
        "generate_toc": true,
        "show_page_breaks": false,
        "pdf_password": "",
        "force_ocr": true,
        "ocr_language": "eng",
        "ocr_preprocess": true,
        "ocr_dpi_scale": 3.0,
        "image_min_width": 15,
        "image_min_height": 15,
        "image_min_aspect_ratio": 0.02,
        "image_max_aspect_ratio": 50,
        "image_white_ratio_threshold": 0.98,
        "image_tiny_boost": true
    },
    "input_output": {
        "input_path": "./input",
        "output_path": "./output",
        "recursive": true,
        "log_file": "pdf_converter.log",
        "log_level": "INFO"
    }
}
```

### Configuration reference

#### `llm_api`

| Key | Default | Description |
|---|---|---|
| `base_url` | `http://localhost:11434/v1` | OpenAI-compatible API endpoint. Use `https://api.openai.com/v1` for OpenAI, or your Ollama/vLLM/LM Studio URL. |
| `api_key` | `your-key-here` | API key (use `ollama` or any non-empty string for local servers that don't require one). |
| `model` | `llama-3.2` | Model name. Must support chat completions. For image filtering, must have vision support. |

#### `conversion`

| Key | Default | Description |
|---|---|---|
| `max_context_tokens` | `4096` | Maximum tokens sent to the LLM per call. |
| `chunk_overlap_tokens` | `200` | Reserved for future overlap implementation. |
| `temperature` | `0.1` | LLM temperature. Lower = more deterministic formatting. |
| `max_tokens_response` | `4000` | Maximum tokens the LLM can generate per response. |
| `save_images` | `true` | Extract and save images from the PDF. |
| `filter_images_by_llm` | `true` | Use the LLM to decide if an image is worth saving. Disable for speed. |
| `image_prefix` | `img` | Filename prefix for saved images. |
| `batch_pages` | `true` | Merge small consecutive pages into single LLM calls. |
| `batch_fill_ratio` | `0.6` | How full a batch must be before starting a new one (0.0–1.0). Lower = more batching = fewer API calls. |
| `generate_toc` | `true` | Prepend a linked Table of Contents from extracted headings. |
| `show_page_breaks` | `false` | Add `*[Page N]*` markers between pages in output. |
| `pdf_password` | `""` | Password for encrypted PDFs. |
| `force_ocr` | `false` | Force ocr to process each page. If text is found, it overrides existing ocr. |
| `ocr_language` | `"eng"` | Tesseract language code. Combine with `+` (e.g. `"eng+fra"`). |
| `ocr_preprocess` | `true` | Apply contrast/binarisation before OCR. Requires `numpy`. |
| `ocr_dpi_scale` | `3.0` | Render scale for OCR images. Higher = more accurate but slower. |
| `image_min_width` | `15` | Minimum image width (in PDF points) before rejection. |
| `image_min_height` | `15` | Minimum image height (in PDF points) before rejection. |
| `image_min_aspect_ratio` | `0.02` | Minimum width/height ratio. Filters ultra-thin lines. |
| `image_max_aspect_ratio` | `50` | Maximum width/height ratio. Filters ultra-wide strips. |
| `image_white_ratio_threshold` | `0.98` | Max fraction of white pixels before an image is considered blank. |
| `image_tiny_boost` | `true` | Raise the white-ratio threshold for small images (they have proportionally more border whitespace). |

#### `input_output`

| Key | Default | Description |
|---|---|---|
| `input_path` | `./input` | Path to a PDF file or a directory of PDFs. |
| `output_path` | `./output` | Directory where Markdown and images are saved. |
| `recursive` | `true` | Search subdirectories for PDFs. |
| `log_file` | `pdf_converter.log` | Path to the log file. |
| `log_level` | `INFO` | Console log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

## Usage

### Basic

```bash
python pdf_converter.py
```

This reads `config.json` from the current directory and processes all PDFs
found at `input_path`.

### Custom config path

```bash
python pdf_converter.py --config /path/to/my_config.json
```

### Output structure

For a file called `report.pdf`, the converter creates:

```
output/
└── report/
    ├── report.md          # Final Markdown output
    └── images/
        ├── img_1_cover.jpg
        ├── img_3_0.jpg
        ├── img_3_1.jpg
        └── ...
```

### Example config for OpenAI

```json
{
    "llm_api": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-...",
        "model": "gpt-4o"
    },
    "conversion": {
        "max_context_tokens": 16000,
        "max_tokens_response": 8000,
        "temperature": 0.1,
        "save_images": true,
        "filter_images_by_llm": true,
        "batch_pages": true,
        "batch_fill_ratio": 0.5,
        "generate_toc": true,
        "ocr_language": "eng"
    },
    "input_output": {
        "input_path": "./papers",
        "output_path": "./markdown_output",
        "recursive": true,
        "log_level": "INFO"
    }
}
```

### Example config for Ollama (local)

```json
{
    "llm_api": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "llama3.2"
    },
    "conversion": {
        "max_context_tokens": 4096,
        "max_tokens_response": 4000,
        "temperature": 0.1,
        "save_images": true,
        "filter_images_by_llm": true,
        "batch_pages": true,
        "generate_toc": true
    },
    "input_output": {
        "input_path": "./input",
        "output_path": "./output"
    }
}
```

## How it works

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐
│  Input   │───▶│  Extract      │───▶│  Build       │───▶│  Convert   │
│  PDFs    │     │  text &      │     │  page        │     │  via LLM   │
│          │     │  images      │     │  batches     │     │            │
└──────────┘     └──────────────┘     └──────────────┘     └────────────┘
                        │                                           │
                        ▼                                           ▼
                 ┌──────────────┐                            ┌────────────┐
                 │  OCR fallback│                            │  Post-     │
                 │  (scanned    │                            │  process:  │
                 │   pages)     │                            │  clean,    │
                 └──────────────┘                            │  dedup,    │
                                                             │  TOC       │
                                                             └────────────┘
                                                                   │
                                                                   ▼
                                                            ┌────────────┐
                                                            │  .md file  │
                                                            │  + images/ │
                                                            └────────────┘
```

1. **Extraction** — PyMuPDF extracts native text and embedded images from each
   page.
2. **OCR fallback** — Pages with < 50 characters of extractable text are
   processed through Tesseract OCR.
3. **Image filtering** — Decorative images are removed via geometry/pixel
   heuristics, then remaining candidates are classified by the LLM.
4. **Batching** — Small consecutive pages are merged to minimise API calls.
5. **Conversion** — Each batch is sent to the LLM with a strict prompt that
   re-flows paragraphs, detects headings, and preserves tables/lists.
6. **Post-processing** — Template/placeholder text is stripped, heading
   hierarchy is normalised, duplicates are removed, and a TOC is generated.
7. **Output** — A single Markdown file is written with images saved alongside.

## Troubleshooting

### "Image relevance check failed (is the model multimodal?)"

The image filtering step requires a vision-capable model. If your model
doesn't support images, either:

- Use a multimodal model (e.g. `llama3.2-vision`, `gpt-4o`), or
- Disable image filtering:
  ```json
  "filter_images_by_llm": false
  ```

### Scanned PDF produces no text

Install the OCR dependencies:

```bash
pip install pytesseract Pillow numpy
```

And ensure the Tesseract binary is installed and on your PATH:

```bash
tesseract --version
```

### "API Call failed" or timeout errors

- Verify the `base_url` is reachable and the model is loaded.
- For local servers, confirm the server is running before starting the
  converter.
- Increase `timeout` in `call_llm_with_retry` if your model is slow.
- The converter retries transient errors (429, 5xx, timeouts) up to 3 times
  with exponential backoff.

### Output has template text like `[Insert heading here]`

This usually means the model is too small or isn't following instructions well.
Try:

- Use an instruct model.
- A larger / more capable model.
- Increasing `max_context_tokens` so the model gets more context.

## Requirements file

```
# requirements.txt
PyMuPDF>=1.24.0
openai>=1.0.0
tiktoken>=0.5.0
tqdm>=4.60.0

# Optional — OCR support
pytesseract>=0.3.10
Pillow>=10.0.0
numpy>=1.24.0
```

```bash
pip install -r requirements.txt
```
