import os
import json
import re
import fitz
import openai
import tiktoken
import time
import sys
import base64
import logging
import random
from tqdm import tqdm
from pathlib import Path

# Optional OCR support for scanned documents
try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

CONFIG_FILE = "config.json"

# Patterns that indicate LLM-generated template/placeholder text
TEMPLATE_PATTERNS = [
    r'\[insert\s+actual\s+heading',
    r'\[reflowed\s+body\s+text',
    r'\[insert\s+heading',
    r'\[insert\s+text',
    r'\[insert\s+content',
    r'\[placeholder',
    r'\[your\s+text\s+here',
    r'\[content\s+here',
    r'\[heading\s+here',
    r'\[body\s+text\s+here',
    r'\[add\s+content',
    r'since\s+no\s+actual\s+raw\s+text\s+was\s+provided',
    r'replace\s+bracketed\s+content\s+with\s+the\s+real',
    r'this\s+is\s+a\s+template',
    r'note:\s+since\s+no\s+actual',
    r'example\s*\(if\s+applicable\)',
    r'\[page\s+\d+\s+content\s+here\]',
    r'\[enter\s+text',
    r'template\s+output',
    r'no\s+content\s+was\s+provided',
    r'as\s+an\s+ai',
    r'i\s+cannot\s+process',
    r'please\s+provide\s+the\s+actual',
]
TEMPLATE_REGEX = re.compile('|'.join(TEMPLATE_PATTERNS), re.IGNORECASE)

_COPYRIGHT_RE = re.compile(
    r'(?:copyright|©|℗|\(c\)|isbn|all rights reserved|'
    r'library of congress|catalog(?:uing|ing)?\s+in\s+publication|'
    r'printed in|published by|first published|'
    r'this\s+(?:is\s+)?(?:a\s+)?(?:trade\s+)?(?:paper\s+)?back|'
    r'hardcover edition|paperback edition|ebook edition|'
    r'the\s+author|the\s+publisher)',
    re.IGNORECASE,
)

_TOC_RE = re.compile(
    r'(?:table of contents|contents|index|list of (?:figures|tables|illustrations))',
    re.IGNORECASE,
)

_TITLE_PAGE_RE = re.compile(
    r'(?:\bby\b\s+[A-Z]|'
    r'\b(?:edited|translated|compiled|introduced|foreword|preface|'
    r'illustrated)\s+by\b|'
    r'\b(?:press|publish(?:er|ing)?|books?|edition|'
    r'university|inc\.?|ltd\.?|llc)\b)',
    re.IGNORECASE,
)


class PDFProcessor:
    def __init__(self, config_path=CONFIG_FILE):
        self.load_config(config_path)
        self.setup_client()
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.setup_logging()

    # ------------------------------------------------------------------ #
    #  Configuration
    # ------------------------------------------------------------------ #
    def load_config(self, path):
        """Loads settings from JSON or creates a default."""
        if not os.path.exists(path):
            print(f"[ERROR] Config file '{path}' not found.")
            sys.exit(1)
        with open(path, 'r') as f:
            self.config = json.load(f)
        self.lm_settings = self.config.get("llm_api", {})
        self.conv_settings = self.config.get("conversion", {})
        self.io_settings = self.config.get("input_output", {})
        self.base_url = self.lm_settings.get("base_url", "http://localhost:11434/v1")
        self.api_key = self.lm_settings.get("api_key", "your-key-here")
        self.model = self.lm_settings.get("model", "llama-3.2")
        self.max_tokens = self.conv_settings.get("max_context_tokens", 4096)
        self.overlap = self.conv_settings.get("chunk_overlap_tokens", 200)
        self.save_images = self.conv_settings.get("save_images", True)
        self.filter_images = self.conv_settings.get("filter_images_by_llm", True)
        self.batch_pages = self.conv_settings.get("batch_pages", True)
        self.batch_fill_ratio = self.conv_settings.get("batch_fill_ratio", 0.6)
        self.generate_toc = self.conv_settings.get("generate_toc", True)
        self.show_page_breaks = self.conv_settings.get("show_page_breaks", False)
        self.force_ocr = self.conv_settings.get("force_ocr", False)
        self.ocr_language = self.conv_settings.get("ocr_language", "eng")
        self.img_min_width = self.conv_settings.get("image_min_width", 15)
        self.img_min_height = self.conv_settings.get("image_min_height", 15)
        self.img_min_aspect = self.conv_settings.get("image_min_aspect_ratio", 0.02)
        self.img_max_aspect = self.conv_settings.get("image_max_aspect_ratio", 50)
        self.img_white_threshold = self.conv_settings.get("image_white_ratio_threshold", 0.98)
        self.img_tiny_boost = self.conv_settings.get("image_tiny_boost", True)
        # Image scaling and compression settings
        self.img_max_dimension = self.conv_settings.get("image_max_dimension", 2500)
        self.img_jpeg_quality = self.conv_settings.get("image_jpeg_quality", 80)

    def setup_client(self):
        """Initializes the OpenAI-compatible client."""
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    # ------------------------------------------------------------------ #
    #  Cover / title / copyright / TOC detection helpers
    # ------------------------------------------------------------------ #

    def setup_logging(self):
        """Configure structured logging to file and console."""
        log_file = self.io_settings.get("log_file", "pdf_converter.log")
        log_level_str = self.io_settings.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        root_logger = logging.getLogger("pdf_converter")
        if root_logger.handlers:
            self.logger = root_logger
            return
        root_logger.setLevel(log_level)
        root_logger.propagate = False
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(fmt)
        root_logger.addHandler(ch)
        self.logger = root_logger

    # ------------------------------------------------------------------ #
    #  File discovery
    # ------------------------------------------------------------------ #
    def get_files(self):
        """Locates PDF files based on input settings."""
        input_path = self.io_settings.get("input_path", ".")
        output_path = self.io_settings.get("output_path", "output")
        files_to_process = []
        if not os.path.exists(input_path):
            self.logger.error(f"Input path does not exist: {input_path}")
            return [], output_path
        if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
            files_to_process.append(input_path)
        elif os.path.isdir(input_path):
            recursive = self.io_settings.get("recursive", True)
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        files_to_process.append(os.path.join(root, file))
                if not recursive:
                    break
        return files_to_process, output_path

    def num_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    # ------------------------------------------------------------------ #
    #  Image resize & save helpers
    # ------------------------------------------------------------------ #
    def _resize_image(self, image, max_dim=None):
        """Scale an image down if its longest side exceeds max_dim pixels."""
        if max_dim is None:
            max_dim = self.img_max_dimension
        w, h = image.size
        longest = max(w, h)
        if longest <= max_dim:
            return image
        ratio = max_dim / float(longest)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        resized = image.resize((new_w, new_h), Image.LANCZOS)
        self.logger.debug(
            f"Resized image {w}×{h} → {new_w}×{new_h} "
            f"(max_dim={max_dim})"
        )
        return resized

    def _pixmap_to_pil(self, pix):
        """Convert a PyMuPDF Pixmap to a PIL Image (always RGB)."""
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        if image.mode == "CMYK":
            image = image.convert("RGB")
        elif image.mode == "P":
            image = image.convert("RGB")
        elif image.mode == "RGBA":
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")
        # "RGB" stays as-is
        return image

    def _save_image_jpeg(self, image, filepath):
        """
        Resize (if needed) and save a PIL Image as JPEG
        with the configured quality and max dimension.
        """
        image = self._resize_image(image)
        # Ensure RGB mode for JPEG saving
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(
            filepath,
            "JPEG",
            quality=self.img_jpeg_quality,
            optimize=True
        )

    # ------------------------------------------------------------------ #
    #  Template detection / cleaning
    # ------------------------------------------------------------------ #
    def is_template_text(self, text):
        """Check if text contains LLM template/placeholder patterns."""
        if not text or not text.strip():
            return True
        if TEMPLATE_REGEX.search(text):
            return True
        if re.search(r'#{1,6}\s+.*\[(?:insert|enter|add|your)\b', text, re.IGNORECASE):
            return True
        return False

    def clean_template_text(self, text):
        """Remove template/placeholder lines from LLM output."""
        if not text:
            return text
        lines = text.split('\n')
        clean_lines = []
        in_template_block = False
        for line in lines:
            stripped = line.strip()
            if TEMPLATE_REGEX.search(line):
                in_template_block = True
                continue
            heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
            if heading_match:
                heading_text = heading_match.group(2).strip()
                if (heading_text.startswith('[') or
                        'section title' in heading_text.lower() or
                        TEMPLATE_REGEX.search(heading_text)):
                    in_template_block = True
                    continue
            if stripped.startswith('> ') and TEMPLATE_REGEX.search(stripped):
                continue
            if stripped == '---' and in_template_block:
                in_template_block = False
                continue
            in_template_block = False
            clean_lines.append(line)
        result = '\n'.join(clean_lines)
        result = re.sub(r'\n{4,}', '\n\n\n', result)
        return result.strip()

    # ------------------------------------------------------------------ #
    #  Image relevance
    # ------------------------------------------------------------------ #
    def is_image_likely_irrelevant(self, rect, pix):
        """
        Fast heuristic pre-filter to skip irrelevant images before LLM call.
        """
        if rect.width < self.img_min_width or rect.height < self.img_min_height:
            return True
        aspect_ratio = rect.width / rect.height
        if aspect_ratio > self.img_max_aspect or aspect_ratio < self.img_min_aspect:
            return True
        if pix:
            try:
                step = pix.n
                samples = pix.samples
                total_pixels = len(samples) // step
                if total_pixels == 0:
                    return True
                max_samples = 10000
                pixel_stride = max(1, total_pixels // max_samples)
                stride = pixel_stride * step
                white_count = 0
                total_checked = 0
                for i in range(0, len(samples) - step + 1, stride):
                    r = samples[i]
                    g = samples[i + 1]
                    b = samples[i + 2]
                    if r > 240 and g > 240 and b > 240:
                        white_count += 1
                    total_checked += 1
                if total_checked == 0:
                    return True
                white_ratio = white_count / total_checked
                threshold = self.img_white_threshold
                if self.img_tiny_boost:
                    area = rect.width * rect.height
                    if area < 10000:
                        tiny_scale = max(0, (area - 400) / (10000 - 400))
                        threshold = 0.998 - (0.998 - threshold) * tiny_scale
                if white_ratio > threshold:
                    return True
            except Exception:
                pass
        return False

    def is_image_relevant(self, image_bytes, page_text):
        """Sends the image and text context to the LLM to check relevance."""
        if not self.filter_images:
            return True
        try:
            b64_image = base64.b64encode(image_bytes).decode('utf-8')
            context_text = (page_text or "")[:500]
            prompt = (
                "You are a strict image relevance classifier. "
                "Analyze the provided image.\n\n"
                "REJECT (return NO) and do NOT save if the image is: "
                "Words, text, one or more paragraphs, a book page, part of a book page, "
                "a heading, a title a page number, a header/footer ornament, "
                "a decorative border element, a bullet-point glyph, "
                "a tiny company logo watermark, "
                "or any lines of running text extracted from the page body.\n\n"
                "ACCEPT (return YES) and save if the image is ANY of the following "
                "(regardless of size): "
                "a diagram, chart, graph, map, technical illustration, photograph, "
                "equation rendered as an image, figure, small icon with informational "
                "meaning, screenshot, flowchart, or any visual content that conveys "
                "information beyond plain text.\n\n"
                f"Text Context (for reference): {context_text}\n\n"
                "Reply strictly with 'YES' or 'NO'."
            )
            response_content = self.call_llm_with_retry(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=5
            )
            result = response_content.strip().upper()
            return "YES" in result
        except Exception as e:
            self.logger.warning(
                f"Image relevance check failed (is the model multimodal?): {e}"
            )
            return False

    # ------------------------------------------------------------------ #
    #  LLM call with retry
    # ------------------------------------------------------------------ #
    def call_llm_with_retry(self, messages, max_tokens, max_retries=3,
                            temperature=None):
        """Call the LLM with exponential backoff on transient errors."""
        if temperature is None:
            temperature = self.conv_settings.get("temperature", 0.1)
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=120
                )
                return response.choices[0].message.content
            except (openai.RateLimitError,
                    openai.APIConnectionError,
                    openai.APITimeoutError) as e:
                wait = (2 ** attempt) + random.uniform(0, 1)
                self.logger.warning(
                    f"API transient error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait:.1f}s…"
                )
                time.sleep(wait)
            except openai.APIStatusError as e:
                if e.status_code >= 500:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    self.logger.warning(
                        f"Server error {e.status_code} "
                        f"(attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {wait:.1f}s…"
                    )
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"LLM call failed after {max_retries} attempts.")

    # ------------------------------------------------------------------ #
    #  Heading hierarchy & deduplication
    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize_heading_text(text):
        """Normalize heading text for comparison."""
        text = text.strip().lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def normalize_heading_hierarchy(self, markdown_text):
        """Ensure heading levels are logically consistent."""
        lines = markdown_text.split('\n')
        new_lines = []
        last_level = 0
        for line in lines:
            match = re.match(r'^(#{1,6})\s+(.*)', line)
            if match:
                text = match.group(2).strip()
                level = len(match.group(1))
                if last_level > 0 and level > last_level + 1:
                    level = last_level + 1
                if len(text) < 3 and level <= 2:
                    new_lines.append(f"**{text}**")
                    continue
                last_level = level
                new_lines.append(f"{'#' * level} {text}")
            else:
                new_lines.append(line)
        return '\n'.join(new_lines)

    def deduplicate_headings(self, markdown_text, seen_headings):
        """Removes duplicate headings."""
        lines = markdown_text.split('\n')
        new_lines = []
        for line in lines:
            match = re.match(r'^(#{1,6})\s+(.*)', line)
            if match:
                raw_text = match.group(2).strip()
                normalized = self.normalize_heading_text(raw_text)
                if normalized in seen_headings:
                    new_lines.append(f"**{raw_text}**")
                else:
                    seen_headings[normalized] = raw_text
                    new_lines.append(line)
            else:
                new_lines.append(line)
        return "\n".join(new_lines), seen_headings

    # ------------------------------------------------------------------ #
    #  Table of Contents generation
    # ------------------------------------------------------------------ #
    def generate_table_of_contents(self, all_page_markdowns):
        """Scan all collected Markdown for headings and build a linked TOC."""
        headings_found = []
        for page_md in all_page_markdowns:
            for line in page_md.split('\n'):
                match = re.match(r'^(#{1,6})\s+(.*)', line)
                if match:
                    level = len(match.group(1))
                    text = match.group(2).strip()
                    if TEMPLATE_REGEX.search(text):
                        continue
                    headings_found.append((level, text))
        if not headings_found:
            return ""
        toc_lines = ["# Table of Contents\n"]
        seen_anchors = {}
        for level, text in headings_found:
            anchor = re.sub(r'[^\w\s-]', '', text.lower())
            anchor = re.sub(r'[\s]+', '-', anchor).strip('-')
            if anchor in seen_anchors:
                seen_anchors[anchor] += 1
                anchor = f"{anchor}-{seen_anchors[anchor]}"
            else:
                seen_anchors[anchor] = 0
            indent = "  " * (level - 1)
            toc_lines.append(f"{indent}- [{text}](#{anchor})")
        return "\n".join(toc_lines)

    # ------------------------------------------------------------------ #
    #  OCR helpers
    # ------------------------------------------------------------------ #
    def extract_text_ocr(self, page, page_num):
        """Extract text from a scanned page using OCR."""
        if not OCR_AVAILABLE:
            return ""
        try:
            scale = 3.0
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(image, lang=self.ocr_language)
            result = text.strip()
            self.logger.debug(
                f"OCR page {page_num}: extracted {len(result)} chars "
                f"(lang={self.ocr_language})"
            )
            return result
        except Exception as e:
            self.logger.warning(f"OCR failed on page {page_num}: {e}")
            return ""

    def check_if_scanned(self, doc):
        """Check if a PDF appears to be scanned."""
        pages_to_check = min(5, len(doc))
        empty_pages = 0
        for i in range(pages_to_check):
            text = doc[i].get_text("text").strip()
            if len(text) < 50:
                empty_pages += 1
        return empty_pages > pages_to_check * 0.7

    # ------------------------------------------------------------------ #
    #  Page extraction helpers
    # ------------------------------------------------------------------ #

    def _looks_like_copyright(self, text):
        """Return True when the text clearly belongs to a copyright page."""
        if len(text) > 800:
            return False
        return bool(_COPYRIGHT_RE.search(text))

    def _looks_like_toc(self, text):
        """Return True when the text is a table-of-contents page."""
        if _TOC_RE.search(text):
            return True
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) < 4:
            return False
        page_ref_lines = sum(1 for l in lines if re.search(r'\s+\d+\s*$', l))
        return page_ref_lines / len(lines) > 0.5

    def _looks_like_title_page(self, text):
        """Return True when the text reads like a formal title page."""
        if len(text) < 30 or len(text) > 1000:
            return False
        if _TITLE_PAGE_RE.search(text):
            return True
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if 2 <= len(lines) <= 8:
            has_by = any(' by ' in l.lower() or l.lower().startswith('by ') for l in lines)
            has_publisher = any(
                re.search(r'press|publish|books?|edition|university', l, re.I)
                for l in lines
            )
            if has_by or has_publisher:
                return True
        return False


    def is_cover_page(self, page, page_num):
        """
        Heuristic to detect *actual* cover pages while excluding title,
        copyright, and TOC pages that commonly appear in the first few
        pages of a book.

        Strategies
        ----------
        1. Dominant full-page image  →  cover
        2. Large image + very little text  →  cover
        3. Text-only with recognisable cover layout  →  cover (rare)

        Exclusion filters
        -----------------
        • Copyright / ISBN / rights pages
        • Table-of-contents / index pages
        • Formal title pages (author + publisher info)
        """
        max_cover_page = self.conv_settings.get("cover_check_pages", 4)
        if page_num > max_cover_page:
            return False

        text = page.get_text("text").strip()
        text_length = len(text)
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        if page_area < 100:
            return False

        # ── EXCLUSION FILTERS ──────────────────────────────────────────
        if self._looks_like_copyright(text):
            self.logger.debug(f"Page {page_num+1}: skipped (copyright page)")
            return False
        if self._looks_like_toc(text):
            self.logger.debug(f"Page {page_num+1}: skipped (table of contents)")
            return False
        if self._looks_like_title_page(text):
            self.logger.debug(f"Page {page_num+1}: skipped (title page)")
            return False

        # ── IMAGE-BASED COVER DETECTION ────────────────────────────────
        images = page.get_images(full=True)
        num_images = len(images)
        largest_area = 0.0
        largest_coverage = 0.0

        for img_info in images:
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                for rect in rects:
                    img_area = rect.width * rect.height
                    coverage = img_area / page_area
                    if img_area > largest_area:
                        largest_area = img_area
                    if coverage > largest_coverage:
                        largest_coverage = coverage
            except Exception:
                pass

        # Strategy 1: Dominant full-page image.
        if (largest_coverage > 0.60
                and largest_area > 100000
                and text_length < 300):
            self.logger.debug(
                f"Cover detected (page {page_num+1}): "
                f"dominant image ({largest_coverage:.0%}, "
                f"{largest_area:,.0f} pt², {text_length} chars text)"
            )
            return True

        # Strategy 2: Image present + almost no text at all
        if num_images >= 1 and text_length < 80:
            self.logger.debug(
                f"Cover detected (page {page_num+1}): "
                f"image + minimal text ({text_length} chars)"
            )
            return True

        # Strategy 3: Pure text cover — extremely short, centred, no images.
        if page_num <= 1 and num_images == 0 and text_length < 60:
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if len(lines) <= 2:
                self.logger.debug(
                    f"Cover detected (page {page_num+1}): "
                    f"text-only cover ({len(lines)} lines)"
                )
                return True

        return False

    def extract_page_data(self, doc, page_num, images_dir, ocr_pbar=None, image_pbar=None):
        """Extract raw text and image references for a single page."""
        page = doc[page_num]
        text = page.get_text("text").strip()

        if len(text) < 50 or self.force_ocr:
            ocr_text = self.extract_text_ocr(page, page_num + 1)
            if ocr_text:
                text = ocr_text
            if ocr_pbar is not None:
                ocr_pbar.update(1)

        if text and len(text) > 20 and self.is_template_text(text):
            self.logger.debug(
                f"Page {page_num + 1}: template-like text discarded "
                f"({len(text)} chars)"
            )
            text = ""

        image_refs = []
        if not self.save_images:
            return text, image_refs

        is_cover = self.is_cover_page(page, page_num)
        if is_cover:
            image_refs = self._save_cover_snapshot(page, page_num, images_dir)
        else:
            image_refs = self._save_page_images(
                doc, page, page_num, images_dir, text, image_pbar
            )
        return text, image_refs

    # ------------------------------------------------------------------ #
    #  Cover snapshot
    # ------------------------------------------------------------------ #

    def _save_cover_snapshot(self, page, page_num, images_dir):
        """
        Render and save a cover page snapshot as a compressed JPEG.

        FIX: PyMuPDF's get_pixmap already applies the page's rotation
        angle, so we no longer rotate the pixmap a second time.
        """
        image_refs = []
        try:
            mat = fitz.Matrix(3.0, 3.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            image = self._pixmap_to_pil(pix)
            prefix = self.conv_settings.get('image_prefix', 'img')
            img_filename = f"{prefix}_{page_num + 1}_cover.jpg"
            img_path = os.path.join(images_dir, img_filename)
            self._save_image_jpeg(image, img_path)

            if os.path.exists(img_path) and os.path.getsize(img_path) > 1000:
                image_refs.append(
                    f"![Image: {img_filename}](./images/{img_filename})"
                )
                self.logger.debug(
                    f"Saved cover snapshot page {page_num + 1}: "
                    f"{img_filename} ({os.path.getsize(img_path):,} bytes)"
                )
            else:
                self.logger.warning(
                    f"Cover snapshot for page {page_num + 1} is suspiciously "
                    f"small or missing ({img_path}). Falling back."
                )
                raise ValueError("Cover snapshot too small")

        except Exception as e:

            self.logger.warning(
                f"Primary cover render failed for page {page_num + 1} ({e}). "
                f"Trying fallback render."
            )
            try:
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # Try pix.save as a last resort (Pillow-free path)
                prefix = self.conv_settings.get('image_prefix', 'img')
                img_filename = f"{prefix}_{page_num + 1}_cover.jpg"
                img_path = os.path.join(images_dir, img_filename)

                # Convert through PIL as well (handles CMYK, etc.)
                image = self._pixmap_to_pil(pix)
                self._save_image_jpeg(image, img_path)

                if os.path.exists(img_path) and os.path.getsize(img_path) > 500:
                    image_refs.append(
                        f"![Image: {img_filename}](./images/{img_filename})"
                    )
                    self.logger.info(
                        f"Saved cover snapshot (fallback) page {page_num + 1}: "
                        f"{img_filename}"
                    )
                else:
                    self.logger.warning(
                        f"Fallback cover snapshot for page {page_num + 1} "
                        f"also failed."
                    )
            except Exception as e2:
                self.logger.error(
                    f"All attempts to save cover page {page_num + 1} failed: {e2}"
                )

        return image_refs

    # ------------------------------------------------------------------ #
    #  Image extraction & saving
    # ------------------------------------------------------------------ #
    def _count_total_images(self, doc):
        """Count total images across all pages for progress tracking."""
        total = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            img_list = page.get_images(full=True)
            total += len(img_list)
        return total

    def _save_page_images(self, doc, page, page_num, images_dir, text, image_pbar=None):
        """Extract, filter, and save relevant images from a page."""
        image_refs = []
        img_list = page.get_images(full=True)
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        is_likely_scanned = len(text.strip()) < 100
        for i, img in enumerate(img_list):
            try:
                xref = img[0]
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                rect = rects[0]
                img_area = rect.width * rect.height
                coverage_ratio = img_area / page_area if page_area > 0 else 0
                # Skip background / full-page scan images
                should_skip = False
                skip_reason = ""
                coverage_threshold = 0.75 if is_likely_scanned else 0.95
                if coverage_ratio > coverage_threshold:
                    should_skip = True
                    skip_reason = f"covers {coverage_ratio:.1%} of page"
                elif (abs(rect.x0) < 10 and abs(rect.y0) < 10 and
                      abs(rect.x1 - page_rect.x1) < 10 and
                      abs(rect.y1 - page_rect.y1) < 10):
                    should_skip = True
                    skip_reason = "full-page background positioning"
                elif len(img_list) == 1 and coverage_ratio > 0.75:
                    should_skip = True
                    skip_reason = "single image covering >75% of page"
                elif is_likely_scanned and coverage_ratio > 0.4:
                    should_skip = True
                    skip_reason = f"large image ({coverage_ratio:.1%}) on text-sparse page"
                if should_skip:
                    self.logger.debug(
                        f"Skipping image {i} on page {page_num+1}: "
                        f"{skip_reason} (likely background scan)"
                    )
                    continue
                if self.is_image_likely_irrelevant(rect, None):
                    continue
                area = rect.width * rect.height
                if area < 2500:
                    scale = 2.0
                elif area < 10000:
                    scale = 1.5
                else:
                    scale = 1.0
                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
                if page.rotation != 0:
                    pix = pix.rotate(page.rotation)
                if self.is_image_likely_irrelevant(rect, pix):
                    continue
                image = self._pixmap_to_pil(pix)
                img_bytes_io = io.BytesIO()
                image_for_check = self._resize_image(image)
                if image_for_check.mode != "RGB":
                    image_for_check = image_for_check.convert("RGB")
                image_for_check.save(
                    img_bytes_io, "JPEG",
                    quality=self.img_jpeg_quality, optimize=True
                )
                img_bytes = img_bytes_io.getvalue()
                if self.is_image_relevant(img_bytes, text):
                    prefix = self.conv_settings.get('image_prefix', 'img')
                    img_filename = f"{prefix}_{page_num + 1}_{i}.jpg"
                    img_path = os.path.join(images_dir, img_filename)
                    self._save_image_jpeg(image, img_path)
                    image_refs.append(
                        f"![Image: {img_filename}](./images/{img_filename})"
                    )
                    self.logger.debug(
                        f"  Saved image {i} from page {page_num+1} "
                        f"({rect.width:.0f}×{rect.height:.0f} pts, "
                        f"scale={scale}x, coverage={coverage_ratio:.1%})"
                    )
            except Exception as img_err:
                self.logger.warning(
                    f"Error processing image {i} on page {page_num + 1}: "
                    f"{img_err}"
                )
            finally:
                if image_pbar is not None:
                    image_pbar.update(1)
        return image_refs

    # ------------------------------------------------------------------ #
    #  Page batching
    # ------------------------------------------------------------------ #
    def _page_chunk_text(self, page_data):
        """Combine a page's text and image refs into a single string."""
        parts = []
        if page_data["text"].strip():
            parts.append(page_data["text"])
        if page_data["images"]:
            parts.append("\n".join(page_data["images"]))
        return "\n\n".join(parts) if parts else ""

    def _should_merge_page(self, current_tokens, next_text):
        """Return True when adding next_text stays within the fill ratio."""
        next_tokens = self.num_tokens(next_text)
        return (current_tokens + next_tokens) < (
                self.max_tokens * self.batch_fill_ratio
        )

    def build_batches(self, page_data_list):
        """Merge small consecutive pages into batches."""
        batches = []
        current = {"pages": [], "text": "", "tokens": 0}
        for pd in page_data_list:
            page_text = self._page_chunk_text(pd)
            if not page_text.strip():
                if current["tokens"] > 0:
                    current["pages"].append(pd["page_num"])
                else:
                    batches.append({
                        "pages": [pd["page_num"]],
                        "text": "",
                        "start": pd["page_num"],
                        "end": pd["page_num"],
                    })
                continue
            if (current["tokens"] > 0 and
                    not self._should_merge_page(current["tokens"], page_text)):
                batches.append({
                    "pages": current["pages"],
                    "text": current["text"],
                    "start": current["pages"][0],
                    "end": current["pages"][-1],
                })
                current = {"pages": [], "text": "", "tokens": 0}
            current["pages"].append(pd["page_num"])
            if current["text"]:
                current["text"] += "\n\n---\n\n" + page_text
            else:
                current["text"] = page_text
            current["tokens"] = self.num_tokens(current["text"])
        if current["tokens"] > 0:
            batches.append({
                "pages": current["pages"],
                "text": current["text"],
                "start": current["pages"][0],
                "end": current["pages"][-1],
            })
        return batches

    # ------------------------------------------------------------------ #
    #  Main per-file processing
    # ------------------------------------------------------------------ #
    def process_file(self, pdf_path, output_dir):
        """Main logic to process a single PDF."""
        filename = os.path.basename(pdf_path)
        self.logger.info(f"Processing: {filename}")
        name_without_ext = os.path.splitext(filename)[0]
        file_output_dir = os.path.join(output_dir, name_without_ext)
        os.makedirs(file_output_dir, exist_ok=True)
        images_dir = os.path.join(file_output_dir, "images")
        if self.save_images:
            os.makedirs(images_dir, exist_ok=True)
        try:
            doc = fitz.open(pdf_path)
            if doc.needs_pass:
                password = self.conv_settings.get("pdf_password", "")
                if not password:
                    self.logger.error(
                        f"'{filename}' is password-protected. "
                        f"Set 'pdf_password' in config."
                    )
                    return
                if not doc.authenticate(password):
                    self.logger.error(f"Wrong password for '{filename}'.")
                    return
        except Exception as e:
            self.logger.error(f"Could not open PDF: {e}")
            return

        is_scanned = self.check_if_scanned(doc)
        if is_scanned:
            if OCR_AVAILABLE:
                self.logger.info(
                    "Scanned document detected – using OCR "
                    f"(lang={self.ocr_language})."
                )
            else:
                self.logger.warning(
                    "Scanned document detected but OCR is not available. "
                    "Install pytesseract & Pillow for scanned PDF support."
                )

        # ---- Phase 1: extract all pages ----
        self.logger.debug("Phase 1 — Extracting text and images from all pages…")
        page_data_list = []
        ocr_page_count = 0
        if self.force_ocr:
            ocr_page_count = len(doc)
        else:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()
                if len(text) < 50:
                    ocr_page_count += 1
        use_ocr_pbar = (ocr_page_count > 0 and is_scanned) or self.force_ocr
        total_images = self._count_total_images(doc) if self.save_images and self.filter_images else 0
        use_image_pbar = total_images > 0
        pbar_contexts = []
        if use_ocr_pbar:
            pbar_contexts.append(
                tqdm(total=ocr_page_count, desc="Extracting (OCR)", unit="page",
                     position=0, leave=True)
            )
        if use_image_pbar:
            pbar_contexts.append(
                tqdm(total=total_images, desc="Filtering images (LLM)", unit="img",
                     position=1 if use_ocr_pbar else 0, leave=True)
            )
        if pbar_contexts:
            ocr_pbar = pbar_contexts[0] if use_ocr_pbar else None
            image_pbar = pbar_contexts[-1] if use_image_pbar else None
            try:
                for page_num in range(len(doc)):
                    text, image_refs = self.extract_page_data(
                        doc, page_num, images_dir, ocr_pbar, image_pbar
                    )
                    page_data_list.append({
                        "page_num": page_num + 1,
                        "text": text,
                        "images": image_refs,
                    })
            finally:
                for pb in pbar_contexts:
                    pb.close()
        else:
            for page_num in range(len(doc)):
                text, image_refs = self.extract_page_data(
                    doc, page_num, images_dir
                )
                page_data_list.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "images": image_refs,
                })

        # ---- Phase 2: build batches ----
        if self.batch_pages:
            batches = self.build_batches(page_data_list)
            self.logger.info(
                f"Batched {len(page_data_list)} pages into "
                f"{len(batches)} LLM call(s)."
            )
        else:
            batches = []
            for pd in page_data_list:
                batches.append({
                    "pages": [pd["page_num"]],
                    "text": self._page_chunk_text(pd),
                    "start": pd["page_num"],
                    "end": pd["page_num"],
                })

        # ---- Phase 3: convert each batch ----
        self.logger.debug("Phase 2 — Converting batches via LLM…")
        all_markdowns = []
        seen_headings = {}
        with tqdm(total=len(batches), desc="Converting", unit="batch") as pbar:
            for batch in batches:
                page_label = (
                    f"Pages {batch['start']}-{batch['end']}"
                    if batch['start'] != batch['end']
                    else f"Page {batch['start']}"
                )
                if not batch["text"].strip():
                    md = f"<!-- {page_label}: No extractable content -->"
                else:
                    md = self.convert_to_markdown(batch["text"], page_label)
                md = self.clean_template_text(md)
                md = self.validate_and_fix(md, page_label)
                md = self.normalize_heading_hierarchy(md)
                md, seen_headings = self.deduplicate_headings(md, seen_headings)
                if self.show_page_breaks and batch['start'] != batch['end']:
                    md = f"*[Pages {batch['start']}–{batch['end']}]*\n\n" + md
                elif self.show_page_breaks:
                    md = f"*[Page {batch['start']}]*\n\n" + md
                all_markdowns.append(md)
                pbar.update(1)
        doc.close()

        # ---- Phase 4: assemble final output ----
        self.logger.debug("Phase 3 — Assembling final Markdown…")
        parts = []
        if self.generate_toc:
            toc = self.generate_table_of_contents(all_markdowns)
            if toc:
                parts.append(toc)
                self.logger.info("Table of Contents generated.")
        parts.extend(all_markdowns)
        combined = "\n\n---\n\n".join(p for p in parts if p.strip())
        if not combined or all(
                line.startswith('<!--') or not line.strip()
                for line in combined.split('\n')
        ):
            self.logger.warning(
                f"No meaningful content extracted from {filename}"
            )
            combined = "<!-- No content extracted -->"
        output_md_path = os.path.join(
            file_output_dir, f"{name_without_ext}.md"
        )
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(combined)
        self.logger.info(f"Saved to: {output_md_path}")

        # ---- Phase 5: log summary of saved images ----
        cover_count = sum(
            1 for pd in page_data_list
            if pd["images"] and any("_cover." in ref for ref in pd["images"])
        )
        self.logger.info(
            f"Done: {len(page_data_list)} pages, "
            f"{len(batches)} batches, "
            f"{cover_count} cover snapshot(s) saved."
        )

    # ------------------------------------------------------------------ #
    #  LLM conversion
    # ------------------------------------------------------------------ #
    def convert_to_markdown(self, text_chunk, page_label):
        """Sends text to LLM for Markdown conversion."""
        system_prompt = (
            "You are a strict OCR post-processor and text formatter. "
            "Your ONLY job is to convert the raw text below into clean, "
            "readable Markdown.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY the reformatted content from the source text. "
            "Nothing else.\n"
            "2. NEVER output placeholder text, template markers, bracketed "
            "instructions, or example outputs.\n"
            "3. NEVER say 'no text was provided' or generate "
            "example/template content.\n"
            "4. If the source text is minimal or unclear, output ONLY what "
            "is actually there.\n"
            "5. Re-flow the text into standard paragraphs.\n"
            "6. Detect headings and format with #, ##, ### as appropriate.\n"
            "7. Preserve tables, lists, and structured data as-is.\n"
            "8. IMPORTANT: The input text may contain image references in the format "
            "'![Image: filename](./images/filename)'. You MUST preserve these EXACTLY "
            "as they appear in the output.\n"
            "9. Do NOT add any introductory or concluding remarks."
        )
        user_prompt = (
            f"Convert the following raw text from {page_label} into "
            f"properly formatted Markdown. Reflow content into paragraphs, "
            f"mark headings clearly. The text includes image references that MUST "
            f"be preserved exactly as they appear:\n\n{text_chunk}"
        )
        try:
            result = self.call_llm_with_retry(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.conv_settings.get("max_tokens_response", 4000)
            )
            result = self.clean_template_text(result)
            if self.is_template_text(result):
                self.logger.warning(
                    f"{page_label}: LLM returned template text. Discarding."
                )
                return ""
            return result
        except Exception as e:
            self.logger.error(f"API call failed for {page_label}: {e}")
            return f"**Error converting {page_label}**"

    def validate_and_fix(self, text, page_label):
        """Simple heuristic error detection."""
        if not text or len(text.strip()) < 10:
            self.logger.warning(
                f"{page_label}: returned very little text."
            )
        return text

    # ------------------------------------------------------------------ #
    #  Entry point
    # ------------------------------------------------------------------ #
    def run(self):
        self.logger.info("--- PDF to Markdown Converter ---")
        if not OCR_AVAILABLE:
            self.logger.info(
                "OCR not available. Install pytesseract and Pillow for "
                "scanned PDF support:  pip install pytesseract Pillow"
            )
        files, output_dir = self.get_files()
        if not files:
            self.logger.error("No PDF files found.")
            return
        self.logger.info(
            f"Found {len(files)} file(s). Output: {output_dir}"
        )
        self.logger.info(
            f"Image filtering: {'ON' if self.filter_images else 'OFF'} | "
            f"Page batching: {'ON' if self.batch_pages else 'OFF'} | "
            f"TOC: {'ON' if self.generate_toc else 'OFF'} | "
            f"Max img dim: {self.img_max_dimension}px | "
            f"JPEG quality: {self.img_jpeg_quality}%"
        )
        for file_path in tqdm(files, desc="Total Files"):
            try:
                self.process_file(file_path, output_dir)
            except Exception as e:
                self.logger.error(
                    f"Failed to process {file_path}: {e}",
                    exc_info=True
                )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert PDF documents to clean Markdown using an LLM."
    )
    parser.add_argument(
        "--config", default="config.json",
        help="Path to config file (default: config.json)"
    )
    args = parser.parse_args()
    processor = PDFProcessor(config_path=args.config)
    processor.run()
