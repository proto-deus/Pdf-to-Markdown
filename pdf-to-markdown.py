import os
import json
import re
import fitz
import openai
import tiktoken
import time
import sys
import base64
from tqdm import tqdm
from pathlib import Path

CONFIG_FILE = "config.json"

class PDFProcessor:
    def __init__(self, config_path=CONFIG_FILE):
        self.load_config(config_path)
        self.setup_client()
        self.enc = tiktoken.get_encoding("cl100k_base")

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

    def setup_client(self):
        """Initializes the OpenAI-compatible client."""
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def get_files(self):
        """Locates PDF files based on input settings."""
        input_path = self.io_settings.get("input_path", ".")
        output_path = self.io_settings.get("output_path", "output")

        files_to_process = []

        if not os.path.exists(input_path):
            print(f"[ERROR] Input path does not exist: {input_path}")
            return [], output_path

        if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
            files_to_process.append(input_path)
        elif os.path.isdir(input_path):
            recursive = self.io_settings.get("recursive", True)
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        full_path = os.path.join(root, file)
                        files_to_process.append(full_path)
                if not recursive:
                    break

        return files_to_process, output_path

    def num_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def is_image_likely_irrelevant(self, rect, pix):
        """
        Fast heuristic pre-filter to skip irrelevant images before LLM call.
        """
        # 1. Geometric Check (Dimensions)
        if rect.width < 50 or rect.height < 50:
            return True
        aspect_ratio = rect.width / rect.height
        if aspect_ratio > 20 or aspect_ratio < 0.05:
            return True

        # 2. Content Check (Pixel Density)
        if pix:
            try:
                step = pix.n
                samples = pix.samples
                total_pixels = len(samples) // step

                max_samples = 10000
                pixel_stride = max(1, total_pixels // max_samples)
                stride = pixel_stride * step

                white_count = 0
                total_checked = 0
                for i in range(0, len(samples), stride):
                    r = samples[i]
                    g = samples[i+1]
                    b = samples[i+2]

                    if r > 240 and g > 240 and b > 240:
                        white_count += 1
                    total_checked += 1

                if total_checked > 0:
                    white_ratio = white_count / total_checked
                    if white_ratio > 0.95:
                        return True
            except Exception as e:
                pass
        return False

    def is_image_relevant(self, image_bytes, page_text):
        """Sends the image and text context to the LLM to check relevance."""
        if not self.filter_images:
            return True

        try:
            b64_image = base64.b64encode(image_bytes).decode('utf-8')
            context_text = page_text[:500]

            prompt = (
                "You are a strict image relevance classifier. "
                "Analyze the provided image. \n\n"
                "REJECT (return NO) and do NOT save if the image is: "
                "a Table of Contents, an Index, a simple list of page numbers, "
                "headers, footers, small icons, decorative borders, "
                "or a full page of scanned text (document page). "
                "REJECT if the image is just lines of text or a small logo next to text. \n\n"
                "ACCEPT (return YES) and save if the image is: "
                "a full-page illustration or photo (like a book cover), "
                "a complex chart, graph, diagram, map, or technical illustration. \n\n"
                f"Text Context (for reference): {context_text}\n\n"
                "Reply strictly with 'YES' or 'NO'."
            )

            response = self.client.chat.completions.create(
                model=self.model,
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
                max_tokens=5,
                temperature=0.1
            )

            result = response.choices[0].message.content.strip().upper()
            return "YES" in result

        except Exception as e:
            print(f"[WARN] Image relevance check failed (Is the model multimodal?): {e}")
            return False

    def deduplicate_headings(self, markdown_text, seen_headings):
        """
        Removes duplicate headings. If a heading text has already appeared,
        it converts it to bold text to preserve content but prevent duplicates.
        """
        lines = markdown_text.split('\n')
        new_lines = []
        for line in lines:
            # Regex to find Markdown headings (# Text)
            match = re.match(r'^(#{1,6})\s+(.*)', line)
            if match:
                level, text = match.groups()
                text_clean = text.strip()

                # Check if we have seen this exact heading text before
                if text_clean in seen_headings:
                    new_lines.append(f"**{text_clean}**")
                else:
                    seen_headings.append(text_clean)
                    new_lines.append(line)
            else:
                new_lines.append(line)

        return "\n".join(new_lines), seen_headings

    def process_file(self, pdf_path, output_dir):
        """Main logic to process a single PDF."""

        filename = os.path.basename(pdf_path)
        print(f"\nProcessing: {filename}")

        name_without_ext = os.path.splitext(filename)[0]
        file_output_dir = os.path.join(output_dir, name_without_ext)
        os.makedirs(file_output_dir, exist_ok=True)

        images_dir = os.path.join(file_output_dir, "images")
        if self.save_images:
            os.makedirs(images_dir, exist_ok=True)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"[ERROR] Could not open PDF: {e}")
            return

        full_text_markdown = []

        # Track headings to prevent duplicates across the document
        seen_headings = []

        with tqdm(total=len(doc), desc="Processing Pages", unit="pg") as pbar:
            for page_num in range(len(doc)):
                page = doc[page_num]

                # 1. Extract Text
                text = page.get_text("text")
                is_cover_page = (page_num == 0)

                # 2. Handle Images
                image_refs = []
                if self.save_images:
                    if is_cover_page:
                        try:
                            mat = fitz.Matrix(4.0, 4.0)
                            pix = page.get_pixmap(matrix=mat)
                            if page.rotation != 0:
                                pix = pix.rotate(page.rotation)

                            img_filename = f"{self.conv_settings.get('image_prefix', 'img')}_{page_num+1}_cover.jpg"
                            img_path = os.path.join(images_dir, img_filename)
                            pix.save(img_path)
                            image_refs.append(f"![Image: {img_filename}](./images/{img_filename})")
                        except Exception as e:
                            print(f"[WARN] Failed to save cover page snapshot: {e}")
                    else:
                        img_list = page.get_images(full=True)
                        for i, img in enumerate(img_list):
                            try:
                                xref = img[0]
                                rects = page.get_image_rects(xref)
                                if not rects: continue
                                rect = rects[0]

                                # Initial Geometry Check
                                if self.is_image_likely_irrelevant(rect, None):
                                    continue

                                # Render image for deeper analysis
                                mat = fitz.Matrix(4.0, 4.0)
                                pix = page.get_pixmap(matrix=mat, clip=rect)
                                if page.rotation != 0:
                                    pix = pix.rotate(page.rotation)

                                # Pixel Density Check (Content Analysis)
                                if self.is_image_likely_irrelevant(rect, pix):
                                    continue

                                # LLM Relevance Check
                                img_bytes = pix.tobytes("jpeg")
                                if self.is_image_relevant(img_bytes, text):
                                    img_filename = f"{self.conv_settings.get('image_prefix', 'img')}_{page_num+1}_{i}.jpg"
                                    img_path = os.path.join(images_dir, img_filename)
                                    pix.save(img_path)
                                    image_refs.append(f"![Image: {img_filename}](./images/{img_filename})")
                            except Exception as img_err:
                                print(f"[WARN] Error processing image {i} on page {page_num+1}: {img_err}")

                # 3. Chunking Logic
                current_chunk_text = text
                if image_refs:
                    current_chunk_text += "\n\n" + "\n".join(image_refs)

                tokens = self.num_tokens(current_chunk_text)

                if tokens > self.max_tokens:
                    print(f"[WARN] Page {page_num+1} is too large for context ({tokens} tokens). Truncating.")
                    current_chunk_text = current_chunk_text[:int(len(current_chunk_text) * (self.max_tokens/tokens))]

                # 4. Call LLM for conversion
                markdown_content = self.convert_to_markdown(current_chunk_text, page_num + 1)

                # 5. Error/Quality Check
                markdown_content = self.validate_and_fix(markdown_content, page_num + 1)

                # 6. Remove duplicate headings found in previous pages
                markdown_content, seen_headings = self.deduplicate_headings(markdown_content, seen_headings)

                full_text_markdown.append(markdown_content)
                pbar.update(1)

        # Save output
        output_md_path = os.path.join(file_output_dir, f"{name_without_ext}.md")
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(full_text_markdown))

        print(f"[SUCCESS] Saved to: {output_md_path}")

    def convert_to_markdown(self, text_chunk, page_num):
        """Sends text to LLM for conversion."""
        system_prompt = (
            "You are a strict OCR post-processor and formatter. "
            "Your task is to convert the raw extracted text into clean, readable Markdown. "
            "IMPORTANT: The source text may be extracted from a PDF with columns, tables, or unusual layout. "
            "You must RE-FLOW the text into a standard reading format (normal paragraphs) unless the text is clearly a table, list, or heading. "
            "Detect headings in the source text. "
            "If a line is short, all caps, or distinct from body text, format it as a Markdown header (#, ##, ###). "
            "Ignore the original PDF column layout. Combine fragmented lines into cohesive paragraphs, but preserve heading hierarchy. "
            "Do NOT add any introductory or concluding text. "
            "Output ONLY the formatted Markdown content."
        )

        user_prompt = (
            f"Convert the following raw text from Page {page_num} into properly formatted Markdown. "
            "Reflow the content into standard paragraphs and ensure headings are properly marked:\n\n"
            f"{text_chunk}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.conv_settings.get("temperature", 0.1),
                max_tokens=self.conv_settings.get("max_tokens_response", 4000)
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] API Call failed for page {page_num}: {e}")
            return f"**Error converting page {page_num}**"

    def validate_and_fix(self, text, page_num):
        """Simple heuristic error detection."""
        if not text or len(text) < 10:
            print(f"[WARN] Page {page_num} returned very little text.")
        return text

    def run(self):
        print("--- PDF to Markdown Converter ---")
        files, output_dir = self.get_files()

        if not files:
            print("[ERROR] No PDF files found.")
            return

        print(f"Found {len(files)} file(s). Output: {output_dir}")
        print(f"Image Filtering Enabled: {self.filter_images}")

        for file_path in tqdm(files, desc="Total Files"):
            try:
                self.process_file(file_path, output_dir)
            except Exception as e:
                print(f"[FATAL] Failed to process {file_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="Path to config file")
    args = parser.parse_args()

    processor = PDFProcessor(config_path=args.config)
    processor.run()
