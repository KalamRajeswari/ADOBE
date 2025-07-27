import fitz  # PyMuPDF
import json
import re
import spacy
import os
from collections import defaultdict
from datetime import datetime

nlp = spacy.load("en_core_web_sm")

class HeadingExtractor:
    def __init__(self):
        self.min_font_occurrences = 3
        self.max_heading_words = 12

    def extract_from_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            if toc and len(toc) > 2:
                # Use TOC if available
                return self._extract_from_toc(doc, toc)
            return self._extract_with_nlp(doc)
        except Exception as e:
            print(f"Error extracting from PDF: {e}")
            return {"title": "Unknown", "outline": []}

    def _extract_from_toc(self, doc, toc):
        title = doc.metadata.get("title", "")
        outline = []
        for item in toc:
            level, heading_text, page = item
            if level <= 3:
                outline.append({
                    "level": f"H{level}",
                    "text": heading_text,
                    "page": page  # usually 1-based from TOC
                })
        if not title and outline:
            title = outline[0]["text"]
        return {
            "title": title,
            "outline": sorted(outline, key=lambda x: x["page"])
        }

    def _extract_with_nlp(self, doc):
        font_stats = self._analyze_fonts(doc)
        font_stats["spans"] = self._merge_close_spans(font_stats["spans"])
        candidates = self._extract_candidate_headings(doc, font_stats)
        headings = self._classify_headings_with_nlp(candidates)
        outline = self._assign_heading_levels(headings)
        outline = self._deduplicate_outline(outline)

        # Convert zero-based page index to 1-based
        for item in outline:
            item["page"] += 1

        outline.sort(key=lambda x: x["page"])

        title = doc.metadata.get("title", "")
        if not title and outline:
            title = outline[0]["text"]
        return {
            "title": title,
            "outline": outline
        }

    def _analyze_fonts(self, doc):
        fonts = defaultdict(int)
        font_details = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = round(span.get("size", 0), 1)
                        is_bold = span.get("flags", 0) & 2 > 0
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        fonts[size] += 1
                        font_details.append({
                            "text": text,
                            "size": size,
                            "bold": is_bold,
                            "page": page_num,
                            "y_pos": span.get("bbox")[1],
                            "bbox": span.get("bbox")
                        })
        common_fonts = sorted(
            [(size, count) for size, count in fonts.items()
             if count >= self.min_font_occurrences],
            key=lambda x: x[0], reverse=True
        )
        return {"common_fonts": common_fonts, "spans": font_details}

    def _merge_close_spans(self, spans, y_threshold=1.5):
        merged = []
        current = None
        for span in sorted(spans, key=lambda x: (x['page'], x['y_pos'])):
            if current and abs(current['y_pos'] - span['y_pos']) < y_threshold and current['page'] == span['page']:
                current['text'] += ' ' + span['text']
            else:
                if current:
                    merged.append(current)
                current = span.copy()
        if current:
            merged.append(current)
        return merged

    def _extract_candidate_headings(self, doc, font_stats):
        candidates = []
        common_fonts = font_stats["common_fonts"]
        spans = font_stats["spans"]
        spans_by_page = defaultdict(list)
        for span in spans:
            spans_by_page[span["page"]].append(span)
        for span in spans:
            text = span["text"]
            size = span["size"]
            is_bold = span["bold"]
            page = span["page"]
            if len(text.split()) > self.max_heading_words:
                continue

            features = {
                "font_size": size,
                "is_bold": is_bold,
                "word_count": len(text.split()),
                "has_number_prefix": bool(re.match(r'^\d+(\.\d+)*\.?\s', text)),
                "is_all_caps": text.isupper(),
                "ends_with_colon": text.endswith(':'),
                "at_page_top": self._is_at_page_top(span, spans_by_page[page]),
                "standalone_line": self._is_standalone(span, spans_by_page[page])
            }
            score = self._score_candidate(features, common_fonts)
            if score > 0.5:
                candidates.append({
                    "text": text,
                    "page": page,
                    "features": features,
                    "score": score,
                    "size": size,
                    "is_bold": is_bold,
                    "y_pos": span["y_pos"],
                    "bbox": span["bbox"],
                })
        return candidates

    def _is_at_page_top(self, span, page_spans):
        if not page_spans:
            return False
        sorted_spans = sorted(page_spans, key=lambda x: x["y_pos"])
        return span["y_pos"] <= sorted_spans[0]["y_pos"] + 0.15 * (sorted_spans[-1]["y_pos"] - sorted_spans[0]["y_pos"])

    def _is_standalone(self, span, page_spans):
        bbox = span["bbox"]
        for other in page_spans:
            if other == span:
                continue
            y_overlap = max(0, min(bbox[3], other["bbox"][3]) - max(bbox[1], other["bbox"][1]))
            if y_overlap > 0:
                return False
        return True

    def _score_candidate(self, features, common_fonts):
        score = 0
        for i, (font_size, _) in enumerate(common_fonts[:3]):
            if abs(features["font_size"] - font_size) < 0.5:
                score += 0.3 - (i * 0.1)
                break
        if features["is_bold"]:
            score += 0.2
        if features["has_number_prefix"]:
            score += 0.3
        if features["is_all_caps"]:
            score += 0.15
        if features["ends_with_colon"]:
            score += 0.15
        # No keyword-based scoring here
        if features["at_page_top"]:
            score += 0.25
        if features["standalone_line"]:
            score += 0.2
        if features["word_count"] > 8:
            score -= 0.1 * (features["word_count"] - 8)
        return min(1.0, max(0.0, score))

    def _classify_headings_with_nlp(self, candidates):
        if not candidates:
            return []
        texts = [c["text"] for c in candidates]
        docs = list(nlp.pipe(texts, disable=["ner"]))
        filtered = []
        for i, (candidate, doc) in enumerate(zip(candidates, docs)):
            text = candidate["text"]
        # Must start uppercase letter
            if not text[0].isupper():
                continue
        # Word count limits
            wc = len(text.split())
            if wc < 3 or wc > 12:
                continue
            has_verb = any(token.pos_ == "VERB" for token in doc)
        # Penalize if too long or contains verbs strongly
            if has_verb and wc > 5:
                continue
            if '.' in text:
                continue
            score = candidate["score"]
        # Increase score if phrase-like pattern matched
            pos_pattern = " ".join([token.pos_ for token in doc])
            common_heading_patterns = [
            "^(DET )?(ADJ )*(NOUN|PROPN)",  # noun phrase start
            "^NUM",
            "^ADV ADJ"
        ]
            matches_pattern = any(re.search(pattern, pos_pattern) for pattern in common_heading_patterns)
            if matches_pattern:
                score += 0.2
            if score >= 0.7:
                candidate["score"] = score
                filtered.append(candidate)
        return filtered


    def _assign_heading_levels(self, headings):
        if not headings:
            return []
        sorted_headings = sorted(headings, key=lambda x: x["score"], reverse=True)
        sizes = [h["size"] for h in sorted_headings]
        size_counts = defaultdict(int)
        for size in sizes:
            size_counts[size] += 1
        common_sizes = sorted(size_counts.items(), key=lambda x: x[1], reverse=True)
        size_to_level = {}
        for i, (size, _) in enumerate(common_sizes[:3]):
            size_to_level[size] = f"H{i+1}"
        default_level = "H3"
        outline = []
        for heading in sorted_headings:
            level = size_to_level.get(heading["size"], default_level)
            if heading["score"] < 0.75 and level == "H3":
                continue
            outline.append({
                "level": level,
                "text": heading["text"],
                "page": heading["page"],
                "y_pos": heading["y_pos"]
            })
        return outline

    def _deduplicate_outline(self, outline):
        seen = set()
        result = []
        for item in outline:
            key = (item["text"].strip().lower(), item["page"])
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result


def extract_refined_text(pdf_path, page_number, heading_y_pos, max_sentences=3):
    """Extract up to max_sentences sentences below heading on the page."""
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]  # zero-based
    blocks = page.get_text("dict")["blocks"]

    candidate_blocks = [b for b in blocks if b.get("bbox") and b["bbox"][1] > heading_y_pos + 2]
    candidate_blocks = sorted(candidate_blocks, key=lambda b: b["bbox"][1])

    accumulated_text = ""
    sentence_count = 0

    for block in candidate_blocks:
        block_text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                block_text += span.get("text", "") + " "
        block_text = block_text.strip()

        sentences = re.split(r'(?<=[.!?])\s+', block_text)

        for sent in sentences:
            if sentence_count < max_sentences:
                accumulated_text += sent + " "
                sentence_count += 1
            else:
                break

        if sentence_count >= max_sentences:
            break

    accumulated_text = accumulated_text.strip()
    if sentence_count >= max_sentences:
        accumulated_text += "..."

    return accumulated_text if accumulated_text else ""



def main():
    pdf_folder = "input_pdfs"  # Your PDFs folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    persona = "Travel Planner"
    job_to_be_done = "Plan a trip of 4 days for a group of 10 college friends."
    processing_timestamp = datetime.now().isoformat()

    extractor = HeadingExtractor()
    all_sections = []

    for pdf_file in pdf_files:
        full_path = os.path.join(pdf_folder, pdf_file)
        result = extractor.extract_from_pdf(full_path)
        for section in result.get("outline", []):
            all_sections.append({
                "document": pdf_file,
                "section_title": section["text"],
                "page_number": section["page"],  # Already 1-based
                "y_pos": section.get("y_pos", 0)
            })

    filtered_sections = [s for s in all_sections if not is_generic_heading(s["section_title"])]

    if len(filtered_sections) < 10:
        intro_sections = [s for s in all_sections if is_generic_heading(s["section_title"])]
        intro_by_doc = {}
        for s in intro_sections:
            if s["document"] not in intro_by_doc:
                intro_by_doc[s["document"]] = s
        filtered_sections.extend(intro_by_doc.values())

    all_sections = filtered_sections

    all_sections.sort(key=lambda s: (section_score(s), -s["page_number"]), reverse=True)

    seen_titles = set()
    final_sections = []
    for s in all_sections:
        title_lower = s["section_title"].strip().lower()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            final_sections.append(s)
        if len(final_sections) >= 10:   # <-- Changed from 5 to 10 here
            break

    final_subsections = []
    for s in final_sections:
        refined_text = extract_refined_text(
            os.path.join(pdf_folder, s["document"]),
            s["page_number"],
            s.get("y_pos", 0)
        )
        final_subsections.append({
            "document": s["document"],
            "refined_text": refined_text,
            "page_number": s["page_number"]
        })

    output = {
        "metadata": {
            "input_documents": pdf_files,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": processing_timestamp
        },
        "extracted_sections": [
            {
                "document": s["document"],
                "section_title": s["section_title"],
                "importance_rank": i + 1,
                "page_number": s["page_number"]
            } for i, s in enumerate(final_sections)
        ],
        "subsection_analysis": final_subsections
    }

    with open("output_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print("✅ Extraction complete. Output saved to output_result.json")


if __name__ == "__main__":
    main()



   
