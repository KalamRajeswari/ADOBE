import fitz  # PyMuPDF
import json
import re
import spacy
import traceback
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

class HeadingExtractor:
    def __init__(self):
        self.heading_keywords = [
            "introduction", "abstract", "conclusion", "references",
            "methodology", "results", "discussion", "background",
            "chapter", "section", "appendix"
        ]
        self.min_font_occurrences = 3
        self.max_heading_words = 12

    def extract_from_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            if toc and len(toc) > 2:
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
                    "page": page
                })
        if not title and outline:
            title = outline[0]["text"]
        return {
            "title": title,
            "outline": sorted(outline, key=lambda x: x["page"])  # Ensure page-order
        }

    def _extract_with_nlp(self, doc):
        font_stats = self._analyze_fonts(doc)
        font_stats["spans"] = self._merge_close_spans(font_stats["spans"])
        candidates = self._extract_candidate_headings(doc, font_stats)
        headings = self._classify_headings_with_nlp(candidates)
        outline = self._assign_heading_levels(headings)
        outline = self._deduplicate_outline(outline)

        # ✅ Sort the outline by page number
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
                            "page": page_num + 1,
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
                "has_heading_keyword": any(keyword in text.lower() for keyword in self.heading_keywords),
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
                    "is_bold": is_bold
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
        if features["has_heading_keyword"]:
            score += 0.2
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
        for i, (candidate, doc) in enumerate(zip(candidates, docs)):
            has_verb = any(token.pos_ == "VERB" for token in doc)
            pos_pattern = " ".join([token.pos_ for token in doc])
            common_heading_patterns = [
                "^(DET )?(ADJ )*(NOUN|PROPN)",
                "^NUM",
                "^(VERB|AUX)",
                "^ADV ADJ"
            ]
            matches_pattern = any(re.search(pattern, pos_pattern) for pattern in common_heading_patterns)
            if matches_pattern:
                candidates[i]["score"] += 0.2
            if has_verb and len(doc) > 5:
                candidates[i]["score"] -= 0.25
            if '.' in candidate["text"] or len(candidate["text"].split()) > 12:
                candidates[i]["score"] -= 0.3
        return [c for c in candidates if c["score"] >= 0.6]

    def _assign_heading_levels(self, headings):
        if not headings:
            return []

    # Sort headings by font size descending
        sorted_headings = sorted(headings, key=lambda x: -x["size"])
        size_counts = defaultdict(int)
        for h in sorted_headings:
            size_counts[h["size"]] += 1

    # Sort font sizes by frequency and size
        most_common_sizes = sorted(size_counts.items(), key=lambda x: (-x[0], -x[1]))
        unique_sizes = sorted({size for size, _ in most_common_sizes}, reverse=True)

        size_to_level = {}
        for i, size in enumerate(unique_sizes[:3]):
            size_to_level[size] = f"H{i+1}"
        
        default_level = "H3"
        outline = []

        for heading in headings:
            size = heading["size"]
            level = size_to_level.get(size, default_level)

        # Optional: boost precision by being selective
            if heading["score"] < 0.65 and level == "H3":
                continue

            outline.append({
            "level": level,
            "text": heading["text"].strip(),
            "page": heading["page"]
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

def main():
    pdf_path = "file02.pdf"
    json_path = "output1.json"

    extractor = HeadingExtractor()
    result = extractor.extract_from_pdf(pdf_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Extraction complete. Output written to {json_path}")

if __name__ == "__main__":
    main()

