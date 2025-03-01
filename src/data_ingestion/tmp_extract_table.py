"""
Author: Zhengyuan Dong
Created: 2025-02-23
Last Modified: 2025-02-23
Description: Extract tables & figures reference text / captions from s2orc dumped data and save to JSON.
"""

import json

# get first item of the file
input_file = "step29_file"
output_file = "small_sample.json"
with open(input_file, "r", encoding="utf-8") as f:
    first_line = f.readline().strip()
with open(output_file, "w", encoding="utf-8") as f:
    f.write(first_line)


# extract table/figure/figure captions from the sample
input_file = "small_sample.json"
output_file = "small_sample_with_extracted_data.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)
if "content" in data and "text" in data["content"]:
    content_text = data["content"]["text"]
    tableref_raw = data["content"].get("annotations", {}).get("tableref", "[]")
    try:
        tablerefs = json.loads(tableref_raw)
    except json.JSONDecodeError:
        tablerefs = []
    extracted_tables = []
    for tableref in tablerefs:
        start = tableref.get("start")
        end = tableref.get("end")
        if isinstance(start, int) and isinstance(end, int) and start < end:
            extracted_text = content_text[start:end]
            extracted_tables.append({
                "tableref": tableref,
                "extracted_text": extracted_text
            })
    figures_raw = data["content"].get("annotations", {}).get("figure", "[]")
    try:
        figures = json.loads(figures_raw)
    except json.JSONDecodeError:
        figures = []
    extracted_figures = []
    for figure in figures:
        start = figure.get("start")
        end = figure.get("end")
        fig_id = figure.get("attributes", {}).get("id", "unknown")
        if isinstance(start, int) and isinstance(end, int) and start < end:
            extracted_text = content_text[start:end]
            extracted_figures.append({
                "figure_id": fig_id,
                "start": start,
                "end": end,
                "extracted_text": extracted_text
            })
    figurecaptions_raw = data["content"].get("annotations", {}).get("figurecaption", "[]")
    try:
        figurecaptions = json.loads(figurecaptions_raw)
    except json.JSONDecodeError:
        figurecaptions = []
    extracted_figure_captions = []
    for caption in figurecaptions:
        start = caption.get("start")
        end = caption.get("end")
        if isinstance(start, int) and isinstance(end, int) and start < end:
            extracted_caption = content_text[start:end]
            extracted_figure_captions.append({
                "start": start,
                "end": end,
                "caption_text": extracted_caption
            })
    figurerefs_raw = data["content"].get("annotations", {}).get("figureref", "[]")
    try:
        figurerefs = json.loads(figurerefs_raw)
    except json.JSONDecodeError:
        figurerefs = []
    extracted_figurerefs = []
    for figureref in figurerefs:
        start = figureref.get("start")
        end = figureref.get("end")
        ref_id = figureref.get("attributes", {}).get("ref_id", "unknown")
        if isinstance(start, int) and isinstance(end, int) and start < end:
            extracted_text = content_text[start:end]
            extracted_figurerefs.append({
                "figure_ref_id": ref_id,
                "start": start,
                "end": end,
                "extracted_text": extracted_text
            })
    extracted_data = {}
    if extracted_tables:
        extracted_data["extracted_tables"] = extracted_tables
    if extracted_figures:
        extracted_data["extracted_figures"] = extracted_figures
    if extracted_figure_captions:
        extracted_data["extracted_figure_captions"] = extracted_figure_captions
    if extracted_figurerefs:
        extracted_data["extracted_figurerefs"] = extracted_figurerefs
    data.update(extracted_data)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
print(f"Extracted data saved to {output_file}")
