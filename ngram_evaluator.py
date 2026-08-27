import difflib
import json
import os
import re
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import spacy

# Load spaCy NLP model with static vector support
nlp = spacy.load("en_core_web_md")

# ----------------------------------------------------
# 1. TEXT TOKENIZATION (Using spaCy Doc)
# ----------------------------------------------------
def tokenize_with_spacy(text: str) -> List[str]:
    """
    Tokenizes raw text into clean tokens using spaCy,
    filtering out standalone punctuation and whitespace.
    """
    doc = nlp(text)
    return [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

# ----------------------------------------------------
# 2. EXTRACT N-GRAM TRANSFORMATIONS (Using spaCy Spans)
# ----------------------------------------------------
def extract_ngram_transformations(tokens1: List[str], tokens2: List[str], n: int) -> List[Tuple[str, str]]:
    """
    Finds difference boundaries between source and target tokens,
    then uses spaCy Doc indexing to extract cleanly windowed n-gram spans.
    """
    doc1 = spacy.tokens.Doc(nlp.vocab, words=tokens1)
    doc2 = spacy.tokens.Doc(nlp.vocab, words=tokens2)

    matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
    pairs = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ['replace', 'delete', 'insert']:
            if n == 1:
                # 1-Gram direct token replacement
                if tag == 'replace':
                    orig_len = i2 - i1
                    trans_len = j2 - j1
                    min_len = min(orig_len, trans_len)
                    for k in range(min_len):
                        w1 = doc1[i1 + k].text
                        w2 = doc2[j1 + k].text
                        if w1 != w2:
                            pair = (w1, w2)
                            if pair not in pairs:
                                pairs.append(pair)
            else:
                # Window expansion for context-level N-Grams
                orig_len = i2 - i1
                pad_i = max(0, n - orig_len)
                start_i = max(0, i1 - (pad_i // 2 + pad_i % 2))
                end_i = min(len(doc1), i2 + (pad_i // 2))
                if end_i - start_i < n and start_i > 0:
                    start_i = max(0, end_i - n)
                if end_i - start_i < n and end_i < len(doc1):
                    end_i = min(len(doc1), start_i + n)

                trans_len = j2 - j1
                pad_j = max(0, n - trans_len)
                start_j = max(0, j1 - (pad_j // 2 + pad_j % 2))
                end_j = min(len(doc2), j2 + (pad_j // 2))
                if end_j - start_j < n and start_j > 0:
                    start_j = max(0, end_j - n)
                if end_j - start_j < n and end_j < len(doc2):
                    end_j = min(len(doc2), start_j + n)

                # Extract as spaCy Spans and convert to string
                span1 = doc1[start_i:end_i].text.strip()
                span2 = doc2[start_j:end_j].text.strip()

                if span1 and span2 and span1 != span2:
                    pair = (span1, span2)
                    if pair not in pairs:
                        pairs.append(pair)

    return pairs

# ----------------------------------------------------
# 3. VISUALIZATION TABLE PLOTTER
# ----------------------------------------------------
def plot_metric_table(
    metric_list: List[dict],
    mode: str,
    threshold: float,
    precision: float,
    task_name: str,
    date_time_str: str,
    gram_level: int
):
    # Sort: PASS items first, then by Cosine Similarity descending
    if mode == "Synonym":
        metric_list.sort(key=lambda x: (not x['pass_threshold'], -x['cosine_similarity']))
    else:
        metric_list.sort(key=lambda x: (not x['pass_threshold'], x['cosine_similarity']))
    headers = ["Status", "Original Phrase", "Transformed Phrase", "Cosine Sim"]
    table_data = []
    
    for item in metric_list:
        status_text = "PASS" if item['pass_threshold'] else "FAIL"
        table_data.append([
            status_text,
            item['original'],
            item['transformed'],
            f"{item['cosine_similarity']:.4f}"
        ])

    fig_height = max(len(table_data) * 0.5 + 3.0, 5.0)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('off')
    
    threshold_str = f">= {threshold}" if mode == "Synonym" else f"<= {threshold}"
    title_text = f"N-Gram Level: {gram_level}-Gram | Mode: {mode} Mode (spaCy)\n" \
                 f"Task: {task_name} | Precision: {precision * 100:.2f}% | Pass Threshold: {threshold_str}"
    plt.title(title_text, fontsize=13, fontweight='bold', pad=15)

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2B4C7E')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            if col == 0:
                if table_data[row - 1][0] == "PASS":
                    cell.set_facecolor('#D4EDDA')
                    cell.set_text_props(color='#155724', fontweight='bold')
                else:
                    cell.set_facecolor('#F8D7DA')
                    cell.set_text_props(color='#721C24', fontweight='bold')
            else:
                cell.set_facecolor('#F8F9FA' if row % 2 == 0 else '#FFFFFF')

    plt.subplots_adjust(top=0.82, bottom=0.05, left=0.05, right=0.95)

    output_dir = os.path.join("plots", mode.capitalize(), f"plot-{gram_level}gram")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_filename = f"{task_name}_{mode.lower()}_{date_time_str}_{gram_level}gram_table.png"
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved table plot to '{plot_path}'")
    plt.show()

# ----------------------------------------------------
# 4. MAIN EXECUTION PIPELINE
# ----------------------------------------------------
def main():
    file_path = input("Enter JSON file path (e.g., data.json): ").strip()

    if not os.path.exists(file_path):
        print(f"\n[Error] File not found: '{file_path}'")
        return
    
    print("\nSelect N-Gram level:")
    print("1: 1-Gram (Word-to-Word)")
    print("2: 2-Gram (Bi-gram Context)")
    print("3: 3-Gram (Tri-gram Context)")
    gram_choice = input("Enter choice (or type any integer): ").strip()
    try:
        gram_level = int(gram_choice) if int(gram_choice) >= 1 else 1
    except ValueError:
        gram_level = 1

    print("\nSelect evaluation mode:")
    print("1: Synonym (Cosine >= 0.8)")
    print("2: Antonym (Cosine <= -0.8)")
    mode_choice = input("Enter choice (1 or 2): ").strip()

    if mode_choice == '2':
        mode = "Antonym"
        threshold = -0.8
    else:
        mode = "Synonym"
        threshold = 0.8

    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    task_name = json_data.get('task_name', 'question_answering')

    text1_list = json_data['data'][0]['source_input']
    text1_raw = " ".join(text1_list)
    
    text2_list = json_data['data'][0]['followup_inputs'][2]
    text2_raw = " ".join(text2_list)

    tokens1 = tokenize_with_spacy(text1_raw)
    tokens2 = tokenize_with_spacy(text2_raw)

    aligned_pairs = extract_ngram_transformations(tokens1, tokens2, gram_level)

    if not aligned_pairs:
        print(f"\nNo {gram_level}-Gram transformations found matching criteria.")
        return

    print(f"\nCalculating similarity for {len(aligned_pairs)} pairs using spaCy vectors...")
    metric_list = []
    
    for orig_text, mod_text in aligned_pairs:
        doc1 = nlp(orig_text)
        doc2 = nlp(mod_text)
        
        # spaCy computes cosine similarity between vector representations
        if doc1.vector_norm and doc2.vector_norm:
            sim_score = float(doc1.similarity(doc2))
        else:
            sim_score = 0.0

        # Flip sign for Antonym mode
        if mode == "Antonym":
            sim_score = -sim_score

        if mode == "Synonym":
            is_valid = (sim_score >= threshold)
        else:
            is_valid = (sim_score <= threshold)
                    
        metric_list.append({
            "original": orig_text,
            "transformed": mod_text,
            "cosine_similarity": round(sim_score, 4),
            "pass_threshold": is_valid
        })

    valid_changes = sum(1 for m in metric_list if m['pass_threshold'])
    precision = (valid_changes / len(metric_list)) if metric_list else 0.0

    print(f"\nSelected Level: {gram_level}-Gram | Target Mode: {mode}")
    print(f"Condition: {'Cosine >= ' + str(threshold) if mode == 'Synonym' else 'Cosine <= ' + str(threshold)}")
    print(f"Valid Changes Passed Threshold: {valid_changes} / {len(metric_list)}")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")

    match_dt = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file_path)
    date_time_str = match_dt.group(0) if match_dt else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plot_metric_table(metric_list, mode, threshold, precision, task_name, date_time_str, gram_level)

if __name__ == "__main__":
    main()
