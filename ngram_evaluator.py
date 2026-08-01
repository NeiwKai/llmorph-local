import json
import re
import os
import difflib
from datetime import datetime
from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Define Stopwords List
# Common functional words to exclude when filtering boundary noise.
# ----------------------------------------------------
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'as', 'at', 'by', 'for', 'from', 
    'in', 'into', 'of', 'on', 'to', 'with', 'is', 'are', 'was', 'were', 
    'be', 'been', 'being', 'have', 'has', 'had', 'such', 'this', 'that',
    'what', 'where', 'when', 'why', 'how', 'which', 'who', 'whom', 'whose',
    'can', 'could', 'would', 'should', 'do', 'does', 'did', 'may', 'might'
}

def clean_and_tokenize_by_clauses(text: str) -> List[List[str]]:
    """
    Splits text into clauses by commas, semicolons, or sentence delimiters.
    Preserves original casing and hyphens (-) within words.
    """
    clauses = re.split(r'[,;.\n]+', text)
    clause_tokens = []
    for clause in clauses:
        # Regex \b[\w-]+\b keeps hyphenated words intact and preserves original casing
        tokens = re.findall(r'\b[\w-]+\b', clause)
        if tokens:
            clause_tokens.append(tokens)
    return clause_tokens

def trim_stopwords(phrase_tokens: List[str]) -> List[str]:
    """
    Trims leading and trailing stopwords (including WH-questions).
    Checks stopwords in lower-case while retaining original casing of the output tokens.
    """
    tokens = list(phrase_tokens)
    # Trim leading stopwords
    while tokens and tokens[0].lower() in STOPWORDS:
        tokens.pop(0)
    # Trim trailing stopwords
    while tokens and tokens[-1].lower() in STOPWORDS:
        tokens.pop()
    return tokens

def extract_transformations_from_clause(tokens1: List[str], tokens2: List[str], gram_level: int) -> List[Tuple[str, str]]:
    """
    Extracts transformed phrase pairs strictly within a single clause boundaries.
    """
    matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
    pairs = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ['replace', 'delete', 'insert']:
            if gram_level == 1:
                if tag == 'replace':
                    orig_slice = tokens1[i1:i2]
                    trans_slice = tokens2[j1:j2]
                    min_len = min(len(orig_slice), len(trans_slice))
                    for k in range(min_len):
                        w1, w2 = orig_slice[k], trans_slice[k]
                        if w1 != w2 and w1 not in STOPWORDS and w2 not in STOPWORDS:
                            pair = (w1, w2)
                            if pair not in pairs:
                                pairs.append(pair)
            else:
                target_window = gram_level
                
                # Expand original context window within clause
                orig_len = i2 - i1
                pad_i = max(0, target_window - orig_len)
                start_i = max(0, i1 - (pad_i // 2 + pad_i % 2))
                end_i = min(len(tokens1), i2 + (pad_i // 2))
                
                if end_i - start_i < target_window and start_i > 0:
                    start_i = max(0, end_i - target_window)
                if end_i - start_i < target_window and end_i < len(tokens1):
                    end_i = min(len(tokens1), start_i + target_window)

                # Expand transformed context window within clause
                trans_len = j2 - j1
                pad_j = max(0, target_window - trans_len)
                start_j = max(0, j1 - (pad_j // 2 + pad_j % 2))
                end_j = min(len(tokens2), j2 + (pad_j // 2))
                
                if end_j - start_j < target_window and start_j > 0:
                    start_j = max(0, end_j - target_window)
                if end_j - start_j < target_window and end_j < len(tokens2):
                    end_j = min(len(tokens2), start_j + target_window)

                orig_slice = tokens1[start_i:end_i]
                trans_slice = tokens2[start_j:end_j]

                trimmed_orig = trim_stopwords(orig_slice)
                trimmed_trans = trim_stopwords(trans_slice)

                p1 = " ".join(trimmed_orig).strip()
                p2 = " ".join(trimmed_trans).strip()

                if p1 and p2 and p1 != p2:
                    if p1 not in STOPWORDS or p2 not in STOPWORDS:
                        pair = (p1, p2)
                        if pair not in pairs:
                            pairs.append(pair)

    return pairs

def plot_metric_table(metric_list: List[dict], mode: str, threshold: float, precision: float, task_name: str, date_time_str: str, gram_level: int):
    """
    Renders and saves a high-quality table image displaying N-Gram evaluation metrics.
    Includes the N-Gram level in the table title.
    """
    metric_list.sort(key=lambda x: x['cosine_similarity'], reverse=True)

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

    fig, ax = plt.subplots(figsize=(13, max(len(table_data) * 0.55 + 2, 4)))
    ax.axis('off')
    
    # Title header explicitly stating the N-Gram level
    title_text = f"N-Gram Level: {gram_level}-Gram | Mode: {mode} Mode\n" \
                 f"Task: {task_name} | Precision: {precision * 100:.2f}% | Threshold: {threshold}"
    plt.title(title_text, fontsize=13, fontweight='bold', pad=20)

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

    plt.tight_layout()

    output_dir = f"plot-{gram_level}gram"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{task_name}_{date_time_str}_{gram_level}gram_table.png"
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Saved cleanly formatted {gram_level}-Gram table plot to '{plot_path}'")
    plt.show()

# ----------------------------------------------------
# MAIN EXECUTION PIPELINE
# ----------------------------------------------------
def main():
    file_path = input("Enter JSON file path (e.g., data.json): ").strip()
    
    print("\nSelect N-Gram level:")
    print("1: 1-Gram (Word-to-Word)")
    print("2: 2-Gram (Bi-gram Context)")
    print("3: 3-Gram (Tri-gram Context)")
    gram_choice = input("Enter choice (1, 2, or 3): ").strip()
    gram_level = int(gram_choice) if gram_choice in ['1', '2', '3'] else 1

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

    # Tokenize input texts into clauses using commas/periods as separators
    clauses1 = clean_and_tokenize_by_clauses(text1_raw)
    clauses2 = clean_and_tokenize_by_clauses(text2_raw)

    # Extract transformed phrase pairs clause by clause
    aligned_pairs = []
    min_clauses = min(len(clauses1), len(clauses2))
    for idx in range(min_clauses):
        pairs = extract_transformations_from_clause(clauses1[idx], clauses2[idx], gram_level)
        for p in pairs:
            if p not in aligned_pairs:
                aligned_pairs.append(p)

    if not aligned_pairs:
        print(f"\n⚠️ No {gram_level}-Gram transformations found matching criteria.")
        return

    print(f"\nLoading Transformer Embedding Model (all-mpnet-base-v2)...")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    orig_texts = [p[0] for p in aligned_pairs]
    mod_texts = [p[1] for p in aligned_pairs]

    embeddings_orig = model.encode(orig_texts, convert_to_tensor=True)
    embeddings_mod = model.encode(mod_texts, convert_to_tensor=True)

    cosine_sims = torch.nn.functional.cosine_similarity(embeddings_orig, embeddings_mod, dim=1)
    
    metric_list = []
    for i in range(len(aligned_pairs)):
        sim_score = float(cosine_sims[i])
        
        is_valid = (mode == "Synonym" and sim_score >= threshold) or \
                   (mode == "Antonym" and sim_score <= threshold)
                   
        metric_list.append({
            "original": orig_texts[i],
            "transformed": mod_texts[i],
            "cosine_similarity": round(sim_score, 4),
            "pass_threshold": is_valid
        })

    valid_changes = sum(1 for m in metric_list if m['pass_threshold'])
    precision = (valid_changes / len(metric_list)) if metric_list else 0.0

    print(f"\nSelected Level: {gram_level}-Gram | Target Mode: {mode}")
    print(f"Valid Changes Passed Threshold: {valid_changes} / {len(metric_list)}")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")

    match_dt = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file_path)
    date_time_str = match_dt.group(0) if match_dt else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plot_metric_table(metric_list, mode, threshold, precision, task_name, date_time_str, gram_level)

if __name__ == "__main__":
    main()
