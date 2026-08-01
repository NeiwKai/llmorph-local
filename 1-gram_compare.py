import json
import re
import os
import difflib
from datetime import datetime
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Define Stopwords List
# Common functional words to exclude when filtering single noise words.
# ----------------------------------------------------
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'as', 'at', 'by', 'for', 'from', 
    'in', 'into', 'of', 'on', 'to', 'with', 'is', 'are', 'was', 'were', 
    'be', 'been', 'being', 'have', 'has', 'had', 'such', 'this', 'that'
}

def tokenize_words(text: str) -> List[str]:
    """
    Tokenizes raw text into lowercase word tokens,
    stripping punctuation marks while preserving individual word tokens.
    """
    return re.findall(r'\b\w+\b', text.lower())

def extract_unigram_transformations(tokens1: List[str], tokens2: List[str]) -> List[Tuple[str, str]]:
    """
    Extracts strictly 1-Gram (Word-to-Word) transformations based on positional alignment.
    Aligns modified words one-by-one while excluding multi-word phrases and stopwords.
    """
    matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
    pairs = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            orig_slice = tokens1[i1:i2]
            trans_slice = tokens2[j1:j2]

            # Iterate through aligned index positions word-by-word (1-Gram)
            min_len = min(len(orig_slice), len(trans_slice))
            for k in range(min_len):
                w1 = orig_slice[k]
                w2 = trans_slice[k]

                # Ensure non-identical 1-gram words and filter out standalone stopwords
                if w1 != w2:
                    if w1 not in STOPWORDS and w2 not in STOPWORDS:
                        pair = (w1, w2)
                        if pair not in pairs:
                            pairs.append(pair)

    return pairs

def plot_metric_table(metric_list: List[dict], mode: str, threshold: float, precision: float, task_name: str, date_time_str: str):
    """
    Renders and saves a high-quality table image displaying 1-Gram evaluation metrics,
    sorted by Cosine Similarity in descending order with color-coded PASS/FAIL status.
    """
    # Sort pairs by Cosine Similarity score in descending order
    metric_list.sort(key=lambda x: x['cosine_similarity'], reverse=True)

    # Simplified table headers without "(1-Gram)" labels
    headers = ["Status", "Original Phrase", "Transformed Phrase", "Cosine Sim"]
    table_data = []
    
    # Construct rows for the output table
    for item in metric_list:
        status_text = "PASS" if item['pass_threshold'] else "FAIL"
        table_data.append([
            status_text,
            item['original'],
            item['transformed'],
            f"{item['cosine_similarity']:.4f}"
        ])

    # Dynamically scale figure size based on the number of rows
    fig, ax = plt.subplots(figsize=(13, max(len(table_data) * 0.55 + 2, 4)))
    ax.axis('off') # Hide default chart axes
    
    # Title header displaying summary metrics
    title_text = f"1-Gram Transformation Metrics Table ({mode} Mode)\n" \
                 f"Task: {task_name} | Precision: {precision * 100:.2f}% | Threshold: {threshold}"
    plt.title(title_text, fontsize=13, fontweight='bold', pad=20)

    # Render Matplotlib table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center'
    )
    
    # Configure font sizes and scaling factors
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Apply cell styling for table header, PASS/FAIL indicators, and row striping
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2B4C7E') # Header dark blue background
            cell.set_text_props(color='white', fontweight='bold')
        else:
            if col == 0:
                # Highlight PASS status with soft green and FAIL with soft red
                if table_data[row - 1][0] == "PASS":
                    cell.set_facecolor('#D4EDDA')
                    cell.set_text_props(color='#155724', fontweight='bold')
                else:
                    cell.set_facecolor('#F8D7DA')
                    cell.set_text_props(color='#721C24', fontweight='bold')
            else:
                # Alternating row background colors (zebra striping)
                cell.set_facecolor('#F8F9FA' if row % 2 == 0 else '#FFFFFF')

    plt.tight_layout()

    # Ensure output 'plot' directory exists
    output_dir = "plot-1gram"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{task_name}_{date_time_str}_1gram_table.png"
    plot_path = os.path.join(output_dir, plot_filename)
    
    # Save table image as high-resolution PNG
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Saved cleanly formatted table plot to '{plot_path}'")
    plt.show()

# ----------------------------------------------------
# MAIN EXECUTION PIPELINE
# ----------------------------------------------------
def main():
    # Prompt user for JSON file path
    file_path = input("Enter JSON file path (e.g., data.json): ").strip()
    
    # Prompt user for evaluation mode selection
    print("\nSelect evaluation mode:")
    print("1: Synonym (Cosine >= 0.8)")
    print("2: Antonym (Cosine <= -0.8)")
    mode_choice = input("Enter choice (1 or 2): ").strip()

    # Configure similarity thresholds according to mode selection
    if mode_choice == '2':
        mode = "Antonym"
        threshold = -0.8
    else:
        mode = "Synonym"
        threshold = 0.8

    # Load JSON dataset
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Extract task_name attribute or default to fallback string
    task_name = json_data.get('task_name', 'question_answering')

    # Extract source text (Box 1) and target text (Box 4)
    text1_list = json_data['data'][0]['source_input']
    text1_raw = " ".join(text1_list)
    
    text2_list = json_data['data'][0]['followup_inputs'][2]
    text2_raw = " ".join(text2_list)

    # Tokenize input texts into individual word tokens
    tokens1 = tokenize_words(text1_raw)
    tokens2 = tokenize_words(text2_raw)

    # Extract strictly 1-Gram (Word-to-Word) pairs
    aligned_pairs = extract_unigram_transformations(tokens1, tokens2)

    # Load SentenceTransformer embedding model
    print("\nLoading Transformer Embedding Model (all-mpnet-base-v2)...")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    orig_texts = [p[0] for p in aligned_pairs]
    mod_texts = [p[1] for p in aligned_pairs]

    # Generate dense vector embeddings for 1-gram words
    embeddings_orig = model.encode(orig_texts, convert_to_tensor=True)
    embeddings_mod = model.encode(mod_texts, convert_to_tensor=True)

    # Calculate pairwise cosine similarity scores
    cosine_sims = util.cos_sim(embeddings_orig, embeddings_mod)
    
    metric_list = []
    for i in range(len(aligned_pairs)):
        sim_score = float(cosine_sims[i][i])
        
        # Evaluate pass condition against threshold
        is_valid = (mode == "Synonym" and sim_score >= threshold) or \
                   (mode == "Antonym" and sim_score <= threshold)
                   
        metric_list.append({
            "original": orig_texts[i],
            "transformed": mod_texts[i],
            "cosine_similarity": round(sim_score, 4),
            "pass_threshold": is_valid
        })

    # Calculate overall precision metric
    valid_changes = sum(1 for m in metric_list if m['pass_threshold'])
    precision = (valid_changes / len(metric_list)) if metric_list else 0.0

    print(f"\nTarget Mode: {mode}")
    print(f"Valid Word Changes: {valid_changes} / {len(metric_list)}")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")

    # Extract date-time timestamp pattern from filename or fallback to current timestamp
    match_dt = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file_path)
    date_time_str = match_dt.group(0) if match_dt else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Render and save visualization plot
    plot_metric_table(metric_list, mode, threshold, precision, task_name, date_time_str)

if __name__ == "__main__":
    main()
