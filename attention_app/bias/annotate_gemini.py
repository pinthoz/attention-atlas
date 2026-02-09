"""
NER Annotation Script for GUS-Net (Google GenAI SDK v1.0 - Gemini 2.5 Edition)
==============================================================================
Updated to use 'gemini-2.5-flash' based on your available models.
"""

import json
import os
import time
import argparse
import ast
import re
from typing import Optional, List, Dict

# IMPORT DA SDK v1.0
from google import genai
from google.genai import types

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # ATUALIZADO: Usando um modelo disponível na sua lista
    "model": "gemini-2.5-flash",  
    "batch_size": 10,             
    "rate_limit_delay": 1.5,      # 2.5 Flash é muito rápido, podemos baixar o delay
    "max_retries": 3,             
}

ENTITIES = {
    "GEN": {
        "tag": "B-GEN/I-GEN",
        "description": "GEN (generalizations), broad generalization of a group (e.g., 'all Muslims', 'Women in general', 'everyone')"
    },
    "UNFAIR": {
        "tag": "B-UNFAIR/I-UNFAIR",
        "description": "UNFAIR (unfairness), harsh/unjust characterization or offensive language (e.g., 'idiot', 'stupid idea')"
    },
    "STEREO": {
        "tag": "B-STEREO/I-STEREO",
        "description": "STEREO (stereotypes), statement containing a stereotype targeting a group (e.g., 'women are bad drivers')"
    }
}

EXAMPLES = {
    "GEN": [
        {"input": "Women are all such bad drivers", "output": "['B-GEN', 'I-GEN', 'I-GEN', 'O', 'O', 'O']"},
        {"input": "Everyone from the city is rude", "output": "['B-GEN', 'I-GEN', 'I-GEN', 'I-GEN', 'O', 'O']"},
        {"input": "John is usually late to meetings", "output": "['O', 'O', 'O', 'O', 'O', 'O']"},
    ],
    "UNFAIR": [
        {"input": "Women are all such bad drivers", "output": "['O', 'O', 'B-UNFAIR', 'I-UNFAIR', 'I-UNFAIR', 'I-UNFAIR']"},
        {"input": "Your golf swing is terrible", "output": "['O', 'O', 'O', 'O', 'B-UNFAIR']"},
        {"input": "You might improve your speaking", "output": "['O', 'O', 'O', 'O', 'O']"},
    ],
    "STEREO": [
        {"input": "Women are all such bad drivers", "output": "['B-STEREO', 'I-STEREO', 'I-STEREO', 'I-STEREO', 'I-STEREO', 'I-STEREO']"},
        {"input": "He's good at math because he's Asian", "output": "['O', 'O', 'B-STEREO', 'I-STEREO', 'I-STEREO', 'I-STEREO', 'I-STEREO', 'I-STEREO']"},
        {"input": "She's mean to her teammates", "output": "['O', 'O', 'O', 'O', 'O']"},
    ]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_bias_dataset(filepath: str) -> List[Dict]:
    if not os.path.exists(filepath):
        print(f"Error: Input file '{filepath}' not found.")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON.")
            return []
    
    entries = data.get("entries", [])
    biased = [e for e in entries if e.get("has_bias") is True]
    if not biased: biased = entries

    seen = set()
    unique = []
    for entry in biased:
        t = entry.get("text", "").strip()
        if t and t not in seen:
            seen.add(t)
            unique.append(entry)
    
    print(f"Loaded {len(unique)} unique sentences to annotate.")
    return unique

def create_prompt(entity_type: str, sentence: str) -> str:
    entity = ENTITIES[entity_type]
    examples = EXAMPLES[entity_type]
    examples_text = "\n".join([f"Input: \"{ex['input']}\"\nOutput: {ex['output']}" for ex in examples])
    words = sentence.split()
    
    return f"""You are a NER annotator. Annotate for {entity['tag']}.
Definition: {entity['description']}

Format: Python list of strings (BIO tags).
Input sentence has {len(words)} words. You must return EXACTLY {len(words)} tags.

IMPORTANT RULES:
1. Output ONLY the list (e.g. ['O', 'B-GEN']). 
2. NO Markdown code blocks.
3. NO explanations.

Examples:
{examples_text}

Input: "{sentence}"
Output:"""

def parse_annotation(response_text: str, expected_len: int) -> List[str]:
    # Regex robusto para apanhar listas mesmo com lixo à volta
    match = re.search(r"\[(.*?)\]", response_text, re.DOTALL)
    
    if not match:
        clean = response_text.replace("```python", "").replace("```json", "").replace("```", "").strip()
        if clean.startswith("[") and clean.endswith("]"):
            list_str = clean
        else:
            return ['O'] * expected_len
    else:
        list_str = match.group(0)

    try:
        tags = ast.literal_eval(list_str)
        if not isinstance(tags, list): raise ValueError("Not a list")

        if len(tags) != expected_len:
            if len(tags) < expected_len:
                tags.extend(['O'] * (expected_len - len(tags)))
            else:
                tags = tags[:expected_len]
        return [str(t) for t in tags]
    except:
        return ['O'] * expected_len

def annotate_sentence(client: genai.Client, sentence: str) -> Dict:
    words = sentence.split()
    word_count = len(words)
    annotations = {}
    
    for entity_type in ENTITIES.keys():
        prompt = create_prompt(entity_type, sentence)
        tags = ['O'] * word_count
        success = False
        retries = 0
        
        while retries < CONFIG["max_retries"] and not success:
            try:
                response = client.models.generate_content(
                    model=CONFIG["model"],
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0)
                )
                if not response.text:
                    success = True 
                else:
                    tags = parse_annotation(response.text, word_count)
                    success = True
            except Exception as e:
                print(f"  Error {entity_type} (Try {retries+1}): {e}")
                retries += 1
                time.sleep(2 * (retries + 1))
        
        annotations[entity_type] = tags
        time.sleep(CONFIG["rate_limit_delay"])
    
    merged = []
    for i in range(word_count):
        w_tags = []
        for et, t_list in annotations.items():
            if i < len(t_list) and t_list[i] != 'O':
                w_tags.append(t_list[i])
        if not w_tags: merged.append("O")
        elif len(w_tags) == 1: merged.append(w_tags[0])
        else: merged.append(w_tags)

    return {
        "text_str": sentence,
        "tokens": words,
        "ner_tags_combined": merged,
        "individual_annotations": annotations
    }

# ============================================================================
# MAIN
# ============================================================================

def annotate_dataset(input_path, output_path, api_key, max_s=None, resume=0):
    client = genai.Client(api_key=api_key)
    
    # Teste simples de conexão
    try:
        print(f"Testing API connection with {CONFIG['model']}...")
        client.models.generate_content(model=CONFIG["model"], contents="Hi")
        print("Connection successful!")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    sentences = load_bias_dataset(input_path)
    if max_s: sentences = sentences[:max_s]
    
    results = []
    if resume > 0 and os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f: results = json.load(f)

    total = len(sentences)
    print(f"\nStarting annotation of {total - resume} sentences...")
    
    for i, entry in enumerate(sentences[resume:], start=resume):
        print(f"[{i+1}/{total}] Processing: {entry.get('text', '')[:40]}...")
        try:
            ann = annotate_sentence(client, entry.get("text", ""))
            ann["original_id"] = entry.get("id")
            ann["bias_type"] = entry.get("bias_type")
            results.append(ann)
        except Exception as e:
            print(f"Critical error on row {i}: {e}")
        
        if (i + 1) % CONFIG["batch_size"] == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  Saved progress.")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/ner_sentences_examples.json")
    parser.add_argument("--output", default="dataset/ner_sentences_examples_annotations.json")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--resume", type=int, default=0)
    args = parser.parse_args()
    
    annotate_dataset(args.input, args.output, args.api_key, args.max, args.resume)