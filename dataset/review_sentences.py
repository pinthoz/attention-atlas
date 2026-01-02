"""
Interactive script to review bias dataset sentences.
Allows marking sentences as valid/invalid and saves progress.
"""

import json
import os
from pathlib import Path

DATASET_FILE = Path(__file__).parent / "bias_sentences.json"
PROGRESS_FILE = Path(__file__).parent / "review_progress.json"
REVIEW_RESULTS_FILE = Path(__file__).parent / "review_results.json"


def load_dataset():
    """Load the bias sentences dataset."""
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_progress():
    """Load review progress if exists."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "current_index": 0,
        "reviewed_count": 0
    }


def load_results():
    """Load review results if exists."""
    if REVIEW_RESULTS_FILE.exists():
        with open(REVIEW_RESULTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_progress(progress):
    """Save current progress."""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)


def save_results(results):
    """Save review results."""
    with open(REVIEW_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def display_entry(entry, index, total):
    """Display an entry for review."""
    print("\n" + "="*80)
    print(f"Entry {index + 1}/{total} (ID: {entry['id']})")
    print("="*80)
    print(f"\nType: {entry['type']}")
    print(f"\nText: {entry['text']}")
    print(f"\nHas Bias: {entry['has_bias']}")
    if entry['has_bias']:
        print(f"Bias Type: {entry['bias_type']}")
        print(f"Description: {entry['bias_description']}")
    print("\n" + "-"*80)


def get_user_input():
    """Get user review decision."""
    print("\nOptions:")
    print("  [v] Valid - sentence makes sense")
    print("  [i] Invalid - sentence doesn't make sense")
    print("  [s] Skip - review later")
    print("  [b] Back - go to previous sentence")
    print("  [j] Jump - go to specific entry number")
    print("  [q] Quit - save and exit")

    while True:
        choice = input("\nYour choice: ").strip().lower()
        if choice in ['v', 'i', 's', 'b', 'j', 'q']:
            return choice
        print("Invalid choice. Please choose v, i, s, b, j, or q.")


def get_jump_index(total):
    """Get index to jump to."""
    while True:
        try:
            entry_num = input(f"\nEnter entry number (1-{total}): ").strip()
            index = int(entry_num) - 1
            if 0 <= index < total:
                return index
            print(f"Please enter a number between 1 and {total}.")
        except ValueError:
            print("Please enter a valid number.")


def show_statistics(results, total):
    """Show review statistics."""
    valid = sum(1 for v in results.values() if v == 'valid')
    invalid = sum(1 for v in results.values() if v == 'invalid')
    reviewed = len(results)

    print("\n" + "="*80)
    print("REVIEW STATISTICS")
    print("="*80)
    print(f"Total entries: {total}")
    print(f"Reviewed: {reviewed} ({reviewed/total*100:.1f}%)")
    print(f"Valid: {valid}")
    print(f"Invalid: {invalid}")
    print(f"Remaining: {total - reviewed}")
    print("="*80)


def main():
    """Main review loop."""
    print("="*80)
    print("BIAS DATASET SENTENCE REVIEWER")
    print("="*80)

    # Load data
    dataset = load_dataset()
    entries = dataset['entries']
    total = len(entries)

    progress = load_progress()
    results = load_results()

    print(f"\nLoaded {total} entries from dataset.")
    print(f"Starting from entry {progress['current_index'] + 1}")
    print(f"Already reviewed: {len(results)} entries")

    # Review loop
    current_index = progress['current_index']

    while current_index < total:
        entry = entries[current_index]
        entry_id = str(entry['id'])

        # Display entry
        display_entry(entry, current_index, total)

        # Show if already reviewed
        if entry_id in results:
            print(f"\n[Previously marked as: {results[entry_id]}]")

        # Get user input
        choice = get_user_input()

        if choice == 'v':
            results[entry_id] = 'valid'
            print("✓ Marked as VALID")
            current_index += 1

        elif choice == 'i':
            results[entry_id] = 'invalid'
            print("✗ Marked as INVALID")
            current_index += 1

        elif choice == 's':
            print("⊙ Skipped")
            current_index += 1

        elif choice == 'b':
            if current_index > 0:
                current_index -= 1
                print("← Going back")
            else:
                print("Already at first entry!")

        elif choice == 'j':
            current_index = get_jump_index(total)

        elif choice == 'q':
            print("\nSaving progress and exiting...")
            break

        # Auto-save every 10 entries
        if current_index % 10 == 0:
            progress['current_index'] = current_index
            progress['reviewed_count'] = len(results)
            save_progress(progress)
            save_results(results)
            print("\n[Auto-saved progress]")

    # Final save
    progress['current_index'] = current_index
    progress['reviewed_count'] = len(results)
    save_progress(progress)
    save_results(results)

    # Show statistics
    show_statistics(results, total)

    print("\n✓ Progress saved!")
    print(f"  - Progress: {PROGRESS_FILE}")
    print(f"  - Results: {REVIEW_RESULTS_FILE}")

    if current_index >= total:
        print("\n🎉 All entries reviewed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress has been saved.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
