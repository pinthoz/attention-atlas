"""
Modern minimalist GUI to review bias dataset sentences.
Clean and simple interface with keyboard shortcuts.
"""

import json
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

DATASET_FILE = Path(__file__).parent / "bias_sentences.json"
PROGRESS_FILE = Path(__file__).parent / "review_progress.json"
REVIEW_RESULTS_FILE = Path(__file__).parent / "review_results.json"

# Modern color palette
COLORS = {
    'bg': '#FFFFFF',
    'secondary_bg': '#F8F9FA',
    'text': '#1A1A1A',
    'text_light': '#6B7280',
    'accent': '#2563EB',
    'success': '#10B981',
    'error': '#EF4444',
    'border': '#E5E7EB',
    'hover': '#F3F4F6'
}


class SentenceReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentence Review")
        self.root.geometry("800x550")
        self.root.configure(bg=COLORS['bg'])

        # Load data
        self.dataset = self.load_dataset()
        self.entries = self.dataset['entries']
        self.total = len(self.entries)
        self.progress = self.load_progress()
        self.results = self.load_results()
        self.current_index = self.progress.get('current_index', 0)

        # Setup UI
        self.setup_ui()
        self.display_current_entry()

        # Keyboard shortcuts
        self.root.bind('<Left>', lambda e: self.previous_entry())
        self.root.bind('<Right>', lambda e: self.next_entry())
        self.root.bind('v', lambda e: self.mark_valid())
        self.root.bind('V', lambda e: self.mark_valid())
        self.root.bind('i', lambda e: self.mark_invalid())
        self.root.bind('I', lambda e: self.mark_invalid())
        self.root.bind('s', lambda e: self.skip_entry())
        self.root.bind('S', lambda e: self.skip_entry())
        self.root.bind('<space>', lambda e: self.next_entry())

    def load_dataset(self):
        """Load the bias sentences dataset."""
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_progress(self):
        """Load review progress if exists."""
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"current_index": 0, "reviewed_count": 0}

    def load_results(self):
        """Load review results if exists."""
        if REVIEW_RESULTS_FILE.exists():
            with open(REVIEW_RESULTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_progress(self):
        """Save current progress."""
        self.progress['current_index'] = self.current_index
        self.progress['reviewed_count'] = len(self.results)
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2)

    def save_results(self):
        """Save review results."""
        with open(REVIEW_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def setup_ui(self):
        """Setup minimalist user interface."""
        # Container with padding
        container = tk.Frame(self.root, bg=COLORS['bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=25)

        # Header - Progress counter
        header = tk.Frame(container, bg=COLORS['bg'])
        header.pack(fill=tk.X, pady=(0, 20))

        self.counter_label = tk.Label(
            header,
            text="",
            font=('Segoe UI', 9),
            fg=COLORS['text_light'],
            bg=COLORS['bg']
        )
        self.counter_label.pack(side=tk.LEFT)

        self.stats_label = tk.Label(
            header,
            text="",
            font=('Segoe UI', 9),
            fg=COLORS['text_light'],
            bg=COLORS['bg']
        )
        self.stats_label.pack(side=tk.RIGHT)

        # Thin progress bar
        self.progress_canvas = tk.Canvas(
            container,
            height=2,
            bg=COLORS['border'],
            highlightthickness=0
        )
        self.progress_canvas.pack(fill=tk.X, pady=(0, 30))

        # Main sentence display
        self.sentence_label = tk.Label(
            container,
            text="",
            font=('Segoe UI', 15),
            fg=COLORS['text'],
            bg=COLORS['bg'],
            wraplength=740,
            justify=tk.LEFT
        )
        self.sentence_label.pack(pady=(0, 20))

        # Subtle bias info
        self.bias_label = tk.Label(
            container,
            text="",
            font=('Segoe UI', 10),
            fg=COLORS['text_light'],
            bg=COLORS['bg'],
            wraplength=740,
            justify=tk.LEFT
        )
        self.bias_label.pack(pady=(0, 40))

        # Action buttons - compact
        button_frame = tk.Frame(container, bg=COLORS['bg'])
        button_frame.pack(pady=(10, 0))

        self.valid_btn = self.create_button(
            button_frame,
            "✓  Valid",
            self.mark_valid,
            COLORS['success']
        )
        self.valid_btn.pack(side=tk.LEFT, padx=8)

        self.skip_btn = self.create_button(
            button_frame,
            "Skip",
            self.skip_entry,
            COLORS['text_light']
        )
        self.skip_btn.pack(side=tk.LEFT, padx=8)

        self.invalid_btn = self.create_button(
            button_frame,
            "✗  Invalid",
            self.mark_invalid,
            COLORS['error']
        )
        self.invalid_btn.pack(side=tk.LEFT, padx=8)

        # Bottom navigation - minimal
        nav_frame = tk.Frame(container, bg=COLORS['bg'])
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(25, 0))

        # Previous button
        self.prev_btn = tk.Label(
            nav_frame,
            text="← Previous",
            font=('Segoe UI', 9),
            fg=COLORS['text_light'],
            bg=COLORS['bg'],
            cursor="hand2"
        )
        self.prev_btn.pack(side=tk.LEFT)
        self.prev_btn.bind('<Button-1>', lambda e: self.previous_entry())

        # Next button
        self.next_btn = tk.Label(
            nav_frame,
            text="Next →",
            font=('Segoe UI', 9),
            fg=COLORS['text_light'],
            bg=COLORS['bg'],
            cursor="hand2"
        )
        self.next_btn.pack(side=tk.RIGHT)
        self.next_btn.bind('<Button-1>', lambda e: self.next_entry())

        # Jump input (center)
        jump_frame = tk.Frame(nav_frame, bg=COLORS['bg'])
        jump_frame.pack()

        tk.Label(
            jump_frame,
            text="Jump to",
            font=('Segoe UI', 8),
            fg=COLORS['text_light'],
            bg=COLORS['bg']
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.jump_entry = tk.Entry(
            jump_frame,
            width=6,
            font=('Segoe UI', 9),
            fg=COLORS['text'],
            bg=COLORS['secondary_bg'],
            relief=tk.FLAT,
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=COLORS['border'],
            highlightcolor=COLORS['accent']
        )
        self.jump_entry.pack(side=tk.LEFT, padx=2)
        self.jump_entry.bind('<Return>', lambda e: self.jump_to_entry())

    def create_button(self, parent, text, command, color):
        """Create a modern flat button."""
        btn = tk.Label(
            parent,
            text=text,
            font=('Segoe UI', 11, 'bold'),
            fg='white',
            bg=color,
            padx=25,
            pady=12,
            cursor="hand2"
        )
        btn.bind('<Button-1>', lambda e: command())

        # Hover effect
        def on_enter(e):
            btn.config(bg=self.adjust_color(color, -20))

        def on_leave(e):
            btn.config(bg=color)

        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)

        return btn

    def adjust_color(self, hex_color, amount):
        """Adjust hex color brightness."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = max(0, min(255, r + amount))
        g = max(0, min(255, g + amount))
        b = max(0, min(255, b + amount))
        return f'#{r:02x}{g:02x}{b:02x}'

    def update_progress_display(self):
        """Update progress bar and labels."""
        # Update counter
        self.counter_label.config(
            text=f"{self.current_index + 1} / {self.total}"
        )

        # Update stats
        valid = sum(1 for v in self.results.values() if v == 'valid')
        invalid = sum(1 for v in self.results.values() if v == 'invalid')
        reviewed = len(self.results)

        self.stats_label.config(
            text=f"{reviewed} reviewed  •  {valid} valid  •  {invalid} invalid"
        )

        # Update progress bar
        progress_pct = (self.current_index / self.total)
        bar_width = self.progress_canvas.winfo_width()
        if bar_width > 1:
            self.progress_canvas.delete('all')
            filled_width = bar_width * progress_pct
            self.progress_canvas.create_rectangle(
                0, 0, filled_width, 2,
                fill=COLORS['accent'],
                outline=''
            )

    def display_current_entry(self):
        """Display the current entry."""
        if self.current_index >= self.total:
            messagebox.showinfo("Complete", "All entries reviewed!")
            return

        entry = self.entries[self.current_index]
        entry_id = str(entry['id'])

        # Update sentence text
        self.sentence_label.config(text=entry['text'])

        # Update bias info - minimalist style
        bias_parts = []
        if entry['has_bias']:
            bias_parts.append(f"Contains {entry['bias_type']} bias")
            if entry['bias_description']:
                bias_parts.append(f"• {entry['bias_description']}")
        else:
            bias_parts.append("No bias detected")

        self.bias_label.config(text=" ".join(bias_parts))

        # Update progress
        self.update_progress_display()

    def mark_valid(self):
        """Mark current entry as valid."""
        entry = self.entries[self.current_index]
        entry_id = str(entry['id'])
        self.results[entry_id] = 'valid'
        self.save_results()
        self.save_progress()
        self.next_entry()

    def mark_invalid(self):
        """Mark current entry as invalid."""
        entry = self.entries[self.current_index]
        entry_id = str(entry['id'])
        self.results[entry_id] = 'invalid'
        self.save_results()
        self.save_progress()
        self.next_entry()

    def skip_entry(self):
        """Skip current entry."""
        self.next_entry()

    def next_entry(self):
        """Move to next entry."""
        if self.current_index < self.total - 1:
            self.current_index += 1
            self.save_progress()
            self.display_current_entry()
        else:
            messagebox.showinfo("End", "You've reached the last entry!")

    def previous_entry(self):
        """Move to previous entry."""
        if self.current_index > 0:
            self.current_index -= 1
            self.save_progress()
            self.display_current_entry()
        else:
            messagebox.showinfo("Start", "You're at the first entry!")

    def jump_to_entry(self):
        """Jump to specific entry number."""
        try:
            entry_num = int(self.jump_entry.get())
            if 1 <= entry_num <= self.total:
                self.current_index = entry_num - 1
                self.save_progress()
                self.display_current_entry()
                self.jump_entry.delete(0, tk.END)
            else:
                messagebox.showerror("Error", f"Entry number must be between 1 and {self.total}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")


def main():
    root = tk.Tk()
    app = SentenceReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
