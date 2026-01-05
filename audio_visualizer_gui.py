import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from video_generator import run_visualizer
from PIL import Image, ImageTk
import numpy as np
import threading
import os
import sys
import json
import ctypes

try:
    # Windows 8.1 and later
    ctypes.windll.shcore.SetProcessDpiAwareness(1) 
except:
    # Windows 8.0 or earlier (optional fallback)
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass # Not all versions of Windows have this function

# --- Configuration Constants ---
FLUENT_BLUE = "#0078D4"
FLUENT_BG = "#FFFFFF"
FLUENT_FG = "#1A1A1A"
FLUENT_INPUT_BG = "#F3F9FF"
FLUENT_BORDER = "#BBDFFF"
BUTTON_BG = "#E6F3FF"
BUTTON_FG = "#004E8C"
MASTER_FONT = "Arial"
# --- Pro-Level Presets ---
FREQ_PRESETS = {
    "Bass (20-150Hz)": (20, 150),
    "Low-Mid (150-500Hz)": (150, 500),
    "Mid (500-2kHz)": (500, 2000),
    "High (2k-8kHz)": (2000, 8000),
    "Air (8k-16kHz)": (8000, 16000),
    "Full Range": (20, 20000)
}
PLATFORM_PRESETS = {
    "YouTube (1920x1080)": (1920, 1080),
    "TikTok/Reels (1080x1920)": (1080, 1920),
    "Square/Insta (1080x1080)": (1080, 1080),
    "HD Ready (1280x720)": (1280, 720)
}
class TransformationPopup(tk.Toplevel):
    def __init__(self, parent, data_dict, app):
        super().__init__(parent)
        self.title("Configure Transforms")
        self.geometry("550x550")
        self.configure(bg=FLUENT_BG)
        self.transient(parent)
        self.grab_set()
        self.app = app
        self.data = data_dict
        self.vars = {}
        self._init_vars()
        self._build_ui()
    def _init_vars(self):
        # Default with the new smoothing params
        defaults = {
            'min_scale': 1.0, 'max_scale': 1.5,
            'min_rot': 0.0, 'max_rot': 0.0,
            'min_bright': 1.0, 'max_bright': 1.5,
            'min_sat': 1.0, 'max_sat': 1.0,
            'min_alpha': 1.0, 'max_alpha': 1.0,
            'min_hue': 0.0, 'max_hue': 0.0,
            'freq_min': 100, 'freq_max': 1000,
            'factor': 1.0, 'x': 640, 'y': 360,
            'attack': 0.2, 'release': 0.05
        }
        for key, default in defaults.items():
            val = self.data.get(key, default)
            self.vars[key] = tk.StringVar(value=str(val))
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=FLUENT_BG)
        style.configure("TNotebook.Tab", padding=[10, 5], font=(MASTER_FONT, 9))
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill='both', padx=10, pady=10)
        # --- Tab 1: General (Position & Frequency) ---
        tab_general = tk.Frame(notebook, bg=FLUENT_BG)
        notebook.add(tab_general, text="General")
        self._add_row(tab_general, "X Position:", 'x', 0)
        self._add_row(tab_general, "Y Position:", 'y', 1)
        drag_btn = tk.Button(
            tab_general,
            text="🖱️ Drag Around",
            command=self.open_drag_preview,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            relief="flat"
        )
        drag_btn.grid(row=0, column=2, rowspan=2, padx=10)
        # Audio Response Header
        tk.Label(tab_general, text="--- Audio Response ---", bg=FLUENT_BG, fg="#888", font=(MASTER_FONT, 9, "bold")).grid(row=2, column=0, columnspan=2, pady=10)
        # Frequency Preset Dropdown
        tk.Label(tab_general, text="Freq Range:", bg=FLUENT_BG, fg=FLUENT_FG).grid(row=3, column=0, sticky="w", padx=10)
        self.freq_combo = ttk.Combobox(tab_general, values=list(FREQ_PRESETS.keys()), state="readonly")
        self.freq_combo.grid(row=3, column=1, padx=10, sticky="ew")
        self.freq_combo.bind("<<ComboboxSelected>>", self._apply_freq_preset)
        self.freq_combo.set("Select Band...")
        self._add_row(tab_general, "Freq Min (Hz):", 'freq_min', 4)
        self._add_row(tab_general, "Freq Max (Hz):", 'freq_max', 5)
        self._add_row(tab_general, "Sensitivity:", 'factor', 6)
        # Smoothing Controls
        tk.Label(tab_general, text="--- Smoothing (Motion) ---", bg=FLUENT_BG, fg="#888", font=(MASTER_FONT, 9, "bold")).grid(row=7, column=0, columnspan=2, pady=10)
        self._add_row(tab_general, "Attack (0.0-1.0):", 'attack', 8)
        self._add_row(tab_general, "Release (0.0-1.0):", 'release', 9)
        tk.Label(tab_general, text="(Attack=Response Speed, Release=Decay Time)", bg=FLUENT_BG, fg="#888", font=(MASTER_FONT, 7)).grid(row=10, column=0, columnspan=2)
        # --- Tab 2: Geometry ---
        tab_geo = tk.Frame(notebook, bg=FLUENT_BG)
        notebook.add(tab_geo, text="Geometry")
        tk.Label(tab_geo, text="Size (Scale Factor)", bg=FLUENT_BG, fg=FLUENT_BLUE, font=(MASTER_FONT, 10, "bold")).pack(anchor="w", pady=(10, 5))
        self._add_row_pack(tab_geo, "Min Scale:", 'min_scale')
        self._add_row_pack(tab_geo, "Max Scale:", 'max_scale')
        tk.Label(tab_geo, text="Rotation (Degrees)", bg=FLUENT_BG, fg=FLUENT_BLUE, font=(MASTER_FONT, 10, "bold")).pack(anchor="w", pady=(20, 5))
        self._add_row_pack(tab_geo, "Min Rotation:", 'min_rot')
        self._add_row_pack(tab_geo, "Max Rotation:", 'max_rot')
        # --- Tab 3: Color & Effects ---
        tab_color = tk.Frame(notebook, bg=FLUENT_BG)
        notebook.add(tab_color, text="Color & Effects")
        tk.Label(tab_color, text="Brightness", bg=FLUENT_BG, fg=FLUENT_BLUE, font=(MASTER_FONT, 10, "bold")).pack(anchor="w", pady=(10, 5))
        self._add_row_pack(tab_color, "Min Bright:", 'min_bright')
        self._add_row_pack(tab_color, "Max Bright:", 'max_bright')
        tk.Label(tab_color, text="Saturation", bg=FLUENT_BG, fg=FLUENT_BLUE, font=(MASTER_FONT, 10, "bold")).pack(anchor="w", pady=(10, 2))
        self._add_row_pack(tab_color, "Min Sat:", 'min_sat')
        self._add_row_pack(tab_color, "Max Sat:", 'max_sat')
        tk.Label(tab_color, text="Hue Rotation (0-360 degrees)", bg=FLUENT_BG, fg=FLUENT_BLUE, font=(MASTER_FONT, 10, "bold")).pack(anchor="w", pady=(10, 2))
        self._add_row_pack(tab_color, "Min Hue Rot:", 'min_hue')
        self._add_row_pack(tab_color, "Max Hue Rot:", 'max_hue')
        tk.Label(tab_color, text="Transparency", bg=FLUENT_BG, fg=FLUENT_BLUE, font=(MASTER_FONT, 10, "bold")).pack(anchor="w", pady=(10, 2))
        self._add_row_pack(tab_color, "Min Alpha:", 'min_alpha')
        self._add_row_pack(tab_color, "Max Alpha:", 'max_alpha')
        # --- Save Button ---
        btn_frame = tk.Frame(self, bg=FLUENT_BG)
        btn_frame.pack(fill='x', padx=10, pady=10)
        tk.Button(btn_frame, text="Save & Close", command=self.save_and_close,
                  bg=FLUENT_BLUE, fg="white", relief="flat", padx=15, pady=5).pack(side="right")
    def _apply_freq_preset(self, event):
        choice = self.freq_combo.get()
        if choice in FREQ_PRESETS:
            low, high = FREQ_PRESETS[choice]
            self.vars['freq_min'].set(str(low))
            self.vars['freq_max'].set(str(high))
    def open_drag_preview(self):
        DragPreview(self, self.data, self.app)
    def _add_row(self, parent, label_text, key, row):
        tk.Label(parent, text=label_text, bg=FLUENT_BG, fg=FLUENT_FG).grid(row=row, column=0, sticky="w", padx=10, pady=5)
        e = tk.Entry(parent, textvariable=self.vars[key], bg=FLUENT_INPUT_BG, highlightthickness=1, bd=0)
        e.configure(highlightbackground=FLUENT_BORDER, highlightcolor=FLUENT_BLUE)
        e.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
    def _add_row_pack(self, parent, label_text, key):
        frame = tk.Frame(parent, bg=FLUENT_BG)
        frame.pack(fill='x', padx=10, pady=2)
        tk.Label(frame, text=label_text, width=25, anchor="w", bg=FLUENT_BG, fg=FLUENT_FG).pack(side="left")
        e = tk.Entry(frame, textvariable=self.vars[key], bg=FLUENT_INPUT_BG, width=10, bd=0, highlightthickness=1)
        e.configure(highlightbackground=FLUENT_BORDER, highlightcolor=FLUENT_BLUE)
        e.pack(side="right", fill="x", expand=True)
    def save_and_close(self):
        try:
            for key, var in self.vars.items():
                val = float(var.get())
                if val.is_integer(): val = int(val)
                self.data[key] = val
            self.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")
class DragPreview(tk.Toplevel):
    def __init__(self, parent, image_data, app):
        super().__init__(parent)
        self.title("Drag to Position (Preview)")
        self.configure(bg="black")
        self.resizable(False, False)
        self.parent = parent
        self.app = app
        self.data = image_data
        self.bg_path = app.bg_path.get()
        # 1. Get Real Dimensions
        self.real_w = int(app.bg_width.get())
        self.real_h = int(app.bg_height.get())
        # 2. Calculate Scale Factor (1/3 Screen Width)
        screen_width = self.winfo_screenwidth()
        target_width = screen_width // 3
        self.scale_factor = target_width / self.real_w
        # 3. Calculate Canvas Dimensions
        self.canvas_w = int(self.real_w * self.scale_factor)
        self.canvas_h = int(self.real_h * self.scale_factor)
        # Center window on screen
        x_pos = (screen_width - self.canvas_w) // 2
        y_pos = (self.winfo_screenheight() - self.canvas_h) // 2
        self.geometry(f"{self.canvas_w}x{self.canvas_h + 50}+{x_pos}+{y_pos}")
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg="black", highlightthickness=0)
        self.canvas.pack()
        self.drag_target = None
        self.offset_x = 0
        self.offset_y = 0
        self.active_item_id = None # Track the ID of the specific image we are editing
        self._load_assets()
        self._draw_scene()
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        tk.Button(
            self,
            text="Save Position",
            command=self.save_and_close,
            bg=FLUENT_BLUE,
            fg="white",
            relief="flat",
            padx=10,
            pady=5
        ).pack(pady=8)
    def _load_assets(self):
        from PIL import Image, ImageTk
        # Load and resize background to PREVIEW size
        self.bg_img = Image.open(self.bg_path).resize((self.canvas_w, self.canvas_h), Image.LANCZOS)
        self.bg_tk = ImageTk.PhotoImage(self.bg_img)
        # Store (entry, pil_image, numpy_array) and track canvas items
        self.images = []
        self.canvas_items = []
        for entry in self.app.image_entries:
            pil_img = Image.open(entry['file']).convert("RGBA")
            np_img = np.array(pil_img)
            self.images.append((entry, pil_img, np_img))
    def _draw_scene(self):
        from PIL import ImageTk
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.bg_tk, anchor="nw")
        self.tk_images = [] # Keep references to avoid garbage collection
        self.canvas_items.clear()
        for entry, pil_img, np_img in self.images:
            # 1. Visual approximation: apply static preview scale only (no audio)
            w, h = pil_img.size
            preview_w = int(w * self.scale_factor)
            preview_h = int(h * self.scale_factor)
            resized_img = pil_img.resize((preview_w, preview_h), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(resized_img)
            self.tk_images.append(tk_img)
            # 3. Map Real Coordinates -> Preview Coordinates
            preview_x = entry['x'] * self.scale_factor
            preview_y = entry['y'] * self.scale_factor
            item = self.canvas.create_image(preview_x, preview_y, image=tk_img)
            self.canvas.itemconfig(item, tags=("layer",))
            self.canvas_items.append((item, entry, preview_w, preview_h))
            if entry is self.data:
                self.active_item_id = item
                # draw highlight rectangle
                self._draw_highlight_for_item(item, preview_w, preview_h)
    def on_click(self, event):
        # Find item under mouse
        item = self.canvas.find_closest(event.x, event.y)
        if not item: return
        # Allow dragging any image
        self.drag_target = item[0]
        x0, y0 = self.canvas.coords(self.drag_target)
        self.offset_x = x0 - event.x
        self.offset_y = y0 - event.y
        # If clicked item corresponds to the active entry, update highlight
        for cid, entry, pw, ph in self.canvas_items:
            if cid == self.drag_target and entry is self.data:
                self.active_item_id = cid
                self._draw_highlight_for_item(cid, pw, ph)
    def on_drag(self, event):
        if not self.drag_target: return
        # Move visually in the canvas
        self.canvas.coords(
            self.drag_target,
            event.x + self.offset_x,
            event.y + self.offset_y
        )
        # If highlighted, move the highlight rectangle with the item
        if self.active_item_id:
            if self.drag_target == self.active_item_id:
                # find corresponding w/h
                for cid, entry, pw, ph in self.canvas_items:
                    if cid == self.active_item_id:
                        x, y = self.canvas.coords(self.active_item_id)
                        x0 = int(x - pw/2)
                        y0 = int(y - ph/2)
                        x1 = int(x + pw/2)
                        y1 = int(y + ph/2)
                        if hasattr(self, 'highlight_id') and self.highlight_id:
                            self.canvas.coords(self.highlight_id, x0, y0, x1, y1)
                        break
    def save_and_close(self):
        # Save positions for all canvas items (regardless of which was active)
        for cid, entry, pw, ph in self.canvas_items:
            try:
                preview_x, preview_y = self.canvas.coords(cid)
            except Exception:
                continue
            real_x = int(preview_x / self.scale_factor)
            real_y = int(preview_y / self.scale_factor)
            entry['x'] = real_x
            entry['y'] = real_y
        # Update parent UI vars if the edited one changed
        if self.data:
            self.parent.vars['x'].set(str(self.data.get('x', '')))
            self.parent.vars['y'].set(str(self.data.get('y', '')))
        self.destroy()

    def _draw_highlight_for_item(self, item_id, w, h):
        # Remove old highlight
        if hasattr(self, 'highlight_id') and self.highlight_id:
            try:
                self.canvas.delete(self.highlight_id)
            except Exception:
                pass
        x, y = self.canvas.coords(item_id)
        x0 = int(x - w/2)
        y0 = int(y - h/2)
        x1 = int(x + w/2)
        y1 = int(y + h/2)
        self.highlight_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00C853", width=3)
class PreviewPopup(tk.Toplevel):
    def __init__(self, parent, app):
        super().__init__(parent)
        self._pending_redraw = False
        self.title("Preview - Audio Response")
        self.configure(bg=FLUENT_BG)
        self.resizable(False, False)
        self.app = app
        self.bg_path = app.bg_path.get()
        
        # Get real dimensions
        self.real_w = int(app.bg_width.get())
        self.real_h = int(app.bg_height.get())
        
        # Calculate scale factor (1/3 Screen Width)
        screen_width = self.winfo_screenwidth()
        target_width = screen_width // 3
        self.scale_factor = target_width / self.real_w
        
        # Calculate canvas dimensions
        self.canvas_w = int(self.real_w * self.scale_factor)
        self.canvas_h = int(self.real_h * self.scale_factor)
        
        # Center window on screen
        x_pos = (screen_width - self.canvas_w) // 2
        y_pos = (self.winfo_screenheight() - self.canvas_h) // 2
        self.geometry(f"{self.canvas_w}x{self.canvas_h + 120}+{x_pos}+{y_pos}")
        
        # Canvas for preview
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg=FLUENT_BG, highlightthickness=0)
        self.canvas.pack()
        
        # Load original image assets (unmodified)
        self._load_assets()

        # Slider frame (styled to match app)
        slider_frame = tk.Frame(self, bg=FLUENT_BG, height=80)
        slider_frame.pack(fill="x")
        
        tk.Label(slider_frame, text="Loudness", bg=FLUENT_BG, fg=FLUENT_BLUE, font=(MASTER_FONT, 10, "bold")).pack(anchor="w", padx=10, pady=(5, 2))
        
        # Loudness slider (0 to 1)
        self.loudness_var = tk.DoubleVar(value=0.5)
        self.loudness_slider = tk.Scale(
            slider_frame, 
            from_=0.0, 
            to=1.0, 
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.loudness_var,
            bg="#444444",
            fg="white",
            command=self._on_loudness_change,
            length=self.canvas_w - 20,
            highlightthickness=0
        )
        self.loudness_slider.pack(padx=10, pady=5)
        
        # Info label
        self.info_label = tk.Label(slider_frame, text="", bg=FLUENT_BG, fg="#888888", font=(MASTER_FONT, 8))
        self.info_label.pack(anchor="w", padx=10, pady=(0, 5))
        
        # Initialize smoothed loudness values for each image
        self.smoothed_loudness = {i: 0.0 for i in range(len(app.image_entries))}
        
        # Draw initial scene
        self._draw_scene()
        # Close button (one-time)
        btn_frame = tk.Frame(self, bg=FLUENT_BG)
        btn_frame.pack(fill='x', pady=(6,8))
        close_btn = tk.Button(btn_frame, text="Close", command=self.destroy, bg=FLUENT_BLUE, fg='white', relief='flat')
        self.app._style_button(close_btn, primary=True)
        close_btn.pack(side='right', padx=10)
    
    def _load_assets(self):
        from PIL import Image, ImageTk
        # Load and resize background to PREVIEW size
        self.bg_img = Image.open(self.bg_path).resize((self.canvas_w, self.canvas_h), Image.LANCZOS)
        self.bg_tk = ImageTk.PhotoImage(self.bg_img)

        # Store images as numpy arrays for fast transforms
        self.original_images = []  # list of (entry, pil_img, np_img)
        for entry in self.app.image_entries:
            pil_img = Image.open(entry['file']).convert("RGBA")
            np_img = np.array(pil_img)
            self.original_images.append((entry, pil_img, np_img))
    
    def _apply_transformations_numpy(self, np_img, loudness, config):
        from video_generator import apply_transformations_numpy

        # Disable hue rotation in preview (too slow)
        config = dict(config)
        config['min_hue'] = 0
        config['max_hue'] = 0

        return apply_transformations_numpy(np_img, loudness, config)
    
    def _on_loudness_change(self, value):
        raw_loudness = float(value)

        for i, entry in enumerate(self.app.image_entries):
            attack = entry.get('attack', 0.2)
            release = entry.get('release', 0.05)
            prev = self.smoothed_loudness[i]

            if raw_loudness > prev:
                self.smoothed_loudness[i] = prev + attack * (raw_loudness - prev)
            else:
                self.smoothed_loudness[i] = prev + release * (raw_loudness - prev)

        if not self._pending_redraw:
            self._pending_redraw = True
            self.after(16, self._throttled_redraw)  # ~60 FPS max

    
    def _throttled_redraw(self):
        self._pending_redraw = False
        self._draw_scene()

    def _draw_scene(self):
        from PIL import ImageTk
        if not hasattr(self, "_bg_item"):
            self._bg_item = self.canvas.create_image(0, 0, image=self.bg_tk, anchor="nw")
        else:
            self.canvas.itemconfig(self._bg_item, image=self.bg_tk)
        
        self.tk_images = []  # Keep references to avoid garbage collection
        for i, (entry, pil_img, np_img) in enumerate(self.original_images):
            loudness = self.smoothed_loudness[i]
            # Apply numpy-based transforms (returns RGBA numpy array)
            try:
                transformed_np = self._apply_transformations_numpy(np_img.copy(), loudness, entry)
            except Exception:
                # Fallback to PIL pipeline if numpy transform fails
                transformed_pil = pil_img.copy()
                if hasattr(self, '_apply_transformations'):
                    transformed_pil = self._apply_transformations(transformed_pil, loudness, entry)
                transformed_np = np.array(transformed_pil)

            # Convert numpy RGBA -> PIL and scale down for preview
            try:
                transformed_pil = Image.fromarray(transformed_np)
            except Exception:
                transformed_pil = pil_img.copy()

            w, h = transformed_pil.size
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            resized_img = transformed_pil.resize((max(1, new_w), max(1, new_h)), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(resized_img)
            self.tk_images.append(tk_img)

            preview_x = entry['x'] * self.scale_factor
            preview_y = entry['y'] * self.scale_factor
            self.canvas.create_image(preview_x, preview_y, image=tk_img)
        # style: draw a subtle border
        self.canvas.configure(bg=FLUENT_BG)
        
        
class VisualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Endergonic")
        self.root.configure(bg="white")
        self.root.geometry("850x750")
        self.image_entries = []
        self.audio_path = tk.StringVar()
        self.bg_path = tk.StringVar()
        self.bg_width = tk.StringVar(value="1920")
        self.bg_height = tk.StringVar(value="1080")
        self.fps = tk.StringVar(value="30")
        self.render_done = threading.Event()
        self._build_layout()
    def _style_entry(self, entry):
        entry.configure(font=(MASTER_FONT, 10), bg=FLUENT_INPUT_BG, fg=FLUENT_FG,
                        insertbackground=FLUENT_BLUE, highlightthickness=1,
                        highlightbackground=FLUENT_BORDER, highlightcolor=FLUENT_BLUE, bd=0)
    def _style_button(self, button, primary=False):
        bg = FLUENT_BLUE if primary else BUTTON_BG
        fg = "white" if primary else BUTTON_FG
        button.configure(font=(MASTER_FONT, 10), bg=bg, fg=fg, relief="flat",
                         activebackground=FLUENT_BORDER, padx=10, pady=4, borderwidth=0)
    def _build_layout(self):
        # --- TOP MENUBAR FOR PRESETS ---
        top_bar = tk.Frame(self.root, bg="#f0f0f0", pady=5, padx=10)
        top_bar.pack(fill="x")
        tk.Label(top_bar, text="Project:", bg="#f0f0f0", font=(MASTER_FONT, 9, "bold")).pack(side="left")
        load_btn = tk.Button(top_bar, text="📂 Load Preset", command=self.load_preset, bg="white", relief="flat")
        load_btn.pack(side="left", padx=5)
        save_btn = tk.Button(top_bar, text="💾 Save Preset", command=self.save_preset, bg="white", relief="flat")
        save_btn.pack(side="left", padx=5)
        # --- MAIN HEADER ---
        header = tk.Frame(self.root, bg="white")
        header.pack(fill="x", padx=20, pady=15)
        self._create_file_picker(header, "Audio File:", self.audio_path, 0)
        self._create_file_picker(header, "Background:", self.bg_path, 1)
        # Platform / Resolution Settings
        settings_frame = tk.Frame(header, bg="white")
        settings_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=10)
        tk.Label(settings_frame, text="Output Format:", bg="white", font=(MASTER_FONT, 9, "bold")).pack(side="left", padx=(0,10))
        self.platform_combo = ttk.Combobox(settings_frame, values=list(PLATFORM_PRESETS.keys()), state="readonly", width=25)
        self.platform_combo.pack(side="left")
        self.platform_combo.bind("<<ComboboxSelected>>", self._apply_platform_preset)
        self.platform_combo.set("YouTube (1920x1080)")
        # Manual resolution overrides
        self._add_inline_setting(settings_frame, "W:", self.bg_width)
        self._add_inline_setting(settings_frame, "H:", self.bg_height)
        self._add_inline_setting(settings_frame, "FPS:", self.fps)
        tk.Frame(self.root, height=1, bg="#E0E0E0").pack(fill="x")
        # --- IMAGE LIST ---
        list_container = tk.Frame(self.root, bg="white")
        list_container.pack(fill="both", expand=True, padx=20, pady=10)
        tk.Label(list_container, text="Visual Elements", font=(MASTER_FONT, 12, "bold"), bg="white", fg=FLUENT_FG).pack(anchor="w")
        self.canvas = tk.Canvas(list_container, bg="white", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="white")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        # --- FOOTER ---
        footer = tk.Frame(self.root, bg="#F5F5F5", height=60)
        footer.pack(fill="x", side="bottom")
        self.add_btn = tk.Button(footer, text="+ Add Image", command=self.add_image_entry)
        self._style_button(self.add_btn)
        self.add_btn.pack(side="left", padx=20, pady=15)
        self.preview_btn = tk.Button(footer, text="Preview", command=self.open_preview)
        self._style_button(self.preview_btn, primary=False)
        self.preview_btn.pack(side="left", padx=5, pady=15)
        self.submit_btn = tk.Button(footer, text="Generate Video", command=self.submit)
        self._style_button(self.submit_btn, primary=True)
        self.submit_btn.pack(side="right", padx=20, pady=15)
        self.progress_label = tk.Label(footer, text="", bg="#F5F5F5", fg=FLUENT_BLUE)
        self.progress_label.pack(side="right", padx=10)
    def _create_file_picker(self, parent, text, var, row):
        tk.Label(parent, text=text, bg="white", font=(MASTER_FONT, 10)).grid(row=row, column=0, sticky="w", pady=5)
        e = tk.Entry(parent, textvariable=var, width=50)
        self._style_entry(e)
        e.grid(row=row, column=1, padx=10, pady=5)
        b = tk.Button(parent, text="Browse", command=lambda: self._browse(var))
        self._style_button(b)
        b.grid(row=row, column=2, padx=5)
    def _add_inline_setting(self, parent, text, var):
        f = tk.Frame(parent, bg="white")
        f.pack(side="left", padx=(15, 0))
        tk.Label(f, text=text, bg="white").pack(side="left")
        e = tk.Entry(f, textvariable=var, width=6)
        self._style_entry(e)
        e.pack(side="left", padx=5)
    def _apply_platform_preset(self, event):
        choice = self.platform_combo.get()
        if choice in PLATFORM_PRESETS:
            w, h = PLATFORM_PRESETS[choice]
            self.bg_width.set(str(w))
            self.bg_height.set(str(h))
    def _browse(self, var):
        path = filedialog.askopenfilename()
        if path: var.set(path)
    # --- JSON PRESETS LOGIC ---
    def save_preset(self):
        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not f: return
        data = {
            "bg_width": self.bg_width.get(),
            "bg_height": self.bg_height.get(),
            "fps": self.fps.get(),
            "bg_path": self.bg_path.get(),
            "audio_path": self.audio_path.get(),
            "images": self.image_entries
        }
        with open(f, 'w') as outfile:
            json.dump(data, outfile, indent=4)
        messagebox.showinfo("Saved", "Preset saved successfully.")
    def load_preset(self):
        f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not f: return
        try:
            with open(f, 'r') as infile:
                data = json.load(infile)
            self.bg_width.set(data.get("bg_width", "1920"))
            self.bg_height.set(data.get("bg_height", "1080"))
            self.fps.set(data.get("fps", "30"))
            self.bg_path.set(data.get("bg_path", ""))
            self.audio_path.set(data.get("audio_path", ""))
            self.image_entries = data.get("images", [])
            self._render_list()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load preset: {e}")
    # --- SMART DEFAULTS ---
    def add_image_entry(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not path: return
        try:
            w = int(self.bg_width.get())
            h = int(self.bg_height.get())
        except:
            w, h = 1920, 1080
        # "Opinionated" Default Settings (Smart Default)
        data = {
            'file': path,
            'x': w // 2,
            'y': h // 2,
            # Audio
            'freq_min': 60,   'freq_max': 250, # Kick drum area
            'factor': 1.0,
            # Smoothing (Pro)
            'attack': 1,    'release': 1,
            # Visuals (Subtle pulse)
            'min_scale': 0.95, 'max_scale': 1.05,
            'min_bright': 0.95, 'max_bright': 1.05,
            'min_alpha': 0.9,  'max_alpha': 1.0,
            # Zero these out
            'min_rot': 0, 'max_rot': 0,
            'min_hue': 0, 'max_hue': 0,
            'min_sat': 1, 'max_sat': 1
        }
        self.image_entries.append(data)
        self._render_list()
    def _render_list(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        for i, data in enumerate(self.image_entries):
            row = tk.Frame(self.scrollable_frame, bg="white", pady=5)
            row.pack(fill="x", expand=True)
            tk.Label(row, text=f"Layer {i+1}", bg="#EEE", width=8, font=(MASTER_FONT, 9, "bold")).pack(side="left", padx=5)
            fname = os.path.basename(data['file'])
            if len(fname) > 30: fname = fname[:27] + "..."
            info_text = f"{fname} | Pos: {data.get('x')},{data.get('y')}"
            tk.Label(row, text=info_text, bg="white", anchor="w").pack(side="left", padx=10, fill="x", expand=True)
            edit_btn = tk.Button(row, text="🛠️Edit", command=lambda d=data: TransformationPopup(self.root, d, self))
            self._style_button(edit_btn)
            edit_btn.pack(side="left", padx=2)
            del_btn = tk.Button(row, text="×", bg="#FFEAEA", fg="red", relief="flat", command=lambda idx=i: self._delete_entry(idx))
            del_btn.pack(side="left", padx=5)
            tk.Frame(self.scrollable_frame, height=1, bg="#EEE").pack(fill="x")
    def _delete_entry(self, idx):
        self.image_entries.pop(idx)
        self._render_list()
    def open_preview(self):
        if not self.image_entries:
            messagebox.showwarning("Warning", "Please add at least one image first.")
            return
        if not self.bg_path.get():
            messagebox.showwarning("Warning", "Please select a background image first.")
            return
        PreviewPopup(self.root, self)

    def submit(self):
        self.submit_btn.config(state='disabled', text="Rendering...")
        threading.Thread(target=self.run_thread, daemon=True).start()
    def run_thread(self):
        try:
            config = {
                'audio_file': self.audio_path.get(),
                'bg_file': self.bg_path.get(),
                'bg_size': (int(self.bg_width.get()), int(self.bg_height.get())),
                'fps': float(self.fps.get()),
                'images': self.image_entries
            }
            run_visualizer(config, progress_callback=self.update_progress, on_complete=self.on_done)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.submit_btn.config(state='normal', text="Generate Video"))
    def update_progress(self, current, total, percent, eta):
        msg = f"Rendering: {current} / {total} total frames. | {percent}% completed. ETA: {eta}"
        self.root.after(0, lambda: self.progress_label.config(text=msg))
    def on_done(self):
        self.root.after(0, lambda: [
            self.progress_label.config(text="Done!"),
            self.submit_btn.config(state='normal', text="Generate Video"),
            messagebox.showinfo("Success", "Video rendering complete!")
        ])

class SplashScreen(tk.Toplevel):
    def __init__(self, parent, image_path, duration):
        super().__init__(parent)

        self.overrideredirect(True)
        self.configure(bg="white")

        # Load image
        img = Image.open(image_path)
        # Optional: resize image if you want it smaller
        img = img.resize((img.width // 3, img.height // 3), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)

        # Put image in a label
        label = tk.Label(self, image=self.photo, bg="white")
        label.pack(padx=0, pady=0)

        # Center window on screen
        w, h = img.size
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        self.after(duration, self.destroy)

if __name__ == '__main__':
    root = tk.Tk()
    
    root.iconify()  # Hide main app initially

    splash = SplashScreen(
        root,
        image_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png"),
        duration=3001
    )

    splash.after(3000, root.deiconify)

    root.eval(f'tk::PlaceWindow {str(splash)} center')

    if hasattr(sys, '_MEIPASS'): icon_path = os.path.join(sys._MEIPASS, "favicon.ico")
    else: icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.ico")
    
    if os.path.exists(icon_path): 
        try: root.iconbitmap(icon_path) 
        except Exception: pass

    app = VisualizerGUI(root)

    root.mainloop()
