import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

class ImageComparisonApp:
    """
    GUI application to generate an academic-style comparison plot.
    - Loads the original image without any pre-processing.
    - Matches comparison images to the original's format (color/grayscale).
    - Allows synchronized magnification via ROI selection.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Image Comparison Tool")
        self.original_image_path = ""
        self.comparison_paths = []
        self.roi_coords = None

        # --- GUI Elements ---
        self.main_frame = tk.Frame(root, padx=10, pady=10)
        self.main_frame.pack()
        
        # Changed "Ground Truth" to "Original Image"
        self.original_label = tk.Label(self.main_frame, text="1. Select Original Image (Before Processing):")
        self.original_label.pack(anchor="w")
        
        self.original_frame = tk.Frame(self.main_frame)
        self.original_frame.pack(fill="x", pady=5)
        self.original_path_label = tk.Label(self.original_frame, text="No file selected", fg="gray", width=50, anchor="w")
        self.original_path_label.pack(side="left")
        self.original_button = tk.Button(self.original_frame, text="Browse...", command=self.load_original_image)
        self.original_button.pack(side="right")
        
        self.comp_label = tk.Label(self.main_frame, text="2. Select Comparison Image(s) (After Processing):")
        self.comp_label.pack(anchor="w", pady=(10, 0))
        self.comp_frame = tk.Frame(self.main_frame)
        self.comp_frame.pack(fill="x", pady=5)
        self.comp_listbox = tk.Listbox(self.comp_frame, height=5)
        self.comp_listbox.pack(fill="x")
        self.comp_button = tk.Button(self.main_frame, text="Add Image(s)...", command=self.load_comparison_images)
        self.comp_button.pack(pady=5)
        self.clear_button = tk.Button(self.main_frame, text="Clear Selections", command=self.clear_selections)
        self.clear_button.pack(pady=5)
        self.generate_button = tk.Button(self.main_frame, text="Generate Comparison Plot", command=self.generate_plot)
        self.generate_button.pack(pady=20)

    def load_original_image(self):
        """Opens a file dialog to select the original, unprocessed image."""
        path = filedialog.askopenfilename(
            title="Select Original Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.tif *.bmp")]
        )
        if path:
            self.original_image_path = path
            self.original_path_label.config(text=path.split('/')[-1], fg="black")

    def load_comparison_images(self):
        """Opens a file dialog to select one or more comparison images."""
        paths = filedialog.askopenfilenames(
            title="Select Comparison Image(s)",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.tif *.bmp")]
        )
        if paths:
            for path in paths:
                if path not in self.comparison_paths:
                    self.comparison_paths.append(path)
                    self.comp_listbox.insert(tk.END, path.split('/')[-1])
                    
    def clear_selections(self):
        """Clears all selected image paths."""
        self.original_image_path = ""
        self.comparison_paths = []
        self.original_path_label.config(text="No file selected", fg="gray")
        self.comp_listbox.delete(0, tk.END)

    def _select_roi_callback(self, eclick, erelease):
        """Callback for the RectangleSelector."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.roi_coords = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def _get_roi_from_user(self, image_for_display):
        """Displays an image and lets the user select a rectangular ROI."""
        self.roi_coords = None # Reset ROI
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image_for_display)
        ax.set_title("Click and drag to select ROI, then close this window to continue.", fontsize=12)
        ax.axis('off')

        selector = RectangleSelector(
            ax, self._select_roi_callback, useblit=True, button=[1], 
            minspanx=5, minspany=5, spancoords='pixels', interactive=True
        )
        plt.show() # Blocks execution until the plot window is closed
        return self.roi_coords

    def generate_plot(self):
        """Validates selections, asks for ROI, and generates the plot."""
        if not self.original_image_path or not self.comparison_paths:
            messagebox.showerror("Error", "Please select an original image and at least one comparison image.")
            return

        try:
            # Load original image AS IS (IMREAD_UNCHANGED)
            original_image = cv2.imread(self.original_image_path, cv2.IMREAD_UNCHANGED)
            if original_image is None:
                raise IOError(f"Failed to load original image: {self.original_image_path}")

            # Check if original is grayscale or color
            is_original_grayscale = len(original_image.shape) == 2
            
            # For ROI selection display, convert BGR to RGB if needed
            display_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if not is_original_grayscale else original_image

            # Get ROI from the user on the correctly colored original image
            roi = self._get_roi_from_user(display_original)
            
            # --- Load and process all images based on ROI and original image format ---
            if roi:
                x1, y1, x2, y2 = roi
                cropped_original = original_image[y1:y2, x1:x2]
                plot_title = "Image Comparison (Magnified Region)"
            else:
                cropped_original = original_image
                plot_title = "Image Comparison (Full View)"

            cropped_comps = []
            for path in self.comparison_paths:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise IOError(f"Failed to load comparison image: {path}")
                
                # --- Match format to Original Image ---
                is_comp_grayscale = len(img.shape) == 2
                
                if is_original_grayscale and not is_comp_grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif not is_original_grayscale and is_comp_grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Resize comparison image to match original *before* cropping
                if img.shape[:2] != original_image.shape[:2]:
                    img = cv2.resize(img, (original_image.shape[1], original_image.shape[0]))
                
                # Crop the processed comparison image
                if roi:
                    x1, y1, x2, y2 = roi
                    cropped_comps.append(img[y1:y2, x1:x2])
                else:
                    cropped_comps.append(img)
            
            self.create_academic_plot(cropped_original, cropped_comps, self.comparison_paths, plot_title)

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def create_academic_plot(original_image, comp_images, comp_paths, title):
        """Creates and displays the final matplotlib plot."""
        num_images = len(comp_images) + 1
        fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 6))
        if num_images == 1: axes = [axes]

        is_grayscale = len(original_image.shape) == 2
        
        # --- Plot Original Image ---
        # Convert BGR to RGB ONLY for display, or use grayscale map
        display_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if not is_grayscale else original_image
        cmap_original = 'gray' if is_grayscale else None
        axes[0].imshow(display_original, cmap=cmap_original)
        axes[0].set_title("Original (Before)") # Changed from "Ground Truth"
        axes[0].axis('off')

        # --- Plot Comparison Images ---
        for i, comp_image in enumerate(comp_images):
            ax = axes[i + 1]
            
            display_comp = cv2.cvtColor(comp_image, cv2.COLOR_BGR2RGB) if not is_grayscale else comp_image
            cmap_comp = 'gray' if is_grayscale else None
            ax.imshow(display_comp, cmap=cmap_comp)
            filename = comp_paths[i].split('/')[-1]
            ax.set_title(f"Deconvolved (After):\n{filename}")
            ax.axis('off')

            try:
                # Metrics are calculated against the original image
                psnr_val = psnr(original_image, comp_image, data_range=255)
                ssim_val = ssim(original_image, comp_image, data_range=255, channel_axis=-1 if not is_grayscale else None)
                mse_val = mse(original_image, comp_image)
                
                metrics_text = (f"PSNR: {psnr_val:.2f} dB\n"
                                f"SSIM: {ssim_val:.4f}\n"
                                f"MSE: {mse_val:.2f}")
            except Exception as e:
                metrics_text = f"Error in metrics:\n{e}"

            ax.text(0.5, -0.15, metrics_text, size=11, ha="center", 
                    transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparisonApp(root)
    root.mainloop()