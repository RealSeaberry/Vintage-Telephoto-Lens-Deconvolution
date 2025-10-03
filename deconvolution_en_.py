import sys
import cv2
import numpy as np
# from scipy.signal import fftconvolve # Not used in wiener_deconvolution
import os
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog,
                             QSlider, QHBoxLayout, QLineEdit, QCheckBox, QMessageBox,
                             QProgressDialog) # Added QProgressDialog
from PyQt6.QtCore import Qt

def compute_snr(image):
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    image_gray = image_gray.astype(np.float32)
    signal_power = np.mean(image_gray ** 2)
    noise_map = cv2.Laplacian(image_gray, cv2.CV_32F)
    noise_power = np.mean(noise_map ** 2)
    if noise_power < 1e-8:
        return 100.0 # High SNR if noise is negligible
    return 10 * np.log10(signal_power / (noise_power + 1e-8))

def compute_adaptive_k(image, K0=0.1, alpha=0.1): # K0 and alpha can be fine-tuned
    snr = compute_snr(image)
    k_val = K0 * np.exp(-alpha * snr)
    return max(k_val, 1e-6) # Ensure K is not too small or negative

def wiener_deconvolution(img_channel, psf, k_value):
    img_h, img_w = img_channel.shape
    psf_h, psf_w = psf.shape
    
    psf_padded = np.zeros_like(img_channel, dtype=np.float32)
    
    pad_top = (img_h - psf_h) // 2
    pad_left = (img_w - psf_w) // 2

    if psf_h > img_h or psf_w > img_w:
        psf_cropped = psf[:img_h, :img_w] # Crop PSF if larger than image
        psf_h_c, psf_w_c = psf_cropped.shape
        pad_top_c = (img_h - psf_h_c) // 2
        pad_left_c = (img_w - psf_w_c) // 2
        psf_padded[pad_top_c:pad_top_c+psf_h_c, pad_left_c:pad_left_c+psf_w_c] = psf_cropped
    else:
        psf_padded[pad_top:pad_top+psf_h, pad_left:pad_left+psf_w] = psf

    psf_for_fft = np.fft.ifftshift(psf_padded)

    img_fft = np.fft.fft2(img_channel)
    psf_fft = np.fft.fft2(psf_for_fft)
    
    psf_fft_conj = np.conj(psf_fft)
    # Ensure k_value is positive for stability
    safe_k_value = max(k_value, 1e-8) 
    denominator = np.abs(psf_fft) ** 2 + safe_k_value
    
    # Explicitly handle cases where denominator might be zero (though safe_k_value should prevent this)
    denominator[denominator < 1e-9] = 1e-9 # Prevent division by very small numbers approaching zero
        
    result_fft = (img_fft * psf_fft_conj) / denominator
    result = np.fft.ifft2(result_fft).real
    
    return np.clip(result, 0, 255).astype(np.uint8)

class DeconvolutionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        
        self.image_label = QLabel("Select Input Image or Video:") # Keep this label generic
        self.image_path = QLineEdit()
        self.image_button = QPushButton("Browse")
        self.image_button.clicked.connect(self.load_media)
        
        self.psf_label = QLabel("Select PSF File (Image):")
        self.psf_path = QLineEdit()
        self.psf_button = QPushButton("Browse")
        self.psf_button.clicked.connect(self.load_psf)
        
        self.output_label = QLabel("Select Output Path:")
        self.output_path = QLineEdit()
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.set_output)
        
        self.auto_k_checkbox = QCheckBox("Automatically Calculate K Value")
        self.auto_k_checkbox.setChecked(True)
        self.auto_k_checkbox.toggled.connect(self.toggle_k_slider_state)


        self.k_slider = QSlider(Qt.Orientation.Horizontal)
        # Slider range 0-5000 maps to K value 0.0000-0.5000
        self.k_slider_divisor = 10000.0
        self.k_slider.setRange(0, 5000) 
        # Default K = 0.1 -> slider value 0.1 * 10000 = 1000
        self.k_slider.setValue(1000) 
        self.k_slider.valueChanged.connect(self.update_k_value_display)
        
        self.k_value_label = QLabel(f"Manual K Value: {self.k_slider.value() / self.k_slider_divisor:.4f}")
        
        self.process_button = QPushButton("Start Processing")
        self.process_button.clicked.connect(self.process_media)
        
        # Layout improvements
        layout.addWidget(self.image_label)
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_path)
        image_layout.addWidget(self.image_button)
        layout.addLayout(image_layout)

        layout.addWidget(self.psf_label)
        psf_layout = QHBoxLayout()
        psf_layout.addWidget(self.psf_path)
        psf_layout.addWidget(self.psf_button)
        layout.addLayout(psf_layout)

        layout.addWidget(self.output_label)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.output_button)
        layout.addLayout(output_layout)
        
        layout.addWidget(self.auto_k_checkbox)
        
        k_controls_layout = QHBoxLayout()
        self.manual_k_label_prefix = "Manual K Value: "
        k_controls_layout.addWidget(QLabel("Manual K Value Adjustment (0.0000 - 0.5000):"))
        layout.addLayout(k_controls_layout) # Add label for slider
        layout.addWidget(self.k_slider)
        layout.addWidget(self.k_value_label) # This label will show current K (manual or auto)
        
        layout.addWidget(self.process_button)
        self.setLayout(layout)
        self.setWindowTitle("Wiener Deconvolution Tool (Image/Video)")
        self.toggle_k_slider_state(self.auto_k_checkbox.isChecked()) # Initial state for slider

    def toggle_k_slider_state(self, checked):
        self.k_slider.setEnabled(not checked)
        if checked:
            self.k_value_label.setText("K value will be calculated automatically")
        else:
            self.update_k_value_display(self.k_slider.value())


    def load_media(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Image or Video", "", 
                                               "Media Files (*.png *.jpg *.jpeg *.tif *.mp4 *.avi *.mov);;"
                                               "All Files (*)")
        if fname:
            self.image_path.setText(fname)
            base, ext = os.path.splitext(fname)
            self.output_path.setText(f"{base}_deconvolved{ext}")
    
    def load_psf(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select PSF File", "", "Image Files (*.png *.jpg *.tif);;All Files (*)")
        if fname:
            self.psf_path.setText(fname)
    
    def set_output(self):
        input_path = self.image_path.text()
        suggested_path = self.output_path.text()
        default_filter = "All Files (*)"
        
        if input_path:
            _, input_ext = os.path.splitext(input_path.lower())
            video_exts = ['.mp4', '.avi', '.mov']
            img_exts = ['.png', '.jpg', '.jpeg', '.tif']

            if input_ext in video_exts:
                default_filter = f"Video Files (*{input_ext if input_ext else '.mp4'} *.mp4 *.avi *.mov);;All Files (*)"
            elif input_ext in img_exts:
                default_filter = f"Image Files (*{input_ext if input_ext else '.png'} *.png *.jpg);;All Files (*)"
        
        fname, _ = QFileDialog.getSaveFileName(self, "Select Save Path", suggested_path, default_filter)
        if fname:
            self.output_path.setText(fname)
    
    def update_k_value_display(self, value):
        # This function is connected to the slider, so only update if auto_k is OFF
        if not self.auto_k_checkbox.isChecked():
            self.k_value_label.setText(f"{self.manual_k_label_prefix}{value / self.k_slider_divisor:.4f}")
    
    def process_media(self):
        input_media_path = self.image_path.text()
        psf_image_path = self.psf_path.text()
        output_media_path = self.output_path.text()
        
        if not all([input_media_path, psf_image_path, output_media_path]):
            QMessageBox.warning(self, "Input Error", "Please make sure you have selected an input file, a PSF file, and an output path.")
            return
        
        try:
            psf = cv2.imread(psf_image_path, cv2.IMREAD_GRAYSCALE)
            if psf is None:
                QMessageBox.critical(self, "PSF Error", f"Could not load PSF file: {psf_image_path}")
                return
            psf = psf.astype(np.float32)
            psf_sum = psf.sum()
            if psf_sum < 1e-6:
                QMessageBox.warning(self, "PSF Warning", "The sum of the PSF is close to zero. A central impulse will be used as the PSF.")
                psf = np.zeros_like(psf, dtype=np.float32)
                if psf.shape[0]>0 and psf.shape[1]>0: # Check if psf has valid dimensions
                    psf[psf.shape[0]//2, psf.shape[1]//2] = 1.0
                else: # Fallback for 0-size psf (edge case)
                    QMessageBox.critical(self, "PSF Error", "Invalid PSF file dimensions.")
                    return

            else:
                 psf /= psf_sum

            _, input_ext = os.path.splitext(input_media_path.lower())
            _, output_ext = os.path.splitext(output_media_path.lower())

            video_formats = ['.mp4', '.avi', '.mov']
            image_formats = ['.png', '.jpg', '.jpeg', '.tif']
            
            is_video_input = input_ext in video_formats
            is_image_input = input_ext in image_formats
            is_video_output = output_ext in video_formats
            is_image_output = output_ext in image_formats

            if not (is_video_input or is_image_input):
                QMessageBox.critical(self, "Format Error", f"Unsupported input file format: {input_ext}")
                return
            if (is_video_input and not is_video_output) or \
               (is_image_input and not is_video_output and output_media_path): # Allow image output if output path implies image
                if is_video_input and not is_video_output: # Video in, image out
                     QMessageBox.critical(self, "Format Error", f"The input is a video ({input_ext}), but the output is selected as an image ({output_ext}). Please select a video output format or do not specify an output extension to process the first frame.")
                     return
                # No error if image in, video out, as that's less likely a mistake for this tool
                # (though could be warned)


            self.process_button.setEnabled(False)
            QApplication.processEvents()


            if is_video_input:
                cap = cv2.VideoCapture(input_media_path)
                if not cap.isOpened():
                    QMessageBox.critical(self, "Video Error", f"Could not open video file: {input_media_path}")
                    self.process_button.setEnabled(True)
                    return

                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0: # Handle cases where total_frames is not available or zero
                    QMessageBox.warning(self, "Video Information", "Could not get the total number of frames, the progress bar may be inaccurate.")
                    total_frames = 1 # Avoid division by zero for progress

                fourcc_map = {'.mp4': 'mp4v', '.avi': 'XVID', '.mov': 'mp4v'}
                fourcc_str = fourcc_map.get(output_ext, 'XVID') # Default to XVID
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                
                out_video = cv2.VideoWriter(output_media_path, fourcc, fps, (frame_width, frame_height))
                if not out_video.isOpened():
                    QMessageBox.critical(self, "Output Video Error", f"Could not create output video: {output_media_path}. Check the path and codec ('{fourcc_str}').")
                    cap.release()
                    self.process_button.setEnabled(True)
                    return
                
                progress_dialog = QProgressDialog("Processing video frames...", "Cancel", 0, total_frames, self)
                progress_dialog.setWindowTitle("Processing Video")
                progress_dialog.setModal(True) # Block other UI interaction
                progress_dialog.show()
                
                processed_frames = 0
                success = True
                while True:
                    if progress_dialog.wasCanceled():
                        success = False
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_float = frame.astype(np.float32)
                    
                    if self.auto_k_checkbox.isChecked():
                        k_value = compute_adaptive_k(frame_float)
                        # Update label for auto K only if auto is checked
                        self.k_value_label.setText(f"Automatic K Value (Frame {processed_frames+1}): {k_value:.6f}")

                    else:
                        k_value = self.k_slider.value() / self.k_slider_divisor
                        # Label for manual K is updated by slider's signal or toggle_k_slider_state
                    
                    QApplication.processEvents() 

                    channels = cv2.split(frame_float)
                    deconvolved_channels = [wiener_deconvolution(ch, psf, k_value) for ch in channels]
                    deconvolved_frame = cv2.merge(deconvolved_channels)
                    
                    out_video.write(deconvolved_frame)
                    processed_frames += 1
                    progress_dialog.setValue(processed_frames)

                progress_dialog.close()
                cap.release()
                out_video.release()
                
                # Reset K value label to its appropriate state after processing
                self.toggle_k_slider_state(self.auto_k_checkbox.isChecked())


                if success and processed_frames > 0:
                    QMessageBox.information(self, "Processing Complete", f"Video has been successfully processed ({processed_frames} frames) and saved to\n{output_media_path}")
                elif not success and processed_frames > 0 :
                     QMessageBox.warning(self, "Processing Canceled", f"Video processing was canceled by the user.\nA partially processed video ({processed_frames} frames) may have been saved to\n{output_media_path}")
                elif not success and processed_frames == 0:
                    QMessageBox.warning(self, "Processing Canceled", "Video processing was canceled before the first frame.")
                else: # No frames processed, but not cancelled (e.g. empty video)
                    QMessageBox.warning(self, "Processing Warning", f"Video processing completed, but no frames were processed. Please check the video file: {input_media_path}")


            else: # Image Processing
                img = cv2.imread(input_media_path)
                if img is None:
                    QMessageBox.critical(self, "Image Error", f"Could not load image file: {input_media_path}")
                    self.process_button.setEnabled(True)
                    return

                img_float = img.astype(np.float32)
                
                if self.auto_k_checkbox.isChecked():
                    k_value = compute_adaptive_k(img_float)
                    self.k_value_label.setText(f"Automatic K Value: {k_value:.6f}")
                else:
                    k_value = self.k_slider.value() / self.k_slider_divisor
                    # Label for manual K is updated by slider's signal or toggle_k_slider_state

                QApplication.processEvents()

                channels = cv2.split(img_float)
                deconvolved_channels = [wiener_deconvolution(ch, psf, k_value) for ch in channels]
                deconvolved_img = cv2.merge(deconvolved_channels)
                
                if not cv2.imwrite(output_media_path, deconvolved_img):
                    QMessageBox.critical(self, "Save Error", f"Could not save the image to: {output_media_path}")
                else:
                    QMessageBox.information(self, "Processing Complete", f"Image has been successfully processed and saved to\n{output_media_path}")
                # Reset K value label to its appropriate state after processing
                self.toggle_k_slider_state(self.auto_k_checkbox.isChecked())
        
        except Exception as e:
            QMessageBox.critical(self, "An Error Occurred", f"An error occurred during processing: {str(e)}")
            import traceback
            print(f"Error: {e}\n{traceback.format_exc()}") # Print stack trace to console for debugging
        finally:
            self.process_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = DeconvolutionApp()
    ex.show()
    sys.exit(app.exec())