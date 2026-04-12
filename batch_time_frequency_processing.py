import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import multiprocessing
from tqdm import tqdm

class BatchTimeFrequencyProcessor:
    """
    Batch processor for converting CWRU bearing vibration data to time-frequency diagrams
    """
    
    def __init__(self, base_dir):
        """
        Initialize processor with base directory
        
        Args:
            base_dir (str): Base directory containing CWRU dataset
        """
        self.base_dir = Path(base_dir)
        
    def get_sampling_frequency(self, file_path):
        """
        Determine sampling frequency based on file path
        
        Args:
            file_path (Path): Path to .mat file
            
        Returns:
            int: Sampling frequency in Hz
        """
        path_str = str(file_path)
        if '12k' in path_str:
            return 12000
        elif '48k' in path_str:
            return 48000
        else:
            # Default to 12k if cannot determine
            return 12000
    
    def load_vibration_data(self, file_path):
        """
        Load vibration data from MATLAB file
        
        Args:
            file_path (Path): Path to .mat file
            
        Returns:
            tuple: (data, sampling_freq) or (None, None) if error
        """
        try:
            mat_data = sio.loadmat(str(file_path))
            
            # Find the vibration data array
            for key in mat_data.keys():
                if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                    data = mat_data[key].flatten()
                    sampling_freq = self.get_sampling_frequency(file_path)
                    return data, sampling_freq
            
            return None, None
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def generate_enhanced_spectrogram(self, data, sampling_freq, method='stft'):
        """
        Generate enhanced time-frequency diagram
        
        Args:
            data (numpy.ndarray): Vibration signal
            sampling_freq (int): Sampling frequency
            method (str): 'stft', 'cwt', or 'hht'
            
        Returns:
            matplotlib.figure.Figure: Generated figure
        """
        fig = plt.figure(figsize=(15, 10))
        
        if method == 'stft':
            # Enhanced STFT with multiple window sizes
            self._plot_enhanced_stft(fig, data, sampling_freq)
        elif method == 'cwt':
            self._plot_cwt_analysis(fig, data, sampling_freq)
        elif method == 'hht':
            self._plot_hht_analysis(fig, data, sampling_freq)
        
        return fig
    
    def _plot_enhanced_stft(self, fig, data, sampling_freq):
        """Plot enhanced STFT analysis with multiple window sizes"""
        
        # Time domain signal
        ax1 = fig.add_subplot(3, 2, 1)
        time = np.arange(len(data)) / sampling_freq
        ax1.plot(time, data, 'b-', linewidth=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Time Domain Signal')
        ax1.grid(True, alpha=0.3)
        
        # Frequency domain (FFT)
        ax2 = fig.add_subplot(3, 2, 2)
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/sampling_freq)
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_data[:len(fft_data)//2])
        ax2.semilogy(positive_freqs, positive_fft, 'r-')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Frequency Domain (FFT)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, sampling_freq//2)
        
        # STFT with small window (high time resolution)
        ax3 = fig.add_subplot(3, 2, 3)
        nperseg = 128
        frequencies, times, spectrogram = signal.spectrogram(
            data, fs=sampling_freq, nperseg=nperseg, noverlap=nperseg//2
        )
        im1 = ax3.pcolormesh(times, frequencies, 10*np.log10(spectrogram), 
                           shading='gouraud', cmap='viridis')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_title(f'STFT (Window: {nperseg} samples)')
        plt.colorbar(im1, ax=ax3, label='Power (dB)')
        
        # STFT with medium window (balanced)
        ax4 = fig.add_subplot(3, 2, 4)
        nperseg = 512
        frequencies, times, spectrogram = signal.spectrogram(
            data, fs=sampling_freq, nperseg=nperseg, noverlap=nperseg//2
        )
        im2 = ax4.pcolormesh(times, frequencies, 10*np.log10(spectrogram), 
                           shading='gouraud', cmap='viridis')
        ax4.set_ylabel('Frequency (Hz)')
        ax4.set_xlabel('Time (s)')
        ax4.set_title(f'STFT (Window: {nperseg} samples)')
        plt.colorbar(im2, ax=ax4, label='Power (dB)')
        
        # STFT with large window (high frequency resolution)
        ax5 = fig.add_subplot(3, 2, 5)
        nperseg = 1024
        frequencies, times, spectrogram = signal.spectrogram(
            data, fs=sampling_freq, nperseg=nperseg, noverlap=nperseg//2
        )
        im3 = ax5.pcolormesh(times, frequencies, 10*np.log10(spectrogram), 
                           shading='gouraud', cmap='viridis')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Frequency (Hz)')
        ax5.set_title(f'STFT (Window: {nperseg} samples)')
        plt.colorbar(im3, ax=ax5, label='Power (dB)')
        
        # Envelope analysis for bearing fault detection
        ax6 = fig.add_subplot(3, 2, 6)
        analytic_signal = signal.hilbert(data)
        amplitude_envelope = np.abs(analytic_signal)
        ax6.plot(time, amplitude_envelope, 'g-', linewidth=0.5)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Envelope Amplitude')
        ax6.set_title('Amplitude Envelope')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def _plot_cwt_analysis(self, fig, data, sampling_freq):
        """Plot Continuous Wavelet Transform analysis"""
        
        # Time domain
        ax1 = fig.add_subplot(2, 2, 1)
        time = np.arange(len(data)) / sampling_freq
        ax1.plot(time, data)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Time Domain Signal')
        ax1.grid(True)
        
        # CWT scalogram
        ax2 = fig.add_subplot(2, 2, 2)
        widths = np.arange(1, 128)
        cwtmatr = signal.cwt(data, signal.ricker, widths)
        im = ax2.imshow(np.abs(cwtmatr), extent=[0, len(data)/sampling_freq, 
                                               widths[0], widths[-1]], 
                      aspect='auto', cmap='viridis', origin='lower')
        ax2.set_ylabel('Scale')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('CWT Scalogram')
        plt.colorbar(im, ax=ax2, label='Magnitude')
        
        # Frequency representation at different scales
        ax3 = fig.add_subplot(2, 2, 3)
        selected_scales = [10, 30, 50, 70, 100]
        for scale in selected_scales:
            if scale < len(widths):
                ax3.plot(time, np.abs(cwtmatr[scale-1, :]), 
                        label=f'Scale {scale}')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Magnitude')
        ax3.set_title('CWT Coefficients at Different Scales')
        ax3.legend()
        ax3.grid(True)
        
        # Scale-frequency relationship
        ax4 = fig.add_subplot(2, 2, 4)
        center_freqs = [sampling_freq / (2 * scale) for scale in widths]
        ax4.semilogy(widths, center_freqs)
        ax4.set_xlabel('Scale')
        ax4.set_ylabel('Center Frequency (Hz)')
        ax4.set_title('Scale-Frequency Relationship')
        ax4.grid(True)
        
        plt.tight_layout()
    
    def _plot_hht_analysis(self, fig, data, sampling_freq):
        """Plot Hilbert-Huang Transform analysis"""
        
        # Time domain
        ax1 = fig.add_subplot(2, 2, 1)
        time = np.arange(len(data)) / sampling_freq
        ax1.plot(time, data)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Time Domain Signal')
        ax1.grid(True)
        
        # Hilbert transform - instantaneous frequency
        ax2 = fig.add_subplot(2, 2, 2)
        analytic_signal = signal.hilbert(data)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) / 
                                 (2.0 * np.pi) * sampling_freq)
        instantaneous_frequency = np.append(instantaneous_frequency, 
                                           instantaneous_frequency[-1])
        
        ax2.plot(time, instantaneous_frequency, 'r-')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title('Instantaneous Frequency')
        ax2.set_ylim(0, sampling_freq//2)
        ax2.grid(True)
        
        # Amplitude envelope
        ax3 = fig.add_subplot(2, 2, 3)
        amplitude_envelope = np.abs(analytic_signal)
        ax3.plot(time, amplitude_envelope, 'g-')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Amplitude Envelope')
        ax3.grid(True)
        
        # Hilbert spectrum (time-frequency representation)
        ax4 = fig.add_subplot(2, 2, 4)
        scatter = ax4.scatter(time, instantaneous_frequency, 
                             c=amplitude_envelope, cmap='viridis', s=1)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Instantaneous Frequency (Hz)')
        ax4.set_title('Hilbert Spectrum')
        ax4.set_ylim(0, sampling_freq//2)
        plt.colorbar(scatter, ax=ax4, label='Amplitude')
        
        plt.tight_layout()
    
    def process_single_file(self, file_info):
        """Process a single file and save time-frequency diagram"""
        file_path, output_dir, method = file_info
        
        # Load data
        data, sampling_freq = self.load_vibration_data(file_path)
        if data is None:
            return False
        
        # Create output path
        relative_path = file_path.relative_to(self.base_dir)
        output_path = output_dir / relative_path.with_suffix('.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate time-frequency diagram
        try:
            fig = self.generate_enhanced_spectrogram(data, sampling_freq, method)
            
            # Add title with file information
            fault_type = relative_path.parts[1] if len(relative_path.parts) > 1 else "Unknown"
            fault_severity = relative_path.parts[2] if len(relative_path.parts) > 2 else "Unknown"
            title = f"{fault_type} - {fault_severity} - {relative_path.stem}"
            fig.suptitle(title, fontsize=12, y=0.98)
            
            # Save figure
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    def process_dataset(self, output_base_dir, method='stft', num_processes=None):
        """
        Process entire dataset
        
        Args:
            output_base_dir (str): Base directory for output
            method (str): Time-frequency method
            num_processes (int): Number of parallel processes
        """
        output_dir = Path(output_base_dir) / f"time_frequency_{method}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all .mat files
        mat_files = list(self.base_dir.rglob('*.mat'))
        print(f"Found {len(mat_files)} .mat files")
        
        # Prepare file info for processing
        file_infos = [(file_path, output_dir, method) for file_path in mat_files]
        
        # Process files
        if num_processes is None:
            num_processes = min(multiprocessing.cpu_count(), 8)
        
        print(f"Processing with {num_processes} processes...")
        
        successful = 0
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(self.process_single_file, file_infos), 
                             total=len(file_infos)))
            successful = sum(results)
        
        print(f"Successfully processed {successful}/{len(mat_files)} files")
        print(f"Output saved to: {output_dir}")

def main():
    """Main function for batch processing"""
    
    base_dir = r"c:\Users\M\Desktop\软件\cwru_data"
    output_base_dir = r"c:\Users\M\Desktop\软件\cwru_data\time_frequency_results"
    
    processor = BatchTimeFrequencyProcessor(base_dir)
    
    # Process with different methods
    methods = ['stft', 'cwt', 'hht']
    
    for method in methods:
        print(f"\n=== Processing with {method.upper()} method ===")
        processor.process_dataset(output_base_dir, method=method, num_processes=4)
    
    print("\nBatch processing completed!")

if __name__ == "__main__":
    main()