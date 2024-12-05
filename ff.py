import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from PIL import Image
import librosa
import os
st.set_page_config(page_title="LAB3")
SAMPLING_DELTA = 0.0001
NUM_SAMPLES = 2**10

def calculate_fourier_coefficients(time_points, signal, num_harmonics, lower_limit, upper_limit):
    """
    Calculate Fourier coefficients for a periodic signal
    """
    period = upper_limit - lower_limit
    angular_freq = 2 * np.pi / period
    cosine_coeffs = np.zeros((num_harmonics, 1))
    sine_coeffs = np.zeros((num_harmonics, 1))
    num_points = len(time_points)
    
    # Calculate DC component (A0)
    dc_component = 0
    for i in range(1, num_points):
        dc_component += (1/period) * signal[i] * SAMPLING_DELTA
    
    # Calculate coefficients
    for harmonic in range(1, num_harmonics):
        for point in range(1, num_points):
            cosine_coeffs[harmonic] += ((2/period) * signal[point] * 
                                      np.cos(harmonic * angular_freq * time_points[point]) * SAMPLING_DELTA)
            sine_coeffs[harmonic] += ((2/period) * signal[point] * 
                                    np.sin(harmonic * angular_freq * time_points[point]) * SAMPLING_DELTA)
    
    return cosine_coeffs, sine_coeffs, dc_component

def compute_fourier_transform(signal):
    X_f = np.fft.fft(signal)
    X_fcorr = np.fft.fftshift(X_f)
    X_fcorr_mag = np.abs(X_fcorr)
    return X_fcorr_mag, X_fcorr

def perform_am_modulation(audio_file_path, carrier_freq, cutoff_freq):
    # Load audio
    x_t, fs = librosa.load(audio_file_path)
    t = np.arange(len(x_t)) / fs
    
    # Generate carrier
    carrier = np.cos(2*np.pi*carrier_freq*t)
    
    # Modulation
    y_mod = x_t * carrier
    DE_Y, C_Y = compute_fourier_transform(y_mod)
    
    # Demodulation
    x_prima = y_mod * carrier
    X_prima, X_C = compute_fourier_transform(x_prima)
    
    # Filtering
    delta_f = 1/(len(t)*(t[1] - t[0]))
    eje_f = np.arange(-len(t)/2, len(t)/2) * delta_f
    fbp = 1*(np.abs(eje_f) <= cutoff_freq)
    X_C_filt = fbp * X_C
    X_filt_shifted = np.fft.ifftshift(X_C_filt)
    x_filt = np.fft.ifft(X_filt_shifted)
    
    # Plot results
    fig,(ax1,ax2,ax3)=plt.subplots(3,1)
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    ax1.plot(t, x_t)
    plt.title("Original Signal")
    
    plt.subplot(3, 1, 2)
    ax2.plot(t, y_mod)
    plt.title("Modulated Signal")
    
    plt.subplot(3, 1, 3)
    ax3.plot(t, np.real(x_filt))
    plt.title("Demodulated Signal")
    plt.tight_layout()
    st.pyplot(fig)
    
    return np.real(x_filt)
def perform_qam(audio_file_path, carrier_freq):
    # Load audio
    x_t, fs = librosa.load(audio_file_path)
    t = np.arange(len(x_t)) / fs
    
    # Generate carriers
    p1 = np.cos(2*np.pi*carrier_freq*t)
    p2 = np.sin(2*np.pi*carrier_freq*t)
    
    # Modulate
    y1 = x_t * p1
    y2 = x_t * p2
    y3 = y1 + y2
    
    # Plot results
    fig,(ax1,ax2)= plt.subplots(2,1)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    ax1.plot(t, x_t)
    plt.title("Original Signal")
   
    
    plt.subplot(2, 1, 2)
    ax2.plot(t, y3)
    plt.title("QAM Signal")
    plt.tight_layout()
    st.pyplot(fig)
    
    return y3

def perform_dsb_lc(audio_file_path, carrier_freq, mu):
    # Load audio
    x_t, fs = librosa.load(audio_file_path)
    t = np.arange(len(x_t)) / fs
    
    # Generate carrier
    carrier = np.cos(2*np.pi*carrier_freq*t)
    
    # Modulation
    y_mod = x_t * carrier
    am = np.min(x_t)
    A = am/mu
    y_LC = y_mod + A*carrier
    
    # Envelope detection
    y_rec = np.abs(y_LC)
    
    # Plot results
    fig,(ax1,ax2,ax3)=plt.subplots(3)
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    ax1.plot(t, x_t)
    plt.title("Original Signal")
    
    plt.subplot(3, 1, 2)
    ax2.plot(t, y_LC)
    plt.title("DSB-LC Modulated Signal")
    
    plt.subplot(3, 1, 3)
    ax3.plot(t, y_rec)
    plt.title("Detected Signal")
    plt.tight_layout()
    st.pyplot(fig)
    
    return y_rec

def analyze_periodic_signal(signal_type, time_points, num_harmonics):
    """
    Analyze periodic signals using Fourier series
    """
    if signal_type == "triangular":
        # Example for triangular wave
        period = 2 * np.pi
        signal = np.piecewise(time_points, 
                            [time_points < 0, time_points >= 0], 
                            [lambda t: -t, lambda t: t])
    elif signal_type == "square":
        # Example for square wave
        signal = np.sign(np.sin(time_points))
    else:
        raise ValueError("Unsupported signal type")
    
    cosine_coeffs, sine_coeffs, dc_component = calculate_fourier_coefficients(
        time_points, signal, num_harmonics, -np.pi, np.pi)
    
    return cosine_coeffs, sine_coeffs, dc_component

def analyze_custom_signal(time_points, num_harmonics):
    """
    Analyze the custom piecewise linear signal:
    x(t) = 1 + 4t/T  for -T/2 < t ≤ 0
    x(t) = 1 - 4t/T  for 0 ≤ t < T/2
    """
    T = 2  # Periodo fundamental
    
    # Generate the piecewise signal
    signal = np.piecewise(time_points, 
                         [time_points < 0, time_points >= 0],
                         [lambda t: 1 + 4*t/T, lambda t: 1 - 4*t/T])
    
    # Initialize coefficients
    a0 = 0
    an = np.zeros(num_harmonics)
    bn = np.zeros(num_harmonics)
    w0 = 2*np.pi/T
    
    # Calculate a0 (mean value)
    def integrand_a0(t):
        return np.piecewise(t, [t < 0, t >= 0],
                          [lambda t: 1 + 4*t/T, lambda t: 1 - 4*t/T])
    
    a0 = 2/T * np.trapz([integrand_a0(t) for t in np.linspace(-T/2, T/2, 1000)],
                        np.linspace(-T/2, T/2, 1000))
    
    # Calculate an and bn coefficients
    for n in range(1, num_harmonics):
        # Cosine coefficients (an)
        def integrand_an(t):
            return integrand_a0(t) * np.cos(n*w0*t)
        
        an[n] = 2/T * np.trapz([integrand_an(t) for t in np.linspace(-T/2, T/2, 1000)],
                              np.linspace(-T/2, T/2, 1000))
        
        # Sine coefficients (bn)
        def integrand_bn(t):
            return integrand_a0(t) * np.sin(n*w0*t)
        
        bn[n] = 2/T * np.trapz([integrand_bn(t) for t in np.linspace(-T/2, T/2, 1000)],
                              np.linspace(-T/2, T/2, 1000))
    
    # Reconstruct signal
    reconstructed = a0/2 * np.ones_like(time_points)
    for n in range(1, num_harmonics):
        reconstructed += an[n]*np.cos(n*w0*time_points) + bn[n]*np.sin(n*w0*time_points)
    
    return signal, reconstructed, an, bn, a0/2

def analyze_sawtooth_signal(time_points, num_harmonics):
    """
    Analyze the sawtooth signal with period 2π:
    x(t) = t for -π < t ≤ π, repeating every 2π
    """
    T = 2 * np.pi  # Periodo fundamental
    
    # Generate the sawtooth signal
    signal = time_points % T
    mask = signal > np.pi
    signal[mask] -= T
    
    # Initialize coefficients
    a0 = 0
    an = np.zeros(num_harmonics)
    bn = np.zeros(num_harmonics)
    w0 = 2*np.pi/T
    
    # Calculate a0 (mean value)
    def integrand_a0(t):
        return t
    
    a0 = 2/T * np.trapz([integrand_a0(t) for t in np.linspace(-np.pi, np.pi, 1000)],
                        np.linspace(-np.pi, np.pi, 1000))
    
    # Calculate an and bn coefficients
    for n in range(1, num_harmonics):
        # Cosine coefficients (an)
        def integrand_an(t):
            return t * np.cos(n*w0*t)
        
        an[n] = 2/T * np.trapz([integrand_an(t) for t in np.linspace(-np.pi, np.pi, 1000)],
                              np.linspace(-np.pi, np.pi, 1000))
        
        # Sine coefficients (bn)
        def integrand_bn(t):
            return t * np.sin(n*w0*t)
        
        bn[n] = 2/T * np.trapz([integrand_bn(t) for t in np.linspace(-np.pi, np.pi, 1000)],
                              np.linspace(-np.pi, np.pi, 1000))
    
    # Reconstruct signal
    reconstructed = a0/2 * np.ones_like(time_points)
    for n in range(1, num_harmonics):
        reconstructed += an[n]*np.cos(n*w0*time_points) + bn[n]*np.sin(n*w0*time_points)
    
    return signal, reconstructed, an, bn, a0/2
def analyze_parabolic_signal(time_points, num_harmonics):
    """
    Analyze the periodic parabolic signal with period 2π:
    x(t) = t² - π² repeated every 2π (inverted parabola)
    """
    T = 2 * np.pi  # Periodo fundamental
    
    # Generate the parabolic signal
    signal = np.zeros_like(time_points)
    for i, t in enumerate(time_points):
        # Normalizar t al rango [-π, π]
        t_norm = t % T
        if t_norm > np.pi:
            t_norm -= T
        signal[i] = t_norm**2 - np.pi**2
    
    # Initialize coefficients
    a0 = 0
    an = np.zeros(num_harmonics)
    bn = np.zeros(num_harmonics)
    w0 = 2*np.pi/T
    
    # Calculate a0 (mean value)
    def integrand_a0(t):
        return t**2 - np.pi**2
    
    a0 = 2/T * np.trapz([integrand_a0(t) for t in np.linspace(-np.pi, np.pi, 1000)],
                        np.linspace(-np.pi, np.pi, 1000))
    
    # Calculate an and bn coefficients
    for n in range(1, num_harmonics):
        # Cosine coefficients (an)
        def integrand_an(t):
            return (t**2 - np.pi**2) * np.cos(n*w0*t)
        
        an[n] = 2/T * np.trapz([integrand_an(t) for t in np.linspace(-np.pi, np.pi, 1000)],
                              np.linspace(-np.pi, np.pi, 1000))
        
        # Sine coefficients (bn)
        def integrand_bn(t):
            return (t**2 - np.pi**2) * np.sin(n*w0*t)
        
        bn[n] = 2/T * np.trapz([integrand_bn(t) for t in np.linspace(-np.pi, np.pi, 1000)],
                              np.linspace(-np.pi, np.pi, 1000))
    
    # Reconstruct signal
    reconstructed = a0/2 * np.ones_like(time_points)
    for n in range(1, num_harmonics):
        reconstructed += an[n]*np.cos(n*w0*time_points) + bn[n]*np.sin(n*w0*time_points)
    
    return signal, reconstructed, an, bn, a0/2

def analyze_piecewise_new(time_points, num_harmonics):
    """
    Analyze the piecewise function:
    x(t) = t  for -1 < t < 0
    x(t) = 1  for  0 < t < 1
    """
    T = 2  # Periodo fundamental
    
    # Generate the piecewise signal
    signal = np.piecewise(time_points, 
                         [time_points < 0, time_points >= 0],
                         [lambda t: t, lambda t: 1])
    
    # Initialize coefficients
    a0 = 0
    an = np.zeros(num_harmonics)
    bn = np.zeros(num_harmonics)
    w0 = 2*np.pi/T
    
    # Calculate a0 (mean value)
    def integrand_a0(t):
        return np.piecewise(t, [t < 0, t >= 0], [lambda t: t, lambda t: 1])
    
    a0 = 2/T * np.trapz([integrand_a0(t) for t in np.linspace(-1, 1, 1000)],
                        np.linspace(-1, 1, 1000))
    
    # Calculate an and bn coefficients
    for n in range(1, num_harmonics):
        # Cosine coefficients (an)
        def integrand_an(t):
            return integrand_a0(t) * np.cos(n*w0*t)
        
        an[n] = 2/T * np.trapz([integrand_an(t) for t in np.linspace(-1, 1, 1000)],
                              np.linspace(-1, 1, 1000))
        
        # Sine coefficients (bn)
        def integrand_bn(t):
            return integrand_a0(t) * np.sin(n*w0*t)
        
        bn[n] = 2/T * np.trapz([integrand_bn(t) for t in np.linspace(-1, 1, 1000)],
                              np.linspace(-1, 1, 1000))
    
    # Reconstruct signal
    reconstructed = a0/2 * np.ones_like(time_points)
    for n in range(1, num_harmonics):
        reconstructed += an[n]*np.cos(n*w0*time_points) + bn[n]*np.sin(n*w0*time_points)
    
    return signal, reconstructed, an, bn, a0/2
def run_point5():
    signal_choice= st.selectbox("elige el tipo de señal",["1","2","3","4"])
    vis_choice= st.selectbox("elige la visualizacion",["1","2"])
    num_harmonics= st.slider("elige",min_value=1,max_value=20,step=1)
    if signal_choice == '1':
        time_points = np.linspace(-1, 1, 1000)
        original, reconstructed, an, bn, a0 = analyze_custom_signal(
            time_points, num_harmonics)
    if signal_choice == '2':
        time_points = np.linspace(-2*np.pi, 2*np.pi, 1000)
        original, reconstructed, an, bn, a0 = analyze_sawtooth_signal(
            time_points, num_harmonics)
    if signal_choice == '3':
        time_points = np.linspace(-3*np.pi, 3*np.pi, 1000)
        original, reconstructed, an, bn, a0 = analyze_parabolic_signal(
            time_points, num_harmonics)
    else:
        # Para la nueva función, mostramos 2 periodos
        time_points = np.linspace(-2, 2, 1000)
        original, reconstructed, an, bn, a0 = analyze_piecewise_new(
            time_points, num_harmonics)
            
    if vis_choice == '1':
        fig,(ax1,ax2)=plt.subplots(2)
        plt.figure(figsize=(12, 6))
        ax1.plot(time_points, original, 'b-', label='Original', linewidth=2)
        ax2.plot(time_points, reconstructed, 'r--', label=f'Reconstructed ({num_harmonics} harmonics)')
        plt.title('Original vs Reconstructed Signal')
        plt.xlabel('Time (t)')
        plt.ylabel('x(t)')
        plt.grid(True)
        
        # Ajustar etiquetas del eje x según el tipo de señal
        if signal_choice in ['2', '3']:
            plt.xticks(np.arange(min(time_points), max(time_points) + np.pi, np.pi),
                      [f'{int(x/np.pi)}π' if x != 0 else '0' for x in np.arange(min(time_points), max(time_points) + np.pi, np.pi)])
        
        plt.legend()
        st.pyplot(fig)
    else:
        fig,(ax,ax1,ax2)=plt.subplots(3)
        plt.figure(figsize=(10, 6))
        # Plot DC component
        ax.stem([0], [a0], 'b', label='DC Component')
        # Plot Fourier coefficients
        harmonics = np.arange(1, num_harmonics)
        ax1.stem(harmonics, np.abs(an[1:]), 'r', label='|an| (Cosine)')
        ax2.stem(harmonics, np.abs(bn[1:]), 'g', label='|bn| (Sine)')
        plt.title('Fourier Series Coefficients')
        plt.xlabel('n (Harmonic Number)')
        plt.ylabel('Coefficient Magnitude')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

def main():
    st.title("Tercer laboratorio")
    st.subheader(" Juan Polo C   Jesus Carmona   Samir Albor")
    choice=st.selectbox("selecciona",["Ninguno","1","4"])
    if choice in ['1', '2', '3']:
            try:
                file_path = "auido.wav"
                carrier_freq = st.slider("Enter carrier frequency (Hz, recommended 2000): ",min_value=0,max_value=2000,step=500)
                
                if choice == '1':
                    cutoff_freq = st.slider("Enter cutoff frequency (Hz, recommended 700): ",min_value=0,max_value=700,step=100)
                    perform_am_modulation(file_path, carrier_freq, cutoff_freq)
                    
                elif choice == '2':
                    perform_qam(file_path, carrier_freq)
                    
                elif choice == '3':
                    mod_index = float(input("Enter modulation index (recommended 0.8): "))
                    perform_dsb_lc(file_path, carrier_freq, mod_index)
                    
            except FileNotFoundError:
                print("Error: Audio file not found")
            except Exception as e:
                print(f"Error processing audio: {str(e)}")
    if choice == '4':
        run_point5()


# Modify process_audio_signal to accept signal and sample_rate directly
def process_audio_signal(input_signal, sample_rate, modulation_type='am', **params):
    """Process audio signal with specified modulation type"""
    time_points = np.arange(len(input_signal)) / sample_rate
    
    # Generate carrier
    carrier_freq = params.get('carrier_freq', 2000)
    carrier_signal = np.cos(2*np.pi*carrier_freq*time_points)

if __name__ == "__main__":
    main()