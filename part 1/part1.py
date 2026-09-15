import torch
import time
import matplotlib.pyplot as plt

N = 2048
T = 1.0
f0 = 1.0

def square_wave_fourier_pt(t, f0, N_harmonics):
    result = torch.zeros_like(t)
    for k in range(N_harmonics):
        n = 2 * k + 1
        result += torch.sin(2 * torch.pi * f0 * t * n) / n
    return (4 / torch.pi) * result

def naive_dft_pt(x, device='cpu'):
    N = len(x)
    X = torch.zeros(N, dtype=torch.complex64, device=device)
    x_complex = x.to(dtype=torch.complex64, device=device)
    
    for k in range(N):
        for n in range(N):
            angle = -1j * 2 * torch.pi * k * n / N
            X[k] += x_complex[n] * torch.exp(torch.tensor(angle, dtype=torch.complex64, device=device))
    return X

def vectorized_dft_pt(x, device):
    N = len(x)
    x_complex = x.to(dtype=torch.complex64, device=device)
    
    n = torch.arange(N, device=device)
    k = n.view(-1, 1)
    
    exponent = (-1j * 2 * torch.pi * k * n / N).to(torch.complex64)
    M = torch.exp(exponent)
    X = torch.mv(M, x_complex)
    return X

def sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Running on device: {device}")

    t = torch.linspace(0.0, T, N + 1)[:-1] 
    signal = square_wave_fourier_pt(t, f0, 50) 

    print("--- Timing DFT Methods ---")
    
    start_time = time.time()
    naive_result = naive_dft_pt(signal, device='cpu')
    print(f"1. Naive DFT (CPU) Time:       {time.time() - start_time:.4f} seconds")

    signal_device = signal.to(device)
    start_time = time.time()
    vectorized_result = vectorized_dft_pt(signal_device, device=device)
    sync_device(device)
    print(f"2. Vectorized DFT ({device.type.upper()}) Time:  {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    fft_result = torch.fft.fft(signal_device)
    sync_device(device)
    print(f"3. PyTorch Built-in FFT Time:  {time.time() - start_time:.4f} seconds")

    print("\nAre vectorized and FFT results close?", 
          torch.allclose(vectorized_result.cpu(), fft_result.cpu(), atol=1e-4))

    xf = torch.fft.fftfreq(N, d=T/N)[:N//2]
    magnitude = (2.0 / N * torch.abs(fft_result[:N//2])).cpu()

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(t.cpu().numpy(), signal.cpu().numpy(), color='c')
    ax1.set_title('Input Square Wave (50 Harmonics)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, 1.0)

    ax2.stem(xf.numpy(), magnitude.numpy(), basefmt=" ")
    ax2.set_title('Discrete Fourier Transform (Magnitude Spectrum)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlim(0, 50)

    plt.tight_layout()
    plt.savefig('part1_dft_plot.png')
    print("\nPlot saved as 'part1_dft_plot.png'")

if __name__ == "__main__":
    main()