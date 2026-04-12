import numpy as np 
import scipy.signal 
import torch 
 
signal = np.random.randn(1024) 
f, t, Zxx = scipy.signal.stft(signal, nperseg=126, noverlap=110) 
print(Zxx.shape) 
 
img = np.abs(Zxx) 
print("Before crop:", img.shape) 
img = img[:, :64] 
print("After crop:", img.shape)
