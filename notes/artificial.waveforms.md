## Synthesizing waveforms  
in buildwaveforms.py I'm using the sample waveforms from the LeCroy /data\_fs/ave1/C1--LowPulseHighRes-in-100-out1700-an2100--\*.txt  
files to simulate the noise, and artificially dial real signal, and add the synthesized noise to the result  
Now I also have an energy2time() method so that next on deck is to randomply select from a probability   
distribution some energies and then convert that to times that then use the fourier\_delay() method  
to create a synthesized waveform.  

The thing left also to do is to then extend the waveforms by somehow sub-sampling the fourier domain into  
finer omega bins...  

Likely need to unwrap the phase for the inds that have signal and then just randomly choose amps and phases for the points  
that are between values for the noise region of the waveform.

