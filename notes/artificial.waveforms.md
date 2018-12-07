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

## generate\_distribution.py  
# fillcollection()  
This generates a synthetic distribution of photoelectrons, and Auger electrons from e.g. sigstar and pistar.  
We can play with the form of this later, for now, it returns a vector of energies with distributions for three 
different energies and widths, loosely inspired by Oxygen.

## buildwaveforms()  
#  energy2time(e,r=0,d1=5,d2=5,d3=30)  
This is based on the Ave's 3 region TOF design for producing an output time unit of ns for the   
input energy e in eV.  

 * first sum all the Weiner filtered and foureir\_delay() signals  
 * then add the single noise vector back  
 * then IFFT() back to time
