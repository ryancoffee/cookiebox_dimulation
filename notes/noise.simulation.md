## Simulating the noise  
For each example file, I'm reading in the triggered waveform,  
then I'm fourier transforming it (scaled to be in units of ns (time) and GHz (freq)   
Next I choose the noise band to be between 6.5 GHz and 20GHz as per figure plotting.fourier.png   
and then I replace the fourier values in the noise vector (a deep copy of the signal FT vector) for   
indices below 6.5GHz with the complex values randomly chosen from the 6.5GHz--20GHz range.   

## Weiner filter to get the signal  
First we will Weiner filter to get the nearly noise free signal   
Then we will add a phase in the fourier domain according to the energy dependent delay   
Then we will scale the amplitude by a random scale factor (copy from the 2dtimetool\_simulation)  

