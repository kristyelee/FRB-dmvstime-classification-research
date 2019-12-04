# Fast Radio Burst (FRB) Classification Using the DM vs. Time Plot: Distinguishing FRBs from Radio Frequency Interference)
Kristy Lee, Berkeley SETI Research Center, Fall 2019

## Introduction
Fast radio bursts are transient radio signals that result from high energy, yet to be comprehended astrophysical processes in space; and thus there exists the possibility they may be linked to signs of extraterrestrial life, which cause them to be of interest to research. One distinguishing characteristic of FRBs is that they have a large dispersion measure (DM) in comparison to other pulses in space, which I utilize to my advantage in research. The purpose of my research is to train a convolutional neural network model to identify and distinguish the rarely occuring FRBs from noise -- radio frequency interference, or RFI -- in images of space. 

## Observation
We observe that we can use the DM characteristic of an FRB to detect FRBs in images containing RFI. We gathered and calculated the frequency vs. time data from signals and artificial, simulated FRBs alike to display a decreasing, concave up curve where high frequency corresponds to a shorter time period and the signal's frequency decreases as time passes. If the DM is high, the frequency vs. time plot of a signal is spread out among a longer period of time. Each plot of frequency vs. time corresponds to the following proportion: 

<a href="https://www.codecogs.com/eqnedit.php?latex=(t_H&space;-&space;t_L)&space;\propto&space;DM&space;(\frac{1}{F_L^2}&space;-&space;\frac{1}{F_H^2})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(t_H&space;-&space;t_L)&space;\propto&space;DM&space;(\frac{1}{F_L^2}&space;-&space;\frac{1}{F_H^2})" title="(t_H - t_L) \propto DM (\frac{1}{F_L^2} - \frac{1}{F_H^2})" /></a>

where DM is the DM of the signal. 

## Methodology
We can dedisperse the signals as follows: shift the signal at each frequency channel left such that we have collapsed the signal to a single column located at t_H



## Acknowledgements
- Vishal Gajjar for mentorship throughout this project
- Dominic LeDuc for working on the the project with me and generating simulated FRBs that I used to train the convolutional neural network
- Liam Connor and Joeri van Leeuwen for reference material for research through paper "Applying Deep Learning to Fast Radio Bursts"
- Liam Connor and Joeri van Leeuwen for reference material for research through paper "Applying Deep Learning to Fast Radio Bursts"
- Agarwal et. al. for reference material for research through paper "Towards deeper neural networks for Fast Radio Burst detection"
