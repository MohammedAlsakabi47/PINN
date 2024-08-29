# Physics Informed MLP for Linear Velocity Estimation via Doppler Effect
![Description of GIF](results_GIF.gif)

This repository presents a simulation of a Doppler effect caused by a two-blade propeller with a linear speed of 100m/s. The circular movement of the propeller causes the frequency change on an emitted waveform, with a carrier frequency of 3GHz, from a transmitter placed in the direction of the 'Receiver' in the GIF above. The instantaneous linear velocity of the propeller is then predicted with a three-layer MLP whose input is the noisy signal depicted on 'Wave at Receiver'.

## Dataset
Under `Dataset`, the input signal is found in `input_waves.mat`, the instantaneous spectrum is found in `forier_response.mat`, and the ground truth frequency is found in `frequency_progression.mat`.
