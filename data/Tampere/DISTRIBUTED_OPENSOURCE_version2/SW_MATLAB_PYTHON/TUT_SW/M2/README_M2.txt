Copyright (c) 2017, Tampere University of Technology (TUT)
This work is licensed under the MIT license / X11 license.
This documentation is licensed under CC0 license.


This folder contains the following Matlab/Octave files

 Main_MatlabFile_dataanalysis.m             Main file to run analyzes and visualization
 Fct_2Dand3Dstatistics_on_posest.m          Compute positioning error statistics
 Fct_convertdatapergrid_toAPwise_inpNAP.m   Convert structure of database (DB)
 Fct_LogGauss_FPestimate.m                  Wrapper around Log-Gaussian positioning algo.
 Fct_LogGauss_position.m                    Log-Gaussian positioning algorithm
 Fct_plot_Powermaps3Ddata_perfloor.m        Plot power map
 Fct_plotstats_perAP.m                      Plot statistics per AP
 Fct_read_CSVdata_inMatlab.m                Load the database
 Fct_weighted_centroid.m                    Weighted centroid positioning algorithm
 Fct_weightedcentroid_posestim.m            Wrapper around weighted centroid pos. algo.



The Matlab/Octave files visualize and analyze WLAN RSS fingerprint data
provided in the FINGERPRINT_DB folder. There is only one main Matlab/Octave
file: Main_MatlabFile_dataanalysis.m. The rest are supporting functions called
by the main file.

The Main_MatlabFile_dataanalysis.m file analyzes the measurement data, both in
a visual way and it provides two position estimation algorithms as benchmark
for the provided data: a weighted centroid and a log-Gaussian likelihood
position estimator.

How to call this file:
Main_MatlabFile_dataanalysis

Note: the file running takes several minutes because of the log-Gaussian probability estimator pos_estFP.


The software was tested with Matlab R2016b and GNU Octave 4.2.1.

