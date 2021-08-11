Copyright (c) 2017, Tampere University of Technology (TUT)
This work is licensed under the MIT license / X11 license.
This documentation is licensed under CC0 license.


This folder contains the following Matlab/Octave files

 run_pos_est                                main file to run
 fnc_read_CSVdata_inMatlab.m                Load the database
 fnc_coverage_distrib_based                 pointwise coverage area algorithm
 fnc_coverage_pointwise		 	    coverage area defined with distribution algorithm
 fnc_eval_error				    evaluates error in a set of 3D position estimates
 fnc_get_data_properties		    picks defined outputs of rss_stats
 fnc_get_floor_heights			    finds the floor height values			
 fnc_get_knearest			    finds the k rows in an matrix data
 fnc_rss_stats				    compute positioning error statistics

The Matlab/Octave files that analyze the WLAN RSS fingerprint data provided
in FINGERPRINTING_DB folder. There are two implementations of a position
estimation algorithm:
    1) pointwise coverage area, and
    2) coverage area defined with distribution.

More details can be found in
Piché, R. Robust estimation of a reception region from location fingerprints.
In Proceedings of the 2011 International Conference on Localization and GNSS
(ICL-GNSS), Tampere, Finland, 29–30 June 2011; pp. 31–35. (DOI:
10.1109/ICL-GNSS.2011.5955261) and
Raitoharju, M.; Dashti, M.; Ali-Löytty, S.; Piché, R. Positioning with
Multilevel Coverage Area Models.  In Proceedings of the 2012 International
Conference on Indoor Positioning and Indoor Navigation (IPIN2012), Sydney,
Australia, 13–15 November 2012.


The obtained accuracies are:

1) pointwise coverage area (fnc_coverage_pointwise)
     mean_error_3D: 10.0397
     mean_error_2D: 9.4434
    floor_det_prob: 0.8664

2) coverage area defined with distribution (fnc_coverage_distrib_based)
     mean_error_3D: 13.0099
     mean_error_2D: 11.6827
    floor_det_prob: 0.6907




The software was tested with Matlab R2014b and R2016b and with Gnu Octave 4.2.1.

