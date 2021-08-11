% run_pos_est

%% load data
% edit path to data file
clear all; close all; clc
warning off
path_to_csvfiles='../../../FINGERPRINTING_DB/'; %choose the path to where you saved the csv files (Test_rss_21Aug17.csv, etc.)
%convert the csv data into 2 Matlab cell structures: one for the training and
%one for the test data
[WLAN_training, WLAN_test, coordinate_t, coordinate_e, N_AP, floor_heights]=fnc_read_CSVdata_inMatlab(path_to_csvfiles);


%% estimate and evaluate  (pointwise coverage)
% * parameters
pAPmatch = 0.9;
% * process
res  =  fnc_coverage_pointwise( WLAN_training, WLAN_test, pAPmatch )


%% estimate and evaluate  (coverage area)
% * parameters
par.distribution = 'gaussian' % 'gaussian' or 'student' 
par.nu = 5; % degrees of freedom
% * process
res  =  fnc_coverage_distrib_based( WLAN_training, WLAN_test, par )

