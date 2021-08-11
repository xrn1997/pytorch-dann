% Copyright © 2017 Tampere University of Technology (TUT)
%
% Permission is hereby granted, free of charge, to any person obtaining a copy of
% this software and associated documentation files (the "Software"), to deal in
% the Software without restriction, including without limitation the rights to
% use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
% of the Software, and to permit persons to whom the Software is furnished to do
% so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

function [ WLAN_training, WLAN_test, coordinates_t, coordinates_e, N_AP, floor_heights]=Fct_read_CSVdata_inMatlab(path_to_csvfiles)
%path_to_csvfiles  = path to where the csv files were stored on the computer, e.g., D:\TUT_MEASUREMENT_DATA\
addpath(path_to_csvfiles)
% load  training data
rss_dB_t = importdata('Training_rss_21Aug17.csv');
coordinates_t = importdata('Training_coordinates_21Aug17.csv');
device_t = importdata('Training_device_21Aug17.csv', '\t');
date_t = importdata('Training_date_21Aug17.csv', '\t');
floor_heights=unique(coordinates_t(:,3))'; 

% load  test (or estimation) data
rss_dB_e = importdata('Test_rss_21Aug17.csv');
coordinates_e = importdata('Test_coordinates_21Aug17.csv');
device_e = importdata('Test_device_21Aug17.csv', '\t');
date_e = importdata('Test_date_21Aug17.csv', '\t');

%write the data into two cell structures, one for training and one for test
%data
Nm_t=size(coordinates_t,1); %number of measurements in  the training data
Nm_e=size(coordinates_e,1); %number of measurements in  the test data
N_AP=size(rss_dB_t,2);  %number of 'Access Points' (or MAC addresses) in the building. 
WLAN_training=cell(Nm_t,4);
WLAN_test=cell(Nm_e,4);
for jt=1:Nm_t
    WLAN_training{jt,1}=coordinates_t(jt,:);
    %find which APs were heard in this measurement;
    HeardAP=find( rss_dB_t(jt,:)~=100);
    WLAN_training{jt,2}(1,:)=HeardAP;
    WLAN_training{jt,2}(2,:)=rss_dB_t(jt,HeardAP);
    WLAN_training{jt,3}=date_t(jt,:);
    WLAN_training{jt,4}=device_t(jt,:);
end;
for je=1:Nm_e
    WLAN_test{je,1}=coordinates_e(je,:);
    %find which APs were heard in this measurement;
    HeardAP=find( rss_dB_e(je,:)~=100);
    WLAN_test{je,2}(1,:)=HeardAP;
    WLAN_test{je,2}(2,:)=rss_dB_e(je, HeardAP);
    WLAN_test{je,3}=date_e(je,:);
    WLAN_test{je,4}=device_e(je,:);
end;



