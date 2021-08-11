% Copyright © 2017 J. Torres-Sospedra,  Universitat Jaume I (UJI)
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


% Example of using knn_ips
% knn_ips( database_orig , k, datarep, defaultNonDetectedValue, newNonDetectedValue , distanceMetric )


% Loads the database in the struct with predefined format
database = loadDatabase('../../FINGERPRINTING_DB');


k = 1; % k-Value for the k-NN algorithm, number of neighbors

datarep = 'positive'; % Representation of RSSI values for fingerprinting
                      % Possible values 'positive', 'exponential, 'powed'
                      % See the following paper for more information:
                      
                      % Torres-Sospedra, J.; Montoliu, R.; Trilles, S.;
                      %                   Belmonte, O. & Huerta, J. 
                      % Comprehensive Analysis of Distance and Similarity 
                      % Measures for Wi-Fi Fingerprinting Indoor
                      % Positioning Systems
                      % Expert Systems with Applications Vol. 42(23),
                      % pp. 9263-9278, 2015. doi


defaultNonDetectedValue = 100;  % Numeric value representing non-heard AP

newNonDetectedValue     = -103; % New value representing non-heard AP
                                % e.g. -103, -150, -200 ...
                                
distanceMetric          = 'distance_sorensen'
                            % Distance metric in the feature space (RSSI).
                            % Accepted metrics 'euclidean' and 'manhattan'
                            % Custom metrics can be defined as matlab
                            % functions. e.g. 'distance_sorensen' which
                            % used the metric in 'distance_sorense.m' file
                            % Custom metrics must begin with 'distance_'.
                      
results  = knn_ips( database , k, datarep, ...
                    defaultNonDetectedValue, newNonDetectedValue , ...
                    distanceMetric );
                
                
%results is a struct element that contains the following fields.
%      error: [nx4 double] 2D, 3D, height and floor error
% prediction: [nx4 double] Predicted location - 3D coordinates + floor
%    targets: [nx4 double] Real location - 3D coordinates + floor
% candidates: [nx1 double] Number of candidates used to estimate the position
%  distances: [nx1 double] Distance in FS of the best candidate
