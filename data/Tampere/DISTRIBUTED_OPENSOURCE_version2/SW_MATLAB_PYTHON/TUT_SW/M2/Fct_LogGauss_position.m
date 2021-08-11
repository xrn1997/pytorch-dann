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

function [position, cost_function, quality] = Fct_LogGauss_position(WLAN_training, WLAN_current, AP_count, sigma, N)
%supporting function for the log-Gaussian likelihood estimation
% WLAN_training: RSS-fingerprints
% WLAN_current: A vector of the RSS for the current position
% AP_count: Number of access points
% sigma: Shadowing standard deviation in the Gaussian similarity function
% N: Number of nearest neighbour points to average for the result

% Number of grid points
GP_count = size(WLAN_training, 1);

% Generate RSS training matrix
RSS_training = NaN * ones(GP_count, AP_count);
for i = 1:GP_count
    fingerprint = WLAN_training{i,2};
    RSS_training(i, fingerprint(1,:)) = fingerprint(2,:);
end

% Generate equivalent test matrix
RSS_current = NaN * ones(1, AP_count);
RSS_current(WLAN_current{1,2}(1,:)) = WLAN_current{1,2}(2,:);
RSS_current = repmat(RSS_current, GP_count, 1);

% Associate each point a probability / cost corresponding to the
% likelihood that the current RSS vector corresponds to that position

% Probability for each AP in each point using Gaussian similarity
likelihood_matrix = 1/sqrt(2*pi*sigma^2)*exp(-(RSS_training - RSS_current).^2/(2*sigma^2));

% Replace zeros with a very small value
likelihood_matrix(isnan(likelihood_matrix)) = 1e-6;

% Use logaritmic probabilities for incresed efficiency and stability
likelihood_matrix = log(likelihood_matrix);

% Combine individual probabilities
cost_function = sum(likelihood_matrix,2);

% Estimate position

% Find N points with the highest probability and average their positions
[~, id] = sort(cost_function, 'descend');

N = min(N,GP_count);

position = cell2mat(WLAN_training(id(1:N),1));
position = mean(position, 1);

% Total likelihood normalized
quality = sum(cost_function(id(1:N))) / N;
end
