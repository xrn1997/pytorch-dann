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

function [pos_est]=Fct_LogGauss_FPestimate(N_AP, WLAN_data_per_synthpoint, user_data_per_measpoint,sigma_shad_dB, Nneigh)
%log Gaussian likelihood position estimation 


% Remove empty fingerprints
to_delete = false(length(WLAN_data_per_synthpoint),1);

for i = 1:length( WLAN_data_per_synthpoint)
   if isempty( WLAN_data_per_synthpoint{i,2})
       to_delete(i) = true;
   end
end

 WLAN_data_per_synthpoint(to_delete,:) = [];

%
% Estimate location for all test data
%
test_count = length(user_data_per_measpoint);


pos_est=NaN*ones(test_count, 3);
for i = 1:test_count
    % If the RSS is empty skip since it can't be positioned
    if isempty(user_data_per_measpoint{i,2})
        continue;
    end

    % Estimate current position
    [pos_est(i,:)] = Fct_LogGauss_position( WLAN_data_per_synthpoint, ... 
            user_data_per_measpoint(i,:), N_AP, sigma_shad_dB, Nneigh);
%     if rem(i,200)==0,
%         i
%     end;
end

