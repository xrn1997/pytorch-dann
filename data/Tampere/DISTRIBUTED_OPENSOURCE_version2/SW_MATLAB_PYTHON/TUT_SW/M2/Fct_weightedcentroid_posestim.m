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

function [Pos_est] = Fct_weightedcentroid_posestim( WLAN_training_perAP, WLAN_test_per_measpoint )
% Fct_weightedcentroid_posestim( WLAN_training_perAP, WLAN_test_per_measpoint )
% Function which estimates the 3D position based on weighted centroid
% estimation
%INPUTS:
%WLAN_training_perAP is the training matrix in the form of cell pe Access Point (AP); size 1x N_AP
%WLAN_test_per_measpoint is the Nu x 2 cell, showing the user test track
%measurements in each user point out of Nu user points
%OUTPUTS
%Pos_est  = the 3D position estimates of all the test data

 N_AP=length(WLAN_training_perAP);
 %Weighted centroid for APs
 XYZ_APest=NaN*ones(N_AP,3);
 for ap=1:N_AP,
    HeardPoints=WLAN_training_perAP{ap}; %heard points by each ap
    if ~isempty(HeardPoints),
        RSSheard_lin=10.^(HeardPoints(:,4)/10);
        [XYZ_APest(ap,:)] = Fct_weighted_centroid( HeardPoints(:,1:3), RSSheard_lin);
    end;
    clear HeardPoints RSSheard*
 end;
 %% 
 %weighted centroid (classical approach) for track (estimation points)
 Nu=size(WLAN_test_per_measpoint,1);
 Pos_est=NaN*ones(Nu,3);
 for uu=1:Nu
     if ~isempty(WLAN_test_per_measpoint{uu,2}),
             AP_heardindices=WLAN_test_per_measpoint{uu,2}(1,:);
       
             XYZ_coord_heardAPs=XYZ_APest(AP_heardindices,:);
             RSSheard_lin1=10.^(WLAN_test_per_measpoint{uu,2}(2,:)/10);
             Pos_est(uu,:)=Fct_weighted_centroid( XYZ_coord_heardAPs, RSSheard_lin1);
      end;   
     clear AP_heard* XYZ_coord* RSSheard*
 end;
 

end

