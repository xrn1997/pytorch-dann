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

function [AP_grid_set]=Fct_convertdatapergrid_toAPwise_inpNAP(Data_per_gridpoint,N_AP)
%CALL: [AP_grid_set]=Fct_convertdatapergrid_toAPwise_inpNAP(Data_per_gridpoint,N_AP);
%purpose: convert the fingerprint data Data_per_gridpoint, which is given per grid point, into data per
%Access Point
%INPUTS
%Data_per_gridpoint  = a Nmeas x 2 cell with Nmeas= number of measurements;
%                      the elements Data_per_gridpoint{i,1}, i=1:Nmeas show
%                      the (x,y,z) coordinates (in meters) where a measurement is done
%                      and the  elements Data_per_gridpoint{i,2}, i=1:Nmeas
%                      show the access points and the RSS values heard in
%                      that measurement point. For example, if in the point 
%                      (1, 2, 0) wer hear the access points 7 and 10 at -70 dB 
%                      and -65 dB respectively, then
%                      Data_per_gridpoint{i,1}=[1 2 0]; and
%                      Data_per_gridpoint{i,2}=[ 7   10;
%                                               -70  -65];
%N_AP                =total number of access points heard in the fingerprinting dataset 
%                     Data_per_gridpoint
%OUTPUTS
%AP_grid_set         = a N_AP x1 cell which contains the same information
%                    as the Data_per_gridpoint but written in a different
%                    form; each element AP_grid_set{i} contains a M_i x 4
%                    matrix, with M_i=number of points where the i-th AP is
%                    heard. For example if AP number 7 is heard in the
%                    point ( 1 2 0) (in meters) at level -70 and in the
%                    point (2 2 5) at level -90, then 
%                    AP_grid_set{7}=[1 2 0 -70;
%                                    2 2 5  -90];
%
%created by Simona Lohan and Jukka Talvitie
AP_grid_set = cell(1,N_AP);

for i = 1:size(Data_per_gridpoint,1)
    if isempty(Data_per_gridpoint{i,2})
        continue;
    else
    Temp = [];
    AP_id = Data_per_gridpoint{i,2}(1,:);
    for j=1:length(AP_id)
        Temp = [Data_per_gridpoint{i,1}(1),Data_per_gridpoint{i,1}(2),Data_per_gridpoint{i,1}(3),Data_per_gridpoint{i,2}(2*j)];
         % modified here
      
        if AP_id(j)~=0
            [m,n]= size(AP_grid_set{AP_id(j)});    
            AP_grid_set{AP_id(j)}(m+1,:) = Temp;
        end
    end
    clear Temp m n;
    end
end



