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

function [Pd, dist_err3D, dist_err2D]=Fct_2Dand3Dstatistics_on_posest(coordinate_e,pos_estFP, floor_heights)
%coordinate_e  = matrix of 'true' coordinates
%pos_estFP    = matrix of estimated coordinates
%floor_heights =vector of all floor heights in the building


%statistics
dist_err3D=sqrt((coordinate_e(:,1)-pos_estFP(:,1)).^2+(coordinate_e(:,2)-pos_estFP(:,2)).^2+(coordinate_e(:,3)-pos_estFP(:,3)).^2);
Pd=0;dist_err2D=NaN*ones(1,size(coordinate_e,1));
floor_est=zeros(size(coordinate_e,1),1);
floor_true=zeros(size(coordinate_e,1),1);
for ii=1:size(coordinate_e,1)
    [~,floor_est(ii)]=min(abs((pos_estFP(ii,3)-floor_heights)));
    [~,floor_true(ii)]=min(abs((coordinate_e(ii,3)-floor_heights)));
    if floor_true(ii)==floor_est(ii)
        Pd=Pd+1;
        %compute also 2D distance error
        dist_err2D(ii)=sqrt((coordinate_e(ii,1)-pos_estFP(ii,1)).^2+(coordinate_e(ii,2)-pos_estFP(ii,2)).^2);
    end;
end;
Pd=Pd/size(coordinate_e,1)*100;
