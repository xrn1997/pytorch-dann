% Copyright © 2016 Tampere University of Technology (TUT)
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

function [qx,qy,qz, Length_WiFigrid,plot_index]=Fct_plot_Powermaps3Ddata_perfloor(index_AP,FingerprintData_perAP, desired_floor_height)
%CALL: [qx,qy,qz, Length_WiFigrid,plot_index]=Fct_plot_Powermaps3Ddata_perfloor(index_AP,FingerprintData_perAP, desired_floor_height)
%INPUTS:
%index_AP= the Access Point index for which we want to plot the power map;
%          it should be scalar between 1 and N_AP with
%          N_AP=length(FingerprintData_perAP) = number of AP in the
%          database
%FingerprintData_perAP  =  a cell ofsize 1x N_AP; each cell element i (i=1:N_AP) contains 
%                          4 values:  (x,y,z, RSS_dB) 
%                          with (x,y,z)=the coordinates of the measurements
%                          and RSS_dB= the measured  RSS value in decibel
%                          from the i-th AP to the (x,y,z) point
%desired_floor_height = height of the floor at which we plot the data; if
%                       the value is not found in the FingerprintData_perAP
%                       nothing will be plotted
%OUTPUTS (optionals)
%[qx,qy,qz]           = the 3D mesh which is plotted as the power map  
%Length_WiFigrid      = the number of points in which the considered AP is
%                       heard
%plot_index           = a 0 or 1 flag telling if a power map is plotted or
%                     not; in our case a power map is plotted only if the
%                     considered AP is heard in more than Minimum_number_points_for_plot 
%                     points at the
%                     considered floor
%created by Simona Lohan and Jukka Talvitie

%% initialize variables
qx=[]; qy=[]; qz=[];
Minimum_number_points_for_plot=30;

plot_index=0;
%select the desired AP grid        
AP_grid_init =FingerprintData_perAP{index_AP};  %(x,y,z, RSS
%find AP_grid only per desired floor
kk=1; AP_grid=[];
 for ii=1:size(AP_grid_init,1)
        if abs(AP_grid_init(ii,3)-desired_floor_height)<1e-4
            AP_grid(kk,:)=AP_grid_init(ii,[1 2 4]);
            kk=kk+1;
        end;
 end;
 Length_WiFigrid=length(AP_grid);

%plot only if the AP is heard in at least Minimum_number_points_for_plot points at the considered
%floor
if length(AP_grid) >= Minimum_number_points_for_plot
    x_map = [min(AP_grid(:,1)); max(AP_grid(:,1))];
    y_map = [min(AP_grid(:,2)); max(AP_grid(:,2))];

    grid_interval=1;

    grid_vec_x = floor((x_map(1)/grid_interval))*grid_interval:grid_interval:floor((x_map(2)/grid_interval))*grid_interval;
    grid_vec_y = floor((y_map(1)/grid_interval))*grid_interval:grid_interval:floor((y_map(2)/grid_interval))*grid_interval;

    [qx,qy] = meshgrid(grid_vec_x,grid_vec_y);
    qz_orig = griddata(AP_grid(:,1),AP_grid(:,2),AP_grid(:,3),qx,qy);
    clear qz
    qz = qz_orig;
    figure
    [C_label,h_label] = contourf(qx,qy,qz,-100:1:-30,'LineWidth',0.5); 
    hold on;
    caxis([-100 -30]) 
    colorbar
    clabel(C_label,h_label,'FontSize',10,'Color','k','Rotation',0) %lukuarvot kuvaan
    plot(AP_grid(:,1),AP_grid(:,2),'r*') 
    title(['AP#: ' num2str(index_AP)])    
    legend('powers','measurement points');  
    plot_index=1;

end





