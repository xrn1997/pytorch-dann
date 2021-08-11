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

%Convert the csv data into Matlab format and perform basic analysis on the
%data, as published in the MDPI Data journal paper

clear all; close all; clc
warning off
path_to_csvfiles='../../../FINGERPRINTING_DB/'; %choose the path to where you saved the csv files (Test_rss_21Aug17.csv, etc.)
%convert the csv data into 2 Matlab cell structures: one for the training and
%one for the test data
[WLAN_training, WLAN_test, coordinate_t, coordinate_e, N_AP, floor_heights]=Fct_read_CSVdata_inMatlab(path_to_csvfiles);

%% DIMENSIONS OF THE MEASUREMENT SPACE IN THE BUILDING
%area of measurements in meters
building_size_x=max([coordinate_e(:,1); coordinate_t(:,1)])-min([coordinate_e(:,1); coordinate_t(:,1)]);
building_size_y=max([coordinate_e(:,2); coordinate_t(:,2)])-min([coordinate_e(:,2); coordinate_t(:,2)]);
disp(['The horizontal area of the measurement space was: ', num2str(building_size_x), ' x ',  num2str(building_size_y), ' m^2'])
%%

%% ##################NUMBER OF 'INSIGNIFICANT' AP###########################
%number of Access Points heard in less than 3 points
%first convert the current cell structures given per measurement point into
%the equivalent cell structures per Access Point
%convert data per AP format
WLAN_perAP_training=Fct_convertdatapergrid_toAPwise_inpNAP(WLAN_training,N_AP);
WLAN_perAP_test=Fct_convertdatapergrid_toAPwise_inpNAP(WLAN_test,N_AP);

%find out average number of points in which an AP is heard
Npoints=zeros(1, N_AP);
for ap=1:N_AP
    Npoints(ap)=size(WLAN_perAP_training{ap},1)+size(WLAN_perAP_test{ap},1);
end;
%find out the number of APs heard in less than 3 points
N_insignificantAP=length(find(Npoints<3));
fprintf('\n')
fprintf('Number of access points heard in less than 3 points: %d\n', N_insignificantAP)

%%


%% PLOT THE TEST AND TRAINING DATA COORDINATES in 3D and 2D spaces
%3D plot
figure,plot3(coordinate_t(:,1),coordinate_t(:,2),coordinate_t(:,3),'ro'); hold on;
plot3(coordinate_e(:,1),coordinate_e(:,2),coordinate_e(:,3),'b s');
title('University building')
grid on
view(3)
legend('Locations of training data', 'Locations of test data')
xlabel('x [m]'), ylabel('y [m]'), zlabel('z [m]'); drawnow;
%2D plot at first floor
des_floor=1;
idx_t = find(coordinate_t(:,3) == floor_heights(des_floor));
idx_e = find(coordinate_e(:,3) == floor_heights(des_floor));
figure,plot(coordinate_t(idx_t,1),coordinate_t(idx_t,2),'ro'); hold on;
plot(coordinate_e(idx_e,1),coordinate_e(idx_e,2),'b s');
legend('Locations of training data', 'Locations of test data')
xlabel('x [m]'), ylabel('y [m]'),grid on ; drawnow;
%%
%% Histogram of number of measurements per floor

[a1, b1]= hist(coordinate_t(:,3));
[a2, b2]= hist(coordinate_e(:,3));

figure; bar(b1,a1, 0.3, 'k'); hold on;
bar(b2+0.3,a2, 0.3, 'r');
ylabel('Number of measurements'); xlabel('Floor height [m]')
title('Number of measurements per floor');
grid on; legend('Training', 'Test');

%%
%% Histogram of number of measurements per device
models_training=[WLAN_training{:,4}];
models_test=[WLAN_test{:,4}];

all_phone_models_inDatabase=unique([models_training models_test]);
Ndevices=length(all_phone_models_inDatabase); %
data_model_t=cell(1,Ndevices);
data_model_e=cell(1,Ndevices);
for i=1:size(WLAN_training,1)
   ind=strcmp(all_phone_models_inDatabase,WLAN_training{i,4});
   data_model_t{ind}=[data_model_t{ind}, WLAN_training{i,4}];
end
for i=1:size(WLAN_test,1)
   ind_e=strcmp(all_phone_models_inDatabase,WLAN_test{i,4});
   data_model_e{ind_e}=[data_model_e{ind_e}, WLAN_test{i,4}];
end
%number of meas per phone model
data_num_t=zeros(1,length(all_phone_models_inDatabase)); %in the training set
data_num_e=zeros(1,length(all_phone_models_inDatabase)); %in the test set
for i=1:length(data_model_t)
   data_num_t(i)=length(data_model_t{i});
end
for i=1:length(data_model_e)
   data_num_e(i)=length(data_model_e{i});
end
%plot the number of measurements per device
figure; bar([1:Ndevices], data_num_t, 0.3, 'k', 'grouped') ;
hold on; bar([1:Ndevices]+0.3, data_num_e, 0.3, 'r', 'grouped'); grid on;
ylabel('Number of measurements');
title('Number of measurements per device');
grid on; legend('Training', 'Test');
p=get(gca,'position'); % get the axes position vector
p(2)=p(2)+.3; p(4)=p(4)-.3;  % raise bottom, shorten height for label room
set(gca,'position',p)
text(1:Ndevices,zeros(1,Ndevices)-100,all_phone_models_inDatabase,'rotation',90,'horizontalalignment','right')
grid on;

%find the number of devices which reported at least 10 measurements
fprintf('\n')
fprintf('Number of devices out of 21 which reported more than 10 measurements: %d\n', length(find(data_num_t+data_num_e>10)));

%%

%% PLOT THE NUMBER OF MEASUREMENTS PER ACCESS POINT
Fct_plotstats_perAP(WLAN_perAP_training,WLAN_perAP_test)
%%

%% PLOT 3D SCATTER PLOTS OF RSS PER SELECTED AP
%select a desired AP for which to plot the 3D scatter plot
ap_desired=492;
figure
scatter3(WLAN_perAP_training{ap_desired}(:,1), WLAN_perAP_training{ap_desired}(:,2), WLAN_perAP_training{ap_desired}(:,3), 10, WLAN_perAP_training{ap_desired}(:,4),'filled');
colormap(jet);
colorbar;
title(['Training:  access point ', num2str(ap_desired)])
xlabel('easting (m)')
ylabel('northing (m)')
zlabel('height (m)')
figure
scatter3(WLAN_perAP_test{ap_desired}(:,1), WLAN_perAP_test{ap_desired}(:,2), WLAN_perAP_test{ap_desired}(:,3), 10, WLAN_perAP_test{ap_desired}(:,4),'filled');
colormap(jet);
colorbar;
title(['Test:  access point ', num2str(ap_desired)])
xlabel('easting (m)')
ylabel('northing (m)')
zlabel('height (m)')
%%

%% %% PLOT 2D SCATTER PLOTS POWER MAPS OF RSS PER SELECTED AP
%select the desired floor for which the power map is visualized; here 2nd
%floor (height=3.7m)
desired_floor_height=3.7;
Fct_plot_Powermaps3Ddata_perfloor(ap_desired,WLAN_perAP_training, desired_floor_height);
colormap(jet); title(['Training:  access point ', num2str(ap_desired)])
colorbar; drawnow;

Fct_plot_Powermaps3Ddata_perfloor(ap_desired,WLAN_perAP_test, desired_floor_height);
colormap(jet); title(['Test:  access point ', num2str(ap_desired)])
colorbar; drawnow;
%%



%% POSITIONING ESTIMATION VIA 2 ALGORITHMS: WEIGHTED CENTROID AND LOG-GAUSSIAN FINGERPRINTING
%weighted centroid
[pos_estWeC] = Fct_weightedcentroid_posestim(WLAN_perAP_training, WLAN_test);

%LOG-GAUSSIAN FINGERPRINTING
%set the expected shadowing variance (here 10 dB) and the number of
%nearest neighbours used in the log-Gaussian probabilistic algorithm (here
%3)
sigma_shad_dB=10; Nneigh=3;
[pos_estFP]= Fct_LogGauss_FPestimate(N_AP, WLAN_training, WLAN_test,sigma_shad_dB, Nneigh);

%statistics
[Pd(1), dist_err3D(1,:), dist_err2D(1,:)]=Fct_2Dand3Dstatistics_on_posest(coordinate_e,pos_estWeC, floor_heights);
[Pd(2), dist_err3D(2,:), dist_err2D(2,:)]=Fct_2Dand3Dstatistics_on_posest(coordinate_e,pos_estFP, floor_heights);
a=~isnan(dist_err2D);
b=~isnan(dist_err3D);

fprintf('\n');
fprintf('Weighted centroid:\n');
fprintf('Mean positioning error 2D* : %3.3f m\n',mean(dist_err2D(1,a(1,:))));
fprintf('Mean positioning error 3D  : %3.3f m\n',mean(dist_err3D(1,b(1,:))));
fprintf('Floor detection rate: %3.3f %%\n',Pd(1));
fprintf('\n');
fprintf('Gaussian likelihood:\n');
fprintf('Mean positioning error 2D* : %3.3f m\n',mean(dist_err2D(2,a(2,:))));
fprintf('Mean positioning error 3D  : %3.3f m\n',mean(dist_err3D(2,b(2,:))));
fprintf('Floor detection rate: %3.3f %%\n',Pd(2));

%%







