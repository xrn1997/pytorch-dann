% Copyright © 2017 J. Torres-Sospedra,  Universitat Jaume I (UJI)
%
% Permission is hereby granted, free of charge, to any person obtaining a copy of
% this software and associated documentation files (the “Software”), to deal in
% the Software without restriction, including without limitation the rights to
% use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
% of the Software, and to permit persons to whom the Software is furnished to do
% so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

function [ results ] = knn_ips( database_orig , k, datarep, defaultNonDetectedValue, newNonDetectedValue , distanceMetric )
% knn_ips Indoor Positioning System based on kNN
%   
% inputs: 
%    database_orig           : database with RSSI values and Labels
%    k                       : value of nearest neighbors in kNN algoritm
%    datarep                 : Data representation
%    defaultNonDetectedValue : Value in DB for non detected RSSI values
%    newNonDetectedValue     : New Value for non detected RSSI values
%    distanceMetric          : Distance metric used for FP matching
%
% output:
%    results                 : output statistics
%
% Developed by J. Torres-Sospedra,
% Instiute of New Imaging Technologies, Universitat Jaume I
% jtorres@uji.es



% Remap floors numbers to 1:nfloors
origFloors = unique((database_orig.trainingLabels(:,4))); 
nfloors    = size(unique(round((origFloors-1)*2)+2)/2,1);
database0  = remapFloorDB(database_orig,origFloors',1:nfloors);

% Calculate the overall min RSSI (minus 1)
if size(newNonDetectedValue,1) == 0
newNonDetectedValue = min(...
                     [database0.trainingMacs(:)',...                      
                      database0.testMacs(:)']...
                     )-1;
end

% Replace NonDetectedValue
if size(defaultNonDetectedValue,1) ~= 0
    database0 = datarepNewNullDB(database0,defaultNonDetectedValue,newNonDetectedValue);
end
% Apply new data representation                 
if  strcmp(datarep,'positive') % Convert negative values to positive by adding a value
    database  = datarepPositive(database0);
elseif strcmp(datarep,'exponential') % Convert to exponential data representation
    database0 = datarepPositive(database0);
    database  = datarepExponential(database0);
elseif strcmp(datarep,'powed') % Convert to powed data representation
    database0 = datarepPositive(database0); 
    database  = datarepPowed(database0);    
end % Default, no conversion

clear database0;
   
% Get number of samples
rsamples = size(database.trainingMacs,1); 
osamples = size(database.testMacs,1);

% Prepare output vectors
results.error      = zeros(osamples,3);
results.prediction = zeros(osamples,4);
results.targets    = zeros(osamples,4);
results.candidates = zeros(osamples,1);
results.distances  = zeros(osamples,1);

% Get the deviations values for Mahalanobis Distance
if strcmp(distanceMetric,'distance_mahalanobis.m')
    desviacion   = std(trainingMacs);
    desviacionsq = (desviacion+(100000*desviacion==0)).^2;
end

fprintf('Running the knn_ips algorithm with\n')
fprintf('    k                      : %d\n',k)
fprintf('    datarep                : %s\n',datarep)
fprintf('    defaultNonDetectedValue: %d\n',defaultNonDetectedValue)
fprintf('    newNonDetectedValue    : %d\n',newNonDetectedValue)
fprintf('    distanceMetric         : %s\n',distanceMetric )

for i = 1:osamples    

    ofp = database.testMacs(i,:);
    
    % Get distance between the operational fingerprint 
    % and any reference fingerprint
    if     strcmp(distanceMetric,'euclidean');  distances = sqrt(sum((ones(rsamples,1)*ofp - database.trainingMacs).^2,2)); % Optimized Euclidean distance
    elseif strcmp(distanceMetric,'manhattan');  distances =  sum(abs(ones(rsamples,1)*ofp - database.trainingMacs),2);      % Optimized Manhattan distance
    elseif strfind(distanceMetric,'distance_')                                                                     % Custom distance function distance_xxxxx.m                      
       if strcmp(distanceMetric,'distance_mahalanobis')
           distances = zeros(1,rsamples);
           for j = 1:rsamples;
            distances(j) = feval(distanceMetric,database.trainingMacs(j,:),ofp,desviacionsq);
           end                   
       else
           distances = zeros(1,rsamples);
           for j = 1:rsamples;
            distances(j) = feval(distanceMetric,database.trainingMacs(j,:),ofp);
           end
       end        
    end
    
    % Sort distances
    [distancessort,candidates] = sort(distances);
    
    % Set the value of candidates to k (kNN)
    ncandidates = k;
    
    % Check if candidates ranked in positions higher than k have the same
    % distance in the feature space.
    while ncandidates < rsamples
       if abs(distancessort(ncandidates)-distancessort(ncandidates+1))<0.000000000001
           ncandidates = ncandidates+1;
       else
           break
       end
    end
    
    % Estimate the building from the ncandidates
    probFloor = zeros(1,nfloors);
    for floor = 1:nfloors
        probFloor(floor) = sum(database.trainingLabels(candidates(1:ncandidates),4)==floor);
    end
    [~,floor] = max(probFloor);
    
    % Estimate the coordinates from the ncandidates that belong to the
    % estimated floor
    points = 0;
    point  = [0,0,0];
    for j = 1:ncandidates
        if (database.trainingLabels(candidates(j),4)==floor)
            points = points + 1;
            point  = point + database.trainingLabels(candidates(j),1:3);
        end
    end
    point = point / points;
    
    % Extract the real-world floor (revert floor remap)
    realWorldFloor = remapFloor(floor,1:nfloors,origFloors');
    
    % Generate the results statistics
    results.error(i,1) = distance_euclidean(database.testLabels(i,1:2),point(1:2)); % 2D Positioning error in m 
    results.error(i,2) = distance_euclidean(database.testLabels(i,1:3),point(1:3)); % 3D Positioning error in m
    results.error(i,3) = abs(database.testLabels(i,3)-point(3));                    % Height difference in m
    results.error(i,4) = abs(realWorldFloor - database_orig.testLabels(i,4));       % Height difference in floors
    
    results.prediction(i,:) = [point,realWorldFloor];                               % Predicted position [x,y,z,floor]
    results.targets(i,:)    = database_orig.testLabels(i,1:4);                      % Current position   [x,y,z,floor]
    results.candidates(i,1) = ncandidates;                                          % Number of nearest neighbours used to estimate the position
    results.distances(i,1)  = distancessort(1);                                     % Distance in feature space of the best match
    
end


fprintf('\nSuccesfully executed!\n')
fprintf('    Mean Positioning Error 2D* : %3.3f\n',mean(results.error((results.error(:,4)==0),1)));  % Mean 2D Positioning error when estimate and true position are in the same floor
fprintf('    Mean Positioning Error 3D  : %3.3f\n',mean(results.error(:,2)));                        % Mean 3D Positioning error
fprintf('    Floor detection hit rate   : %3.3f\n',mean(results.error(:,4)==0)*100);                 % Floor detection rate
