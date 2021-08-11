function [ out ] = fnc_coverage_pointwise( train, est, pAPmatch )
%coverage_pointwise: pointwise deined coverage area algorithm (training, estimation, and evaluation) for given data 
% inputs
%   train:  training data
%   est:    test data
%   pAPmatch: probability of matching AP 
%
% output
%   out: struct with fields 
%       - estimate_3D
%       - mean_error_3D
%       - mean_error_2D
%       - floor_det_prob
%

%% init output
out = struct( 'estimate_3D', [], 'mean_error_3D', [], 'mean_error_2D', [], ...
    'floor_det_prob', [] );

%% train & estimate
out.estimate_3D = train_and_est( train, est, pAPmatch );

 %% evaluate error
[out.mean_error_3D, out.mean_error_2D, out.floor_det_prob] = ...
    fnc_eval_error( out.estimate_3D, vertcat( est{:,1} ) );


end % coverage_pointwise


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% create_prob_map 
function prob_map = create_prob_map( data_cells, pAPmatch ) 

% find max AP index
prop =  fnc_get_data_properties( data_cells, {'APidxmax'} );
% number of samples 
N = size(data_cells,1);

% init map matrices 
% * APprob (APs heard): rows for training samples, columns for APs
prob_map.APheard = 0*ones(N,prop.APidxmax);
% * coord: training sample coordinates
prob_map.coord = vertcat( data_cells{:,1} );
% * add AP match probability
prob_map.pAPmatch = pAPmatch;

% training sample loop
for ii = 1:N
    idxAPsHeard = data_cells{ii,2}(1,:);
    prob_map.APheard(ii,idxAPsHeard) = 1;
end % for ii

end % create_prob_map

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% estimate_location
function coords = estimate_location( prob_map, idx_APs_heard )

N = size(prob_map.coord,1);
NAP = size(prob_map.APheard,2);

% Find AP matches
APs_heard = zeros(1,NAP); % initialize new observation in vector
APs_heard( idx_APs_heard ) = 1; % add values
APs_match = ( repmat(APs_heard, N, 1) ==  prob_map.APheard );

% % likelihoods of locations (find likelihoods of locations based on the
% match with the new observation and radio map (prob_map)) 
p_AP = ones(size(APs_match)) * (1-prob_map.pAPmatch);
p_AP( APs_match==1 ) = prob_map.pAPmatch;
p = prod( p_AP, 2 );

% estimate as a weighted sum of training sample locations, weighted by
% location likelihoods
p = p/sum(p); % normalize sum to 1
coords = sum( repmat(p,1,3).*(prob_map.coord), 1);

end % estimate_location

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% train_and_est
function est_out = train_and_est( train, est, pAPmatch )

% * Prepare data 
% create probability maps (training)
prob_map = create_prob_map( train, pAPmatch );
% Floor heights of training data
floor_heights = fnc_get_floor_heights( train );
 
 % * estimate
Ne = length(est);
est_out = nan(Ne,3);
% * process measurements loop
disp('Running estimate and avaluate loop ...')
for ii = 1:Ne
    idx_APs_heard = est{ii,2}(1,:); 
    est_out(ii,:) = estimate_location( prob_map, idx_APs_heard );
    % force vertical coordinate to one of the floors
    est_out(ii,3) = floor_heights( fnc_get_knearest( est_out(ii,3), floor_heights, 1 ) );
    if rem(ii,200)==0
        disp(sprintf('sample #%d - %s',ii,datestr(now,'HH:MM:SS.FFF')))
    end % if
end % for ii
 

end % train_and_est



