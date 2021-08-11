function [ out ] = fnc_coverage_distrib_based( train, est, par )
%fnc_coverage_area_based: pointwise deined coverage area algorithm (training, estimation, and evaluation) for given data 
% inputs
%   train:  training data
%   est:    test data
%   par: struct of parameters 
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
out.estimate_3D = train_and_est( train, est, par );

 %% evaluate errordistributions.
[out.mean_error_3D, out.mean_error_2D, out.floor_det_prob] = ...
    fnc_eval_error( out.estimate_3D, vertcat( est{:,1} ) );


end % fnc_coverage_area_based


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% calc_coverage_areas 
function CA = calc_coverage_areas( data_cells, par ) 

% find max AP index
prop =  fnc_get_data_properties( data_cells, {'APidxmax'} );
% number of samples 
N = size(data_cells,1);

% init struct 
CA( 1:prop.APidxmax ) = struct( 'm', zeros(1,3), 'sigma', ones(3,3), 'invSigma', ones(3,3) );

% collect the needed data to matrices
% * coord: training sample coordinates
coord = vertcat( data_cells{:,1} );
% get APs heard  matrix
APheard = 0*ones(N,prop.APidxmax);
% training sample loop
for ii = 1:N
    idxAPsHeard = data_cells{ii,2}(1,:);
    APheard(ii,idxAPsHeard) = 1;
end % for ii

% compute coverage area parameters (m and invSigma)
% AP loop
for iap = 1:prop.APidxmax
    % AP is present in fingerprints where APheard(:,iap)==1
    APheard_iap = (APheard(:,iap)==1); % logical values
    [CA(iap).m, CA(iap).sigma, CA(iap).invSigma] ...
        = calc_one_coverage_area( coord( APheard_iap, : ), par );
end % for iap

end % calc_coverage_areas

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% calc_one_coverage_area

function [m, S, Q] = calc_one_coverage_area( x, par )

[n,d] = size(x);

if n<2 % number of sample too small
    if n==0
        m = par.m0';
    else
        m = x;
    end
    S = par.P0;
    Q = S\eye(d);
    
    return % calc_one_coverage_area
end % if number of sample too small

switch par.distribution
    case 'gaussian'
        maxiter = 1;
    case 'student'
        maxiter = 10;
end % switch

dwtol = 0.003;
iter = 1; % initialize
w = ones(n,1); % initialize
dw = sum(w); % initialize

while iter<=maxiter && dw>dwtol
    m = weighted_mean( x, w );
    S = weighted_cov( x, w, m );
    if rcond(S)<1e-5 % matrix inverse will fail
        S = par.P0;
    end
    switch par.distribution
        case 'gaussian'
            qcoef = 1;
        case 'student'
            qcoef = max(1,(n-d-1)); % note: uses smaller nu if (n-d-1) goes to 0
    end % switch
    Q = qcoef*S\eye(d); 
    % EM-step for weights
    w_old = w;
    for ii = 1:n
        xdif = x(ii,:) - m;
        wdif = xdif * Q * xdif';
        w(ii) = (d+par.nu)/(par.nu + wdif);
    end % for ii
    % loop control
    iter = iter+1;
    dw = sum(abs(w-w_old))/n;
end % while


end % calc_one_coverage_area

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% estimate_location
function m = estimate_location( CA, idx_APs_heard, par )

NAP = length(idx_APs_heard);

% Weighted sum
sumInvSigma = zeros(3,3);
m = zeros(3,1);
for iap = 1:NAP
    sumInvSigma = sumInvSigma + CA(idx_APs_heard(iap)).invSigma;
    m = m + CA(idx_APs_heard(iap)).invSigma * CA(idx_APs_heard(iap)).m';
end % for iap
m = sumInvSigma\m;

m = m'; % output row vector


end % estimate_location

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% train_and_est
function est_out = train_and_est( train, est, par )

% * Prepare data 
% prior distribution
par.m0 = mean( vertcat(train{:,1}) )';
par.P0 = cov( vertcat(train{:,1}) ) ;
if isfield(par,'distribution') && strcmp(par.distribution,'student')
    par.P0 = 100^2*eye(3);
end
% compute coverage areas (training)
CA = calc_coverage_areas( train, par ); 
% Floor heights of training data
floor_heights = fnc_get_floor_heights( train );

 % * estimate
Ne = length(est);
est_out = nan(Ne,3);
% * process measurements loop
for ii = 1:Ne
    idx_APs_heard = est{ii,2}(1,:); 
    est_out(ii,:) = estimate_location( CA, idx_APs_heard, par );
    % force vertical coordinate to one of the floors
    est_out(ii,3) = floor_heights( fnc_get_knearest( est_out(ii,3), floor_heights, 1 ) );
end % for ii
 

end % train_and_est

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% weighted_cov

function S = weighted_cov( x, w, m )

[nr,nc] = size(x);
S = zeros(nc,nc);
for ii = 1:nr
    xdif = x(ii,:) - m;
    S = S + w(ii) * xdif' * xdif;
end % for ii

end % weighted_mean

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% weighted_mean

function wm = weighted_mean( x, w )

nc = size(x,2);
m1 = mean( (x.*repmat(w,1,nc)), 1 );
m2 = mean( w, 1 );
wm = m1/m2;

end % weighted_mean
