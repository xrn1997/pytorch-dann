%% fnc_neval_error
% Evaluates error in a set of 3D position estimates.
% outputs
%   err3D: 3D mean error is the 3D estimation error between the estimated
%       3D position and the true 3D position.
%   err2D: 2D mean error is the error between the estimated position and
%       the true position when the estimate is at the same floor with the
%       true position (incorrect floor estimates are ignored here). 
%   fdp: Floor detection probability is the percentage of the test points
%       in which the floor was estimated correctly 
%
function [err3D, err2D, fdp] = fnc_eval_error( coord_est, coord_true )

coord_diff = coord_est - coord_true;
err3D = mean( sqrt( sum( coord_diff.^2, 2 ) ) ); % mean 3D error
correct_floor = (coord_diff(:,3)==0); % status bits
fdp = sum( correct_floor ) / length( correct_floor ); % floor detection probability
err2D = mean( sqrt( sum( coord_diff(correct_floor,1:2).^2, 2 ) ) ); % mean 2D error

end % fnc_eval_error
