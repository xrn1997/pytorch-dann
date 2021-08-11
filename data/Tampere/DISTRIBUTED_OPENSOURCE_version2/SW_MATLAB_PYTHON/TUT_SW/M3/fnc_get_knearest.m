%% fnc_get_knearest
% Finds the k rows in an matrix data that are closest to the row vector
% ref, or k scalar values in column vectros that a closest to the scalar
% ref.
% The used distance measure is Euclidean distance.
%

function iknearest = fnc_get_knearest( ref, data, k )
% note: ref is scalar or row vector
% iknearest: indices to rows in data that have shortest distances to ref

N = size(data,1);
datadiff = data - repmat( ref, N, 1 );
% sort by squared distances
[~,isort] = sort( sum(datadiff.^2,2) );
iknearest = isort( 1:k );

end % fnc_get_knearest