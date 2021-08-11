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

function Y_centroid = Fct_weighted_centroid(X_matrix, weights)
% If X_matrix is a matrix of N x Ndim coordinates (eg Ndim=3 for 3D) and
% each coordonate pair (each row of X_matrix) has a weight associated to it
% weights(i), i=1..N, this function computes the weighted centroid.
% Y_centroid will have dimension Ndim
N = size(X_matrix,1);
Ndim = size(X_matrix,2);
if length(weights) ~= N,
    error('Incorrect input dimensions')
end;
if size(weights,1) < size(weights,2),
    weights = weights'; % Transform it in a column vector
end;

for nn = 1:Ndim,
    idx = find(~isnan(X_matrix(:,nn)));
    Y_centroid(nn) = sum((X_matrix(idx,nn).*weights(idx))/sum(weights(idx)));
end

end

