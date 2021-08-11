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

function Fct_plotstats_perAP(WLAN_perAP_training, WLAN_perAP_test)
%   Plot the fingerprints per access point

    numAp_t = size(WLAN_perAP_training,2);
    numFpPerAp_t = zeros(1,numAp_t);
    for iAp = 1:numAp_t
        numFpPerAp_t(iAp) = length(WLAN_perAP_training{iAp});
    end
    numAp_e = size(WLAN_perAP_test,2);
    numFpPerAp_e = zeros(1,numAp_e);
    for iAp = 1:numAp_e
        numFpPerAp_e(iAp) = length(WLAN_perAP_test{iAp});
    end
    figure;
    bar([1:numAp_t],numFpPerAp_t, 0.3, 'k'); hold on; grid on;
    bar([1:numAp_e]+0.3,numFpPerAp_e, 0.3, 'r');
    title('Fingerprints per Access Point');
    xlabel('access point ID');
    ylabel('number of fingerprints');
    legend('Training', 'Test')
end
