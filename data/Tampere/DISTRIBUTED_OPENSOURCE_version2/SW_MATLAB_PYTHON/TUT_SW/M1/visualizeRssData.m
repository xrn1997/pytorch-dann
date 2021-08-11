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

function visualizeRssData(path_to_data)
%   Matlab script to visualize WLAN RSS fingerprint data
%
% Input:
%   path_to_data -  path to fingerprint data

close all; clc;


if nargin < 1, path_to_data = '../../../FINGERPRINTING_DB'; end

% load data
rss_dB = importdata([path_to_data, '/Test_rss_21Aug17.csv']);
% or Training_rss_21Aug17.csv
rp_m = importdata([path_to_data, '/Test_coordinates_21Aug17.csv']);
% or Training_coordinates_21Aug17.csv

numFp = size(rss_dB, 1);
numAp = size(rss_dB, 2);
fprintf('Number of fingerprints: %d\n', numFp);
fprintf('Number of access points: %d\n', numAp);


% set invalid RSS to NaN;
rss_dB(rss_dB==100) = NaN;

% plot some statistics
plotStats(rss_dB);

% plot positions of reference points
plotRpGrid(rp_m);
% plot fingerprints of a selected AP; in this example is 492
selAp =  492;
plotFpPerAp(rss_dB, rp_m, selAp);

% plot RSS of single access point for one floor
hgt = unique(rp_m(:,3));
% select the floor number for visualization; in this example it is 2
floor = 2; % choose between 1:5
idxRpFlX = find(rp_m(:,3) == hgt(floor));
% num. of valid FP must be high
% numel(find(~isnan(rss_dB(idxRpFlX, selAp))))
plotRss(rp_m(idxRpFlX, 1:2), rss_dB(idxRpFlX, :), selAp, 'contour', 4);

end % end primary function

function plotStats(rss_dB)
%   Plot some statistics
    numAp = size(rss_dB,2);
    numFpPerAp = zeros(1,numAp);
    for iAp = 1:numAp
        numFpPerAp(:,iAp) = numel(find(~isnan(rss_dB(:,iAp))));
    end
    figure(1)
    bar(numFpPerAp)
    title('Fingerprints per Access Point');
    xlabel('access point ID');
    ylabel('number of fingerprints');
end

function plotRpGrid(rp_m)
%   Plot positions of fingerprints for all floors
    figure(2)
    plot3(rp_m(:,1), rp_m(:,2), rp_m(:,3), '.')
    grid on
    title('Positions of fingerprints');
    xlabel('easting (m)')
    ylabel('northing (m)')
    zlabel('height (m)')
end

function plotFpPerAp(rss_dB, rp_m, ap)
%   Plot 3D scatter plot of RSS for single access point
    figure(3)
    scatter3(rp_m(:,1), rp_m(:,2), rp_m(:,3), 10, rss_dB(:,ap),'filled')
    grid on
    colormap(jet);
    colorbar;
    title(['Fingerprints of access point ', num2str(ap)])
    xlabel('easting (m)')
    ylabel('northing (m)')
    zlabel('height (m)')
end

function h = plotRss(gridPos2D_m, rss_dB, m, varargin)
%   Plot surface, contour, etc. of RSS values for an access point.
% Plot signal strength values of chosen AP over given metric cartesian
% coordinate system.
%
% Input:
%       gridPos2D_m -   N x 2 matrix of positions of coordinate system (grid)
%       rss_dB      -   N x numAp columns of RSS values, according to gridPos2D_m
%       m           -   mth column of rss_dB determines mth access point
%       type        -   string for type of plot ('surf', 'surfc', 'contour',
%                       contour3, contourf, plot3)
%       fig_num     -   figure handle number (to not overwrite other figures
%                       and to reuse them)
%       fig_option  -   cell of plot property value pairs
%
% Output:
%       h           -   figure handle


    % check number of input parameters
    narginchk(3, 6);

    switch nargin
        case 2
            m = 1;
            type = 'plot3';
            fig_number = 500;
            fig_option = {'ko'};
        case 3 % default case
            type = 'plot3';
            fig_number = 500;
            fig_option = {'ko'};
        case 4
            type =  varargin{1};
            fig_number = 500;
            fig_option = {'ko'};
        case 5
            type =  varargin{1};
            fig_number = varargin{2};
            fig_option = {'ko'};
        case 6
            type =  varargin{1};
            fig_number = varargin{2};
            fig_option = varargin{3};
    end
    if exist(type) == 0
        error('Unknown plot type');
    end

    nRowsGrid = size(gridPos2D_m, 1);
    szRss = size(rss_dB);

    if all(szRss ~= nRowsGrid)
        error('gridPos2D_m must be same length as rss_dB.');
    end

    if szRss(1) ~= nRowsGrid
        rss_dB = rss_dB .';
    end


    [X, Y] = meshgrid(gridPos2D_m(:,1), gridPos2D_m(:,2));
    h = figure(fig_number); % high default number to no get in conflict with others
    hold on;
    switch type
        case 'plot3'
            plot3(gridPos2D_m(:,1),gridPos2D_m(:,2),rss_dB(:,m), fig_option{:});
            zlabel('RSS (dB)');
            view(3);
        case 'stem3'
            stem3(gridPos2D_m(:,1),gridPos2D_m(:,2),rss_dB(:,m), fig_option{:});
            zlabel('RSS (dB)');
            view(3);
        case 'surf'
            warning off
            ZI = griddata(gridPos2D_m(:,1), gridPos2D_m(:,2), rss_dB(:,m), X, Y);
            if ~isempty(ZI)
                surf(X, Y, ZI);
                zlabel('RSS (dB)');
                view(3);
            end
        case 'surfc'
            warning off
            ZI = griddata(gridPos2D_m(:,1), gridPos2D_m(:,2), rss_dB(:,m), X, Y);
            if ~isempty(ZI)
                surfc(X, Y, ZI);
                zlabel('RSS (dB)');
                colorbar;
                view(3);
                colormap(jet);
            end
        case 'contour'
            warning off
            ZI = griddata(gridPos2D_m(:,1), gridPos2D_m(:,2), rss_dB(:,m), X, Y);
            if ~isempty(ZI)
                contour(X, Y, ZI);
                colorbar;
                view(2);
                colormap(jet);
            end
        case 'contour3'
            ZI = griddata(gridPos2D_m(:,1), gridPos2D_m(:,2), rss_dB(:,m), X, Y);
            if ~isempty(ZI)
                contour3(X, Y, ZI);
                hold on
%                 surface (X, Y, ZI, 'facecolor', 'none', 'EdgeColor', 'k');
                zlabel('RSS (dB)');
                colorbar;
                view(3);
                colormap(jet);
            end
        case 'contourf'
            warning off
            ZI = griddata(gridPos2D_m(:,1), gridPos2D_m(:,2), rss_dB(:,m), X, Y);
            if ~isempty(ZI)
                contourf(X, Y, ZI);
                colorbar;
                view(2);
                colormap(jet);
            end
    end
    warning on

    if exist('ZI', 'var') && isempty(ZI)
      warning('Not enough data to plot');
    end

    hold off;
    title(['RSS of access point ', num2str(m), ' on the selected floor']);
    xlabel('easting (m)')
    ylabel('northing (m)')
    grid on;
end
