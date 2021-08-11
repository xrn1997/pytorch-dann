function [ out ] = fnc_rss_stats( data_cells )
%fnc_rss_stats: scanns through the data cells and determines the CDF, min, max,
%   median, mean, ans std. Finds also distribution of AP numbers.

% init out
out = struct( 'CDF',[], 'min', nan, 'max', nan, 'median', nan, ...
    'mean', nan, 'std', nan, ...
    'APCDF', [], 'APidxmin', nan, 'APidxmax', nan );

N = size(data_cells,1);

rss_vect = nan( N*1000, 1 );
ap_vect = nan( N*1000, 1 );
inext = 1;
for ii = 1:N
    aptmp = data_cells{ii,2}(1,:);
    rsstmp = data_cells{ii,2}(2,:);
    ndata = length( rsstmp );
    ap_vect( inext:inext+ndata-1 ) = aptmp;
    rss_vect( inext:inext+ndata-1 ) = rsstmp;
    inext = inext + ndata;
end % for ii
% clear the unused vector elements
rss_vect = rss_vect(1:inext-1);
ap_vect = ap_vect(1:inext-1);

if ~isempty(rss_vect)
    out.CDF(:,1) = sort(rss_vect);
    out.CDF(:,2) = (1:length(rss_vect))'/length(rss_vect);
    
    out.min = out.CDF(1,1);
    out.max = out.CDF(end,1);
    out.median = median(rss_vect);
    out.mean = mean(rss_vect);
    out.std = std(rss_vect);
    
    out.APCDF(:,1) = sort(ap_vect);
    out.APCDF(:,2) = (1:length(ap_vect))'/length(ap_vect);
    
    out.APidxmin = out.APCDF(1,1);
    out.APidxmax = out.APCDF(end,1);

end % if

end % fnc_rss_stats

