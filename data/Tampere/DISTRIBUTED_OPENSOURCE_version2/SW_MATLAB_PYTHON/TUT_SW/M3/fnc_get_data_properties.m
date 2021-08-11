%% fnc_get_data_properties
% function picks defined outputs of rss_stats
function [out] = fnc_get_data_properties( data_cells, fnames )

stats = fnc_rss_stats( data_cells );
for ii = 1:length(fnames)
    out.(fnames{ii}) = stats.(fnames{ii});
end % for ii

end % fnc_get_data_properties