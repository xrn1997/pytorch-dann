%% fnc_get_floor_heights
% finds the floor height values present in the collected data
%
function floor_heights = fnc_get_floor_heights( data_cells )

coordinates = vertcat( data_cells{:,1} );
floor_heights = unique( coordinates(:,3) );

end % fnc_get_floor_heights