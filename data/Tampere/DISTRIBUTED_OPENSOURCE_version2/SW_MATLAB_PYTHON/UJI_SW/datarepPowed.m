% Copyright © 2017 J. Torres-Sospedra,  Universitat Jaume I (UJI)
%
% Permission is hereby granted, free of charge, to any person obtaining a copy of
% this software and associated documentation files (the “Software”), to deal in
% the Software without restriction, including without limitation the rights to
% use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
% of the Software, and to permit persons to whom the Software is furnished to do
% so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

function [ db1 ] = datarepPowed( db0 )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    minValue           = min([db0.trainingMacs(:)',db0.testMacs(:)']);
    maxValue           = max([db0.trainingMacs(:)',db0.testMacs(:)']);
    
    normValue          = (maxValue.^exp(1));
    
    db1.trainingMacs   = (db0.trainingMacs.^exp(1))./normValue;
    db1.testMacs       = (db0.testMacs.^exp(1))./normValue;

    db1.trainingLabels   = db0.trainingLabels;
    db1.testLabels       = db0.testLabels;
    
end

