function [ labels, centers ] = ssKpp(data, k, knownLabels, trueLabels, knownNotRedIdx, iterMax)
%Initializes centers using ssKppInit (D^2 weighted with supervision).
%Iterates using Lloyds iteration where data(knownLabels,:) do not get
%reassigned and data(knownNotRed,:) do not get assigned to class 1 
%(red center).
% IMPORTANT GOTCHA:  RED IS ASSUMED TO BE CLASS 1, BUT THE USER INPUTS
% TRUELABELS, SO THEY MUST BE CONSISTENT.  DANGER DANGER DANGER.  OJO.
% CUIDADO.
%############### INPUTS ###############
%data: n x d data matrix
%k: the number of groups
%knownLabels: subset of 1:n indicating indices of data with labels
% trueLabels: vector of size n that holds the true labels.
%             trueLabels(knownLabels) must be between 1 and k
%knownNotRedIdx: subset of 1:n indicating indices of data with labels NOT
%equal to 1.
%iterMax: maximum number of Lloyds iterations to run (defaults to 20).
%############### OUTPUTS ##############
% labels: final labels
% centers: final centers
% dist2red: distance to red center

if nargin < 6
    iterMax = 20;
end

if nargin < 5
    knownNotRedIdx = false;
end

if nargin < 4
    trueLabels = false;
end

if nargin < 3
    knownLabels = false;
end

centers = sskppInit(data,k, knownLabels, trueLabels);
[labels, centers] = ssLloyds(data, centers, knownLabels, trueLabels, knownNotRedIdx, iterMax);
end