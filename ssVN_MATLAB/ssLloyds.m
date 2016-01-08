function [ labels, centers ] = ssLloyds( data, centers, knownLabels, trueLabels, knownNotRedIdx, iterMax)
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
%iterMax: maximum number of iterations to run (defaults to 20).
%############### OUTPUTS ##############
% labels: final labels
% centers: final centers

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

[n,~] = size(data);
[k,~] = size(centers);

lab0 = -ones(n,1);
labels = lab0;

    for iter = 1:iterMax

        %re-label
        for i = 1:n
            isNotRed = any(knownNotRedIdx==i);
            labels(i) = getMinIdx(data(i,:), centers, isNotRed);
        end

        %adjust for semi-supervised
        labels(knownLabels) = trueLabels(knownLabels);

        %re-center
        for gr = 1:k
            centers(gr,:) = mean(data(labels==gr, :), 1); %column mean
        end

        if sum(labels==lab0) == n %if all match
            break
        end

        lab0=labels;
    end
end
