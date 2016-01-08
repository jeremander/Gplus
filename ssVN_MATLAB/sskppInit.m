function [ centers ] = sskppInit( data, k, knownLabels, trueLabels )
%Chooses the centers according to D^2 weighting while taking supervised
%labels into account
%############### INPUTS ###############
%data: n x d data matrix
%k: the number of groups
%knownLabels: subset of 1:n indicating indices of data with labels
% trueLabels: vector of size n that holds the true labels.
%             trueLabels(knownLabels) must be between 1 and k
%############### OUTPUTS ##############
% centers: k x d matrix with the ith row corresponding to ith center
%          Indices are preserved from trueLabels(knownLabels).  That is,
%          suppose trueLabels(knownLabels) = [1, 2, , 2, 3].
%          Then, centers(1,:) = data(knownLabels(1),:)
%          and centers(2,:) = mean(data(knownLabels(2:3), :), 1).

    [n,d] = size(data);
    centers = NaN*ones(k, d);
    remainingGr = 1:k;

    if nargin > 2
        %semi-supervise
        knownGroups = unique(trueLabels(knownLabels));
       
        for gr = knownGroups(:)'
            knownMembers= knownLabels(trueLabels(knownLabels)==gr);
            centers(gr,:) = mean(data(knownMembers, :), 1); %column mean
        end
        remainingGr = setdiff(remainingGr, knownGroups);
    else %sample uniformly if no supervision
        knownLabels=false; % for use in indexing later
        remainingGr = 2:k;
        centers(1,:) = data(randsample(1:n,1),:); 
    end

    %now choose remaining centers with D^2 weighting
    for newCenter = remainingGr
        dSq = zeros(n,1);
        for i = 1:n
            dSq(i) = getDsq(data(i,:), centers);
        end
        dSq(knownLabels) = 0; % don't choose supervised guys

        theChosenOne = randsample(1:n,1,true,dSq);
        centers(newCenter,:) = data(theChosenOne,:);
    end
end


function dSq = getDsq(x, centers)
dists = getDist2AllCenters(x,centers);
dSq = min(dists);
end