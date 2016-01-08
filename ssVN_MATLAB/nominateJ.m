function [order]=nominateJ(A, observe, k, d, embedFun)
% INPUTS:
% A :: the adjacency matrix (n-by-n, for vertices 1,2...n)
% observe ::observe(i) is -1 if vertex i is ambiguous (block membership unknown),
%   and observe(i) is otherwise 1,2,...K according as vertex i is known to be in blocks 1,2,...K, respectively
%   and observe(i) is 0 if vertex i is NOT class 1 (red) but unknown which of
%   the other k-1 it is.
% k :: the number of blocks
% d :: (optional) dimension for embedding
% embedFun :: (optional) function that has the signature embedFun(A, d) that
%   outputs n x d data given adjacency matrix A
% OUTPUTS:

% order :: the ordering of the names i of the ambiguous vertices 
%   such that lower index is higher belief of the vertex being 
%   in block 1, which is the block of interest.

% get ``data'' through call to ASE
if nargin < 5
    if nargin == 4
        embedFun = @(x)standardASE(x,d);
    else
        embedFun = @standardASE;
    end
end
ase = embedFun(A);

%extract semi-supervised format
knownLabels = find(observe>0);
trueLabels = observe;
knownNotRed = find(observe==0);

% get centers
[~, centers] = ssKpp(ase, k, knownLabels, trueLabels, knownNotRed, 100);

% get order
[n, ~] = size(ase);
dist2red = zeros(n, 1);
for i = 1:n
    x=ase(i,:);
    dist2red(i) = sum((x - centers(1,:)).^2);
end
%semi-supervise nomination
% maxD = max(dist2red);
% dist2red(knownNotRed) = dist2red(knownNotRed)+maxD;
% knownOther = knownLabels(trueLabels(knownLabels)>1);
% dist2red(knownOther) = dist2red(knownOther)+maxD;
% 
% knownRed = knownLabels(trueLabels(knownLabels)==1);
% dist2red(knownRed) = 0;
% BLOCK ABOVE: commented out bc we dont report on non ambiguous vertices
dist2red(knownLabels) = Inf; %note that because we will exclude these this is OKAY, even though this includes definitely red labels.
dist2red(knownNotRed) = Inf;
[~, order] = sort(dist2red);
numUnambig = length(order) - length(knownLabels) - length(knownNotRed);
order = order(1:numUnambig);
end

function embeddedCoords = standardASE(A, d)
    
    if nargin < 2
        [n,~]=size(A);
        maxDim = min(n,20);
        [~, evalsA]=eigs(A,maxDim);
        theDims= zhuGhodsi(evalsA, 2);
        d = theDims(2);
    end
    [XA, evalsA]=eigs(A,d);
     
     embeddedCoords=XA*sqrt(abs(evalsA));
end

  