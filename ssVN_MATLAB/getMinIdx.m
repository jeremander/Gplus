function lab = getMinIdx(x, centers, isNotRed)
    if nargin < 3
        isNotRed = false;
    end

    dists = getDist2AllCenters(x,centers);
    if isNotRed
        dists(1) = Inf;
    end
    [~, lab] = min(dists);
end