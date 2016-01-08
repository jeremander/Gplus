function dists = getDist2AllCenters(x, centers)
    [k,~] = size(centers);
    dists = zeros(k,1);
    for cent = 1:k
        dists(cent) = sum((x - centers(cent,:)).^2);
    end
end