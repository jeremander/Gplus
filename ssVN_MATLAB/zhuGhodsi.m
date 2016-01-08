function q = zhuGhodsi(d, n)
%   ## Given a decreasingly sorted vector, return the given number of elbows
%   ##
%   ## Args:
%   ##   d: the decreasingly sorted vector (e.g. a vector of standard deviations)
%   ##   n: the number of returned elbows.
%   ## Return:
%   ##   q: a vector of length n.
%   ##
%   ## Reference:
%   ##   Zhu, Mu and Ghodsi, Ali (2006), "Automatic dimensionality selection from
%   ##   the scree plot via the use of profile likelihood", Computational
%   ##   Statistics & Data Analysis
%   
  d = abs(d);
  d = sort(d, 'descend');
  d = d(d > 1e-7);
  p = length(d);
  assert(p > 0,'d must have elements that are larger than the threshold 1e-7');
               
  lq =zeros(p,1);                   % log likelihood, function of q
  for q = 1:(p-1) 
    mu1 = mean(d(1:q));
    mu2 = mean(d((q+1):end));              % = NaN when q = p
    sigma2 = (sum((d(1:q) - mu1).^2) + sum((d((q+1):end) - mu2).^2)) ./(p - 1 - (q < p));
    lq(q) = sum( log(normpdf(  d(1:q ), mu1, sqrt(sigma2) ))) + sum( log(normpdf(d((q+1):end), mu2, sqrt(sigma2))));
  end
  mu1 = mean(d);
  sigma2 = sum((d - mu1).^2) ./(p - 1);
  lq(p) = sum( log(normpdf(  d, mu1, sqrt(sigma2) )));
  
  
  [~, q] = max(lq);
  if (n > 1 && q < p) 
    q = [q, q + zhuGhodsi(d((q+1):p), n-1)];
  end
end
