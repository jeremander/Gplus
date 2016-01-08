%test nominateJ

n=[200 150 150];
m=[20 0 0];
rho=1;
M=[.5 .3 .4; .3 .8 .6; .4 .6 .3]; 
N=.5*ones(3,3); 
Lam=rho*M+(1-rho)*N;
rng(123); % set the seed
[A, observe, truth] = makeSBM(n,m,Lam);
d = rank(Lam);
k = length(n);
numRedLeft = n(1)-m(1);

%% Test 1: test obvious case
order = nominateJ(A, observe, k, d);%, embedFun)  
reveal = truth(order)==1;
assert(sum(reveal(1:numRedLeft))==numRedLeft);


%% Test 2: test mixed
m=[20 10 10];
rng(123); % set the seed
[A, observe, truth] = makeSBM(n,m,Lam);

order = nominateJ(A, observe, k, d);%, embedFun)  
reveal = truth(order)==1;
assert(sum(reveal(1:numRedLeft))==numRedLeft); 


%% Test 3: test mixed and knownNotRed
%pick out some extra to label not red
isValid = observe==-1 & truth > 1;
validIdx = 1:length(isValid);
validIdx = validIdx(isValid);
knownNotRed = datasample(validIdx, 20);
observe(knownNotRed) = 0;
numUnambig = sum(n)-sum(m)- length(knownNotRed);

order = nominateJ(A, observe, k, d);
reveal = truth(order)==1;
assert(sum(reveal(1:numRedLeft))==numRedLeft);  

%% Test 4: test mixed and knownNotRed without d
order = nominateJ(A, observe, k);
reveal = truth(order)==1;
assert(sum(reveal(1:numRedLeft))==numRedLeft);  


%% Test 5: test fuzzy
n=[200 150 150];
m=[20 0 0];
rho=.3; %chosen based on unnamed MAP table.
M=[.5 .3 .4; .3 .8 .6; .4 .6 .3]; 
N=.5*ones(3,3); 
Lam=rho*M+(1-rho)*N;
rng(123); % set the seed
[A, observe, truth] = makeSBM(n,m,Lam);

%pick out some extra to label not red
isValid = observe==-1 & truth > 1;
validIdx = 1:length(isValid);
validIdx = validIdx(isValid);
knownNotRed = datasample(validIdx, 20);
observe(knownNotRed) = 0;
order = nominateJ(A, observe, k);

% orderedLabels = truth(order);
% c=hist(orderedLabels(1:n(1)), 1:3);

reveal = truth(order)==1;
vecprec = zeros(numRedLeft,1);
for kk = 1:numRedLeft
    vecprec(kk) = sum(reveal(1:kk))/kk;
end
assert(mean(vecprec)>.76); % higher than spectral, lower than likeli-max in unnamed MAP table.
%NB this excludes unambig, but I don't think it does in the table.
%Including unabmg would only help MAP increase .

