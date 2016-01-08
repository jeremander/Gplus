%test sskppInit

data = rand(10, 3);
k=3;

%% Test 1: test that number of centers ok with known/true labels
trueLabels=[1,1,1,1,2,2,2,2,3,3];
knownLabels=[1,2];
centers = sskppInit(data,k, knownLabels, trueLabels);
[n, m] = size(centers);
assert(n==k);

%% Test 2: test that number of centers ok without known/true labels
centers = sskppInit(data,k);
[n, m] = size(centers);
assert(n==k);

%% Test 3: test that supervised centers are indeed centroids
trueLabels=[1,1,1,1,2,2,2,2,3,3];
knownLabels=[1,2,5];
centers = sskppInit(data,k, knownLabels, trueLabels);
assert(all(centers(1,:)==mean(data(1:2,:),1)));
assert(all(centers(2,:)==data(5,:)));