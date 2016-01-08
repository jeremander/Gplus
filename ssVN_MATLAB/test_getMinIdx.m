%test getMinIdx

x=[1,2];
centers=[1,1; 2,7];


%% Test 1: test without isNotRed
expected = 1;
actual = getMinIdx(x, centers);
assert(expected==actual);
actual = getMinIdx(x,centers,false);
assert(expected==actual);

%% Test 2: test with isNotRed
expected = 2;
actual = getMinIdx(x, centers, true);
assert(expected==actual);