%test ssLloyds
rng(123);
data = rand(10, 3)*10;
k=3;

%% Test 1: test that labeled guys dont move (w/o is not red)
trueLabels=[1,1,1,1,2,2,2,2,3,3];
knownLabels=[1,2];
centers = [5,10,1; 
            1, 2, 2; 
            7, 5, 7]; 
[l,c] = ssLloyds(data,centers,knownLabels, trueLabels);
assert(all(l(1:2)==1));

%% Test 2: test that labeled guys dont move (w is not red)
trueLabels=[1,1,1,1,2,2,2,2,3,3];
knownLabels=[1,2];
centers = [5,10,1; 
            1, 2, 2; 
            7, 5, 7]; 
knownNotRed = [9,10];

[l,c] = ssLloyds(data,centers,knownLabels, trueLabels, knownNotRed);
assert(all(l(1:2)==1));
assert(all(l(9:10)~=1));


%% Test 3: test that potential gets reduced
trueLabels=[1,1,1,1,2,2,2,2,3,3];
knownLabels=[1,2];
centers = [5,10,1; 
            1, 2, 2; 
            7, 5, 7]; 
knownNotRed = [9,10];
poten0 = 0;

 for i = 1:10
     x = data(i,:);
     d = getDist2AllCenters(x, centers);
     poten0 = poten0 + d(getMinIdx(x, centers, any(i==knownNotRed)));
 end
 
[l,c] = ssLloyds(data,centers,knownLabels, trueLabels, knownNotRed);

poten = 0;
 for i = 1:10
     x = data(i,:);
     d = getDist2AllCenters(x, c);
     poten = poten + d(l(i));
 end
 assert(poten <= poten0);
