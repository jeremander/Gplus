function [ A,observe,truth ] = makeSBM(n,m,Lam)

% [ A,observe,truth ] = makeSBM(n,m,Lam)
% makes a realization from stochastic block model 
% coordinates of (vector) n are number of nonseeds in the blocks
% coordinates of (vector) m are number of seeds in the blocks
% Lam is the communication matrix
% A is adjacency matrix, 
% observe and truth are block labels observed and not observed

% pre-pre-alpha Sept3, 2013

K=length(n);
B=[];
truth=[];
observe=[];
for i=1:K
    temp=[];
    for j=1:K
        temp=[temp Lam(i,j)*ones(n(i)+m(i),n(j)+m(j))];
    end
    B=[B; temp];
    truth=[truth; i*ones(n(i)+m(i),1)];
    observe=[observe; -ones(n(i),1)];
    observe=[observe; i*ones(m(i),1)];
end
B=B-diag(diag(B));
A=rand(sum(n)+sum(m),sum(n)+sum(m))<B;
A=A-triu(A);
A=A+A';

mix=randperm(sum(n)+sum(m));
A=A(mix,mix);
observe=observe(mix);
truth=truth(mix);


end

