function [f,df] = costfunc(vecWB,X,L,arq,alpha)

%Vector to cells:
[W,b] = vec2cell(vecWB,arq);

[H,S] = forward2(W,b,X);

K = exp(-pdist2(H{end}',H{end}').^2/2)/size(X,2);
[f,P] = matrixConditionalEntropy(K,L,alpha);
[dW,db] = backward2(P,W,X,H,S);

%Cells to vector:
df = cell2vec(dW,db,arq);