function [dW,db] = backward2(P,W,X,H,S)
L = numel(W);
N = size(X,2);
dW = cell(L,1);
for l=L:-1:1
 if l==L
  U = H{l}.*purelin('dn',S{l});
 else
  U = (W{l+1}'*U).*tansig('dn',S{l});
 end
 if l==1
  dW{l} = U*P*X';
 else
  dW{l} = U*P*H{l-1}';      
 end        
 db{l} = U*P*ones(N,1);
end