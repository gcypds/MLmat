function [H,S] = forward2(W,b,X)
L = numel(W);
S = cell(L,1);
H = S;
for l=1:L  
 if l==1
   S{l} = bsxfun(@plus,W{l}*X,b{l});   
 else 
   S{l} = bsxfun(@plus,W{l}*H{l-1},b{l});   
 end
 if l==L
   H{l} = purelin(S{l});
 else
   H{l} = tansig(S{l});
 end
end