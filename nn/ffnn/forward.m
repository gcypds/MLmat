function [W,H,S] = forward(net,X)
L = net.numLayers;
S = cell(L,1);
W = S;
H = S;
for l=1:L  
 if l==1
   W{l} = [net.IW{1} net.b{l}];
   weightFcn = net.inputWeights{1}.weightFcn;
   S{l} = feval(weightFcn,W{l},[X;ones(1,size(X,2))]);    
 else
   W{l} = [net.LW{l,l-1} net.b{l}];
   weightFcn = net.layerWeights{l,l-1}.weightFcn;   
   S{l} = feval(weightFcn,W{l},[H{l-1};ones(1,size(X,2))]);
 end
 H{l} = feval(net.layers{l}.transferFcn,S{l});
end
% Y = net(X);