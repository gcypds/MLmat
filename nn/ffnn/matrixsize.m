function [c,ceq] = matrixsize(omega,arq)

c=[];
[W] = vec2cell(omega,arq);

ceq = zeros(numel(arq)-1,1);

for l = 1:numel(ceq)
  ceq(l) = trace(W{l}*W{l}')-size(W{l},1);  
end