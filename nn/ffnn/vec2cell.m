function [W,b] = vec2cell(vecWB,arq)
W = cell(numel(arq)-1,1);
b = cell(numel(arq)-1,1);
cnt = 1;
for layer = 1:numel(W)
  W{layer} = reshape(vecWB(cnt:cnt-1+prod(arq([layer+1 layer]))),arq([layer+1 layer]));
  cnt = cnt+prod(arq([layer+1 layer]));
  b{layer} = reshape(vecWB(cnt:cnt-1+arq(layer+1)),[arq(layer+1) 1]);
  cnt = cnt+arq(layer+1);
end