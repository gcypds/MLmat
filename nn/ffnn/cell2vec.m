function vec = cell2vec(W,b,arq)
nwb = sum(arq(1:end-1).*arq(2:end)) + sum(arq(2:end));%Number of weights and biases.
vec  = zeros(nwb,1);
cnt = 1;
for layer = 1:numel(W)
  vec(cnt:cnt-1+prod(arq([layer+1 layer]))) = W{layer}(:);  
  cnt = cnt+prod(arq([layer+1 layer]));
  vec(cnt:cnt-1+arq(layer+1)) = b{layer}(:);
  cnt = cnt+arq(layer+1);
end
