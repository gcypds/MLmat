function net = gd(net,dW,lr)

L = net.numLayers;
for l=1:L
  if l==1
    net.IW{1} = net.IW{1} - lr*dW{l}(:,1:end-1);
  else
    net.LW{l,l-1} = net.LW{l,l-1} - lr*dW{l}(:,1:end-1);
  end
  net.b{l} = net.b{l} - lr*dW{l}(:,end);
end