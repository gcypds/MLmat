clear all
close all
clc

%% Function definitions:

g = @(xi,xj,s)exp(-0.5*(xi-xj)'*(xi-xj)/(s^2));
dg = @(xi,xj,s)(-0.5/s^2)*exp(-0.5*(xi-xj)'*(xi-xj)/(s^2))*(xi-xj);
f = @(z)tansig(z);
df = @(z)tansig('dn',z);
lr = 1e-4;

%% Go:
load iris_dataset
%X = zscore(irisInputs')';
X = mapminmax(irisInputs);
layerSizes = [size(X,1) 20 10 5];
L = numel(layerSizes)-1;
labels = vec2ind(irisTargets);
N = size(X,2);

ampaTargets = mapminmax(rand(layerSizes(end),size(X,2)));

net = feedforwardnet(layerSizes(2:end-1),'traingd');
net = configure(net,X,ampaTargets);
net = init(net);
% view(net)

Gx = kernel(X,g);

rng(870827)
for r=1:30
    
 for i=1:numel(net.IW)
  if net.inputConnect(i)
    net.IW{i} = 2*rand(size(net.IW{i}))-1;
  end
 end
 for i=1:numel(net.LW)
  if net.layerConnect(i)
    net.LW{i} = 2*rand(size(net.LW{i}))-1;
  end
 end
 for i=1:numel(net.b)  
  net.b{i} = 2*rand(size(net.b{i}))-1;
 end

 for iter = 1:100
 %forward propagation:
  [W,H,S] = forward(net,X);
 %kernel:
  [G,dG] = kernel(H{L},g,dg);
  V(iter) = sum(G(:))/numel(G);
 %backpropagation:
  dW = backward(net,dG,W,H,S,X);
 %update:
  net = gd(net,dW,lr);
 %plots:
  subplot(2,1,1)
  plot(1:iter,V(1:iter),'b.')
  subplot(2,L+2,L+3)
  imagesc(Gx,[0 1]); axis square
  for l=1:L
    subplot(2,L+2,L+3+l)
    imagesc(abs(W{l})); colorbar; axis square; xlabel('Input'); ylabel('Output'); title(['W' num2str(l)])
  end
  subplot(2,L+2,2*L+4)
  imagesc(G,[0 1]); axis square
  pause(0.1)
 end
 
 nets{r} = net; 
 save nets nets
end
% imagesc(G,[0 1]); colorbar; axis square


%%
for i=1:30
  [W,H,S] = forward(nets{i},X);
  G = kernel(H{L},g);
  V(i) = sum(G(:))/numel(G);
end