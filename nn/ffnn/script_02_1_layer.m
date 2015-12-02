clear all
close all
clc

%MLP with alpha-entropy for matrices as cost function

%% Function definitions:

g = @(xi,xj,s)exp(-0.5*(xi-xj)'*(xi-xj)/(s^2));
f = @(z)tansig(z);
df = @(z)tansig('dn',z);
lr = 1e-4;
alpha = 0.1;
maxiter = 1000;

%% Go:
load iris_dataset
X = mapminmax(irisInputs);
layerSizes = [size(X,1) 20 10 5 size(irisTargets,1)];
L = numel(layerSizes)-1;
labels = 2*vec2ind(irisTargets)-1;
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
 
 net0 = net;
 net1 = net;
%  net  = train(net0,X,irisTargets);
%  out  = net(X);
%  Ko   = kernel(out,g);

 J = zeros(2,maxiter);
 for iter = 1:maxiter
  
  [net1,Y] = adapt(net1,X,irisTargets);
  Ko = kernel(Y,g);
  Ko = Ko/N;  
  [Uk,Sk,Vk] = svd(Ko);  
  Ka = Uk*(Sk.^alpha)*Vk';
  J(1,iter) = log(trace(Ka))/(1-alpha);
          
 %forward propagation:
  [W,H,S] = forward(net,X);
 %kernel:
  %K = exp(-pdist2(H{L}',H{L}').^2/(2*s_y^2));
  [K,s_y] = kernel(H{L},g);
  K = K/N;  
  [Uk,Sk,Vk] = svd(K);
  Ka = Uk*(Sk.^alpha)*Vk';
  J(2,iter) = log(trace(Ka))/(1-alpha);
  
 %backpropagation:
  dW = backward2(net,alpha,s_y,K,W,H,S,X);
 %update:
  net = gd(net,dW,lr);
 %plots:
  subplot(2,1,1)
  plot(1:iter,J(:,1:iter),'.')
  subplot(2,L+2,L+3)
  imagesc(Gx,[0 1]); axis square
  for l=1:L
    subplot(2,L+2,L+3+l)
    imagesc(abs(W{l})); colorbar; axis square; xlabel('Input'); ylabel('Output'); title(['W' num2str(l)])
  end
  subplot(2,L+2,2*L+4)
  imagesc(K); axis square
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