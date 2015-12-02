clear all
close all
clc

%MLP with alpha-entropy for matrices as cost function

%% Function definitions:

gauss_k = @(xi,xj,s)exp(-0.5*(xi-xj)'*(xi-xj)/(s^2));
f{1} = @(z)tansig(z);
df{1} = @(z)tansig('dn',z);
f{2} = @(z)tansig(z);
df{2} = @(z)tansig('dn',z);
f{3} = @(z)purelin(z);
df{3} = @(z)purelin('dn',z);
lr = 1e-4;
alpha = 1.1;
max_iter = 1e5;
max_fail = 20;

%% Database:
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
[labels,sorting]=sort(labels');
X = mapminmax(images(:,sorting));

batch = 200;
[X,labels,trIdx,teIdx] = class_sampling(X,labels,batch);

targets = 2*full(ind2vec(labels+1))-1;
labels = vec2ind(targets);
N = size(X,2);
L = (pdist2(labels',labels')==0)/N; %target kernel

%% Initialization
siz0 = [28 28];
[W,b,linInd] = netInitDownsampler(siz0,2,3);
for l=numel(W)
   tmp = reshape(rands(numel(W{l})),size(W{l}));
   tmp_norm = sum(tmp(:).^2);
   beta = 0.7*size(W{l},1).^(1/size(W{l},2));
   W{l} = beta*tmp/tmp_norm;
end
net.W = W;
net.b = b;
numLayers = numel(net.W);

Gx = kernel(X,gauss_k);
J = zeros(1,max_iter);


%% Go:
net.cost = Inf;

for iter = 1:max_iter
  fprintf('%d-%d...',iter,max_iter)
  %Forward:  
  fprintf(' F ')
  Z = cell(numLayers,1);
  H = cell(numLayers,1);
  for l=1:numLayers  
   if l==1      
     Z{l} = bsxfun(@plus,W{l}*X,b{l});
   else
     Z{l} = bsxfun(@plus,W{l}*H{l-1},b{l});
   end
   H{l} = feval(f{l},Z{l});
  end
  %cost function:
  fprintf(' C ')
  [K,s_y] = kernel(H{numLayers},gauss_k);
  K = K/N;  
  [Uk,Sk,Vk] = svd(K);
  Ka = Uk*(Sk.^alpha)*Vk';
  Sy = 1/(1-alpha)*log(trace(Ka));
  KL = N*K.*L;
  [Ukl,Skl,Vkl] = svd(KL);
  KLa = Ukl*(Skl.^alpha)*Vkl';
  Sly = 1/(1-alpha)*log(trace(KLa));

  J(iter) = Sly-Sy; %cost function

  %Derivatives:
  fprintf(' D ') 
  dSy = alpha/(1-alpha)/trace(Ka)*Uk*(Sk.^(alpha-1))*Vk';
  dSly = alpha/(1-alpha)/trace(KLa)*Ukl*(Skl.^(alpha-1))*Vkl';
  P = (N*L.*dSly - dSy).*K;    
  dW = cell(numLayers,1);
  for l=numLayers:-1:1
    if l==numLayers
       U = H{numLayers}.*feval(df{l},Z{l});
    else
       U = (W{l+1}'*U).*feval(df{l},Z{l});
    end
    if l==1
      dW{l} = U*(P - diag(P*ones(N,1)))*X';
    else
      dW{l} = U*(P - diag(P*ones(N,1)))*H{l-1}';      
    end
    mask = zeros(size(W{l}));
    mask(linInd{l}) = 1; 
    dW{l} = dW{l}.*mask;
    db{l} = U*(P - diag(P*ones(N,1)))*ones(size(X,2),1);
  end

  %Update:
  fprintf(' U\n')
  grad = 0;
  for l=1:numLayers
    W{l} = W{l} - lr*dW{l}/norm(dW{l});
    b{l} = b{l} - lr*db{l}/norm(db{l});
    grad = grad + norm(dW{l}) + norm(b{l});        
  end
    
  %Stop criteria:
  if grad < 1e-5
    break;
  end
    
  if net.cost > J(iter)
    net.W = W;
    net.b = b;
    net.cost = J(iter);
    fails = 0;
  elseif fails < max_fail
    fails = fails + 1;
  else      
    break;      
  end
  
  %Plots:
  subplot(2,1,1)    
  plot(1:iter,J(:,1:iter),'.',[1 iter],[net.cost net.cost],'k')
  title([grad fails]); legend('IT')
  subplot(2,numLayers+2,numLayers+3)
  imagesc(Gx,[0 1]); axis square
%   for l=1:numLayers
%     subplot(2,numLayers+2,numLayers+3+l)
%     imagesc(abs(W{l})); axis square; xlabel('Input'); ylabel('Output'); title(['W' num2str(l)])
%   end
  subplot(2,numLayers+2,2*numLayers+4)
  imagesc(K); axis square
  drawnow
end