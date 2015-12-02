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
lr = 1e-1;
alpha = 1.01;
max_iter = 1e5;
max_fail = 200;

%% Database:
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
[labels,sorting]=sort(labels');
[X,PS]  = mapminmax(images(:,sorting));
X = X(PS.xmax~=PS.xmin,:);

batch = 10;
[X,labels,trIdx,teIdx] = class_sampling(X,labels,batch);

targets = 2*full(ind2vec(labels+1))-1;
N = [sum(trIdx) sum(teIdx)];
L{1} = (pdist2(labels(trIdx)',labels(trIdx)')==0)/sum(trIdx); %target kernel
L{2} = (pdist2(labels(teIdx)',labels(teIdx)')==0)/sum(teIdx); %target kernel

Arqs = {[15],[30],[60],[15 15],[30 15],[60 15],[30 30],[60 30],[60 15],[60 60]};

JJ = zeros(30,numel(Arqs));
NETS = cell(30,numel(Arqs));
%%
for a = 1:numel(Arqs)
for r=1:10
    [a numel(Arqs) r]
%% Initialization
   
siz0 = [28 28];
arq = [size(X,1) Arqs{a} size(targets,1)];
for l=1:numel(arq)-1
   tmp = reshape(rands(arq(l)*arq(l+1)),[arq(l+1) arq(l)]);
   tmp_norm = sum(tmp(:).^2);
   beta = 0.7*arq(l+1).^(1/arq(l));
   W{l} = beta*tmp/tmp_norm;
   b{l} = beta*rands(arq(l+1))/tmp_norm;
end
net.W = W;
net.b = b;
numLayers = numel(net.W);

Gx = kernel(X,gauss_k);
J = zeros(2,max_iter);


%% Go:
net.cost = Inf;

for iter = 1:max_iter
  fprintf('%d-%d...',iter,max_iter)

%   CV = cvpartition(size(X,2),'holdout',0.3);
%   trIdx = CV.training;
%   teIdx = CV.test;
%   L{1} = (pdist2(labels(trIdx)',labels(trIdx)')==0)/sum(trIdx); %target kernel
%   L{2} = (pdist2(labels(teIdx)',labels(teIdx)')==0)/sum(teIdx); %target kernel

  
  %Forward:  
  fprintf(' F ')
  Z = cell(numLayers,2);
  H = cell(numLayers,2);
  for l=1:numLayers  
   if l==1      
     Z{l,1} = bsxfun(@plus,W{l}*X(:,trIdx),b{l});
     Z{l,2} = bsxfun(@plus,W{l}*X(:,teIdx),b{l});
   else
     Z{l,1} = bsxfun(@plus,W{l}*H{l-1,1},b{l});
     Z{l,2} = bsxfun(@plus,W{l}*H{l-1,2},b{l});
   end
   H{l,1} = feval(f{l},Z{l,1});
   H{l,2} = feval(f{l},Z{l,2});
  end
  %cost function:
  fprintf(' C ')  
  %test:
  K = kernel(H{numLayers,2},gauss_k);
  K = K/N(2);
  J(2,iter) = matrixConditionalEntropy(K,L{2},alpha);
  %train:
  K = kernel(H{numLayers,1},gauss_k);
  K = K/N(1);
  [J(1,iter),Ka,KLa,Uk,Sk,Vk,Ukl,Skl,Vkl] = matrixConditionalEntropy(K,L{1},alpha);

  %Derivatives:
  fprintf(' D ') 
  dSy = alpha/(1-alpha)/trace(Ka)*Uk*(Sk.^(alpha-1))*Vk';
  dSly = alpha/(1-alpha)/trace(KLa)*Ukl*(Skl.^(alpha-1))*Vkl';
  P = (N(1)*L{1}.*dSly - dSy).*K;    
  dW = cell(numLayers,1);
  for l=numLayers:-1:1
    if l==numLayers
       U = H{numLayers}.*feval(df{l},Z{l});
    else
       U = (W{l+1}'*U).*feval(df{l},Z{l});
    end
    if l==1
      dW{l} = U*(P - diag(P*ones(N(1),1)))*X(:,trIdx)';
    else
      dW{l} = U*(P - diag(P*ones(N(1),1)))*H{l-1}';      
    end        
    db{l} = U*(P - diag(P*ones(N(1),1)))*ones(N(1),1);
  end

  %Update:
  fprintf(' U\n')
  grad = 0;
  for l=1:numLayers
    W{l} = W{l} - lr*dW{l};
    b{l} = b{l} - lr*db{l};
    grad = grad + norm(dW{l}) + norm(b{l});        
  end
    
  %Stop criteria:
  if grad < 1e-5
    break;
  end
    
  if net.cost > J(2,iter)
    net.W = W;
    net.b = b;
    net.cost = J(2,iter);
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
  for l=1:numLayers
    subplot(2,numLayers+2,numLayers+3+l)
    imagesc(abs(W{l})); axis square; xlabel('Input'); ylabel('Output'); title(['W' num2str(l)])
  end
  subplot(2,numLayers+2,2*numLayers+4)
  imagesc(K); axis square
  drawnow
end
NETS{r,a} = net;
JJ(r,a) = J(2,iter);
save MNIST_NETS NETS JJ
end
end