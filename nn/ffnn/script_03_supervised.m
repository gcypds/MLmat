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
lr = 1e-2;
alpha = 1.1;
max_iter = 1e5;
max_fail = 10;
arq = [5 10 15 20 25 30];

%% Go:
load iris_dataset
X = mapminmax(irisInputs);
labels = vec2ind(irisTargets);
N = size(X,2);
L = (pdist2(labels',labels')==0)/N; %target kernel

for a=1:numel(arq)
 
 nodes = [size(X,1) arq(a) 3];
 numLayers = numel(nodes)-1;

 for r = 1:30
    
  net0 = patternnet(arq(a));
  net0 = configure(net0,X,irisTargets);
  net0 = init(net0);
    
  %Initialization:
  W = cell(numLayers,1);
  b = cell(numLayers,1);
%   for l = 1:numLayers
%     W{l} = 2*rand(nodes(l+1),nodes(l))-1;
%     b{l} = 2*rand(nodes(l+1),1)-1;
%   end
  W{1} = net0.IW{1};
  b{1} = net0.b{1};
  for l = 2:numLayers
    W{l} = net0.LW{l,l-1};
    b{l} = net0.b{l};
  end
  Gx = kernel(X,gauss_k);
  J = zeros(1,max_iter);

  net.cost = Inf;

  for iter = 1:max_iter
    %Forward:
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
     db{l} = U*(P - diag(P*ones(N,1)))*ones(size(X,2),1);
    end

    %Update:
    grad = 0;
    for l=1:numLayers
      W{l} = W{l} - lr*dW{l}/norm(dW{l});
      b{l} = b{l} - lr*db{l}/norm(db{l});
      grad = grad + norm(dW{l}) + norm(b{l});      
    end
    
    %Stop criteria:
    if grad < lr
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
    for l=1:numLayers
      subplot(2,numLayers+2,numLayers+3+l)
      imagesc(abs(W{l})); axis square; xlabel('Input'); ylabel('Output'); title(['W' num2str(l)])
    end
    subplot(2,numLayers+2,2*numLayers+4)
    imagesc(K); axis square
    pause(0.1)
  end
  
  perf(a,r) = net.cost;
  nets{a,r} = net;
  save redes nets perf
end
end