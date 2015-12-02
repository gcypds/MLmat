clear all
close all
clc

%MLP with alpha-entropy for matrices as cost function

load '/home/dcardenasp/Documents/Personal/MRI_CLASS/codes/data/ADNImasked1.mat'

adniTargets = l;
adniInputs = X';

ARQ = {[30 30],[30 20],[30 10],[20 20],[20 10],[15 15],[15 10],[10 10],[30 30 3],[30 20 3],[30 10 3],[20 20 3],[20 10 3],[15 15 3],[15 10 3],[10 10 3]};

lr = 1e-5;
min_grad = 1e-5;
alpha = 1.01;
R = 10;

opts = optimset('Display','iter','GradObj','on');
Aneq = []; bneq = [];
Aeq = []; beq = [];
lb = []; ub = [];

max_fail = 200;

perf      = zeros(numel(ARQ),R);
Jtr       = zeros(numel(ARQ),R);
Jte       = zeros(numel(ARQ),R);
optim_res = cell(numel(ARQ),R);
nets      = cell(numel(ARQ),R);
    

[AA,RR] = ndgrid(1:numel(ARQ),1:R);

%Select classes 1 and 3
[Y,ll] = class_sampling(adniInputs,adniTargets',[52 0 40]);
CV = cvpartition(numel(ll),'kfold',10);

parfor ii=1:numel(ARQ)
  [AA(ii) numel(ARQ) RR(ii) R] 
  
  trIdx = CV.training(RR(ii));
  teIdx = CV.test(RR(ii));
  
  nodes = [size(adniInputs,1) ARQ{AA(ii)}];
  numLayers = numel(nodes)-1;  
  
  [Xtr,PS] = mapstd(Y(:,trIdx),0,1);
  Xte = mapstd('apply',Y(:,teIdx),PS);
  trLabels = ll(trIdx);
  teLabels = ll(teIdx);
  Ltr = (pdist2(trLabels',trLabels')==0)/sum(trIdx); %target kernel
  Lte = (pdist2(teLabels',teLabels')==0)/sum(teIdx); %target kernel
    
  net0 = patternnet(ARQ{AA(ii)});
  net0 = configure(net0,Xtr,eye(nodes(end)));
  net0 = init(net0);

  %Initialization:
  W = cell(numLayers,1);
  b = cell(numLayers,1);
  W{1} = net0.IW{1};
  b{1} = net0.b{1};
  for l = 2:numLayers
   W{l} = net0.LW{l,l-1};
   b{l} = net0.b{l};
  end
  w = cell2vec(W,b,nodes);

  max_iter = 200*numel(w);

  fun_tr = @(omega)costfunc(omega,Xtr,Ltr,nodes,alpha);
  fun_te = @(omega)costfunc(omega,Xte,Lte,nodes,alpha);
  nonlcon = @(omega)matrixsize(omega,nodes);

  figure(1)
  hold on
    
  f_ant = Inf;
  fails = 0;
  iter = 1;
  grad = Inf;
  fte = Inf;
  w_opt = w;
  while iter < max_iter
    [ftr,df_dw] = fun_tr(w);
    fte = fun_te(w);
    grad = norm(df_dw);
    fprintf('%d\t%e\t%e\t%d\n',iter,fte,grad,fails)
    plot(iter,ftr,'b+'); drawnow
    plot(iter,fte,'g+'); drawnow
    w = w - lr*real(df_dw);
    if f_ant < fte
      fails = fails + 1;  
    else
      f_ant = fte;
      fails = 0;      
      w_opt = w;
    end
    if (fails>=max_fail) || (grad<=min_grad)
      iter = max_iter+2;
    end
    iter = iter + 1;
  end
   
  clf
    
  if fails>=max_fail
    fprintf('Optim finished by FAILS\n')
  elseif grad<=min_grad
    fprintf('Optim finished by GRADIENT\n')      
  else
    fprintf('Optim finished by ITERATIONS\n')      
  end    
        
  Jtr(ii) = ftr;
  Jte(ii) = fte;
  [W,b] = vec2cell(w_opt,nodes);
  nets{ii}.W = W;
  nets{ii}.b = b;
    
  %knn classify
  Xtr = mapstd('apply',Y,PS);
  H = forward2(W,b,Xtr);
  Ytr = H{end}(:,trIdx);
  Yte = H{end}(:,teIdx);
  Class = knnclassify(Yte', Ytr', trLabels', 4);    
  perf(ii) = sum(Class~=teLabels')/numel(Class);     
  Class = knnclassify(Ytr', Ytr', trLabels', 4);
  tmp = sum(Class~=trLabels')/numel(Class);
  fprintf('%d\t%e\t%e\n',RR(ii),perf(ii),tmp)

%   [AA(ii) numel(ARQ) RR(ii) R]    
%   [mean(perf(1:a,:),2) std(perf(1:a,:),[],2)]    
%   save adni_FFCE nets perf J optim_res
end

save adni_1_3_FFCE nets perf Jtr Jte