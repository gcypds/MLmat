clear all
close all
clc

%MLP with alpha-entropy for matrices as cost function

load wine_dataset

arq = {8,9,10,11,12,13,14,15,16,17,18,19,20,[15 5],[15 10],[15 15],[15 20],[20 5],[20 10],[20 15],[20 20]};

alpha = 1.01;
R = 10;

opts = optimset('Display','iter','GradObj','on');
Aneq = []; bneq = [];
Aeq = []; beq = [];
lb = []; ub = [];
    
perf   = zeros(numel(arq),R);
J      = zeros(numel(arq),R);
optim_res = cell(numel(arq),R);
nets   = cell(numel(arq),R);
    
for a=1:numel(arq) 
  nodes = [size(wineInputs,1) arq{a} 3];
  numLayers = numel(nodes)-1;  
  parfor r = 1:R      
    [Y,ll,trIdx,teIdx] = class_sampling(wineInputs,vec2ind(wineTargets),40);
    [X,PS] = mapstd(Y,0,1);
    X = X(:,trIdx);
    trLabels = ll(trIdx);    
    teLabels = ll(teIdx);    
    N = sum(trIdx);
    L = (pdist2(trLabels',trLabels')==0)/N; %target kernel
    
    net0 = patternnet(arq{a});
    net0 = configure(net0,X,wineTargets);
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
    w0 = cell2vec(W,b,nodes);
    
    opts_r = opts;
    opts_r.MaxFunEvals = 200*numel(w0);
    opts_r.MaxIter = 200*numel(w0);
    
    fun = @(omega)costfunc(omega,X,L,nodes,alpha);
    nonlcon = @(omega)matrixsize(omega,nodes);
    try
%     [w,fval,exitflag,output] = fmincon(fun,w0,Aneq,bneq,Aeq,beq,lb,ub,nonlcon,opts);
%     [w,fval,exitflag,output] = fminsearch(fun,w0,opts_r);
      [w,fval,exitflag,output] = fminunc(fun,w0,opts);    
      J(a,r) = fval;
      optim_res{a,r}.exitflag = exitflag;
      optim_res{a,r}.output = output;  
    
      [W,b] = vec2cell(w,nodes);
      nets{a,r}.W = W;
      nets{a,r}.b = b;
    
    %knn classify
      X = mapstd('apply',Y,PS);
      H = forward2(W,b,X);
      Ytr = H{end}(:,trIdx);
      Yte = H{end}(:,teIdx);
      Class = knnclassify(Yte', Ytr', trLabels', 4);    
      perf(a,r) = sum(Class~=teLabels')/numel(Class);     
      Class = knnclassify(Ytr', Ytr', trLabels', 4);    
      [r perf(a,r)]
    catch
      perf(a,r) = NaN;
    end
    
  end  
  [a numel(arq)]    
  [mean(perf(1:a,:),2) std(perf(1:a,:),[],2)]    
  save wine_FFCE nets perf J optim_res
end