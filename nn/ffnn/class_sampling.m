function [Y,ll,trIdx,teIdx] = class_sampling(X,labels,N)
%Takes N samples from each class in a dataset

Labels = unique(labels);
if numel(N)==1
  N = N*ones(numel(Labels),1);
end

trIdx = false(1,sum(N));
teIdx = false(1,sum(N));
Y     = zeros(size(X,1),sum(N));
ll    = zeros(1,sum(N));
cnt   = 1;
for l=1:numel(Labels)
  ind = labels==Labels(l);  
  tmp = X(:,ind);
  tmp = tmp(:,randperm(sum(ind),N(l)));
  if size(tmp,2)>0
    CV = cvpartition(size(tmp,2),'Holdout',0.1);
    trIdx(cnt:cnt+N(l)-1) = CV.training;
    teIdx(cnt:cnt+N(l)-1) = CV.test;      
    Y(:,cnt:cnt+N(l)-1) = tmp;
    ll(cnt:cnt+N(l)-1) = Labels(l);
  end
  cnt = cnt+N(l);
end