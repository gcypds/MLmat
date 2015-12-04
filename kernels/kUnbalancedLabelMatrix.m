function L = kUnbalancedLabelMatrix(labels)

L = double(pdist2(labels,labels)==0);
for u=unique(labels)'
  L(labels==u,labels==u) = L(labels==u,labels==u)/sum(labels==u);
end
L = L/trace(L);