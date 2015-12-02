clear all
close all
clc

addpath(genpath('/home/dcardenasp/Documents/MATLAB/MLmat/'))

load wine_dataset
N = size(wineInputs,2);
R = 10;
CV = cvpartition(N,'holdout',0.5);

err = zeros(R,1);
for r=1:R
 CV = repartition(CV);
 trIdx = CV.training;
 teIdx = CV.test;
 [Xtr,mu,sigma] = zscore(wineInputs(:,trIdx),[],2);
 Ltr   = wineTargets(:,trIdx);
 Xte   = bsxfun(@times,bsxfun(@minus,wineInputs(:,teIdx),mu),1./sigma);
 Lte   = wineTargets(:,teIdx);
 
 Kl = pdist2(Ltr',Ltr')==0;
 [A,s,K,fnew] = kMetricLearningMahalanobis(Xtr',Kl,vec2ind(Ltr)',0.9,false,true,[1e-4 1e-5]);
 
 Ytr = Xtr'*A;
 Yte = Xte'*A;
 
 Class = knnclassify(Yte, Ytr, vec2ind(Ltr)', 4);
 err(r) = sum(Class~=vec2ind(Lte)')/numel(Class); 
end