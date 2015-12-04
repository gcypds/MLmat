function acc = cvKnn(X,L,K,N)
% acc = cvKnn(X,L,K,N)
%INPUTS:
% X input samples
% L input labels
% K number of neighbors (default=3)
% N number of folds (default=10)
%OUTPUT:
% acc fold-wise accuracy.

v = strfind(version,'R2015a')>0;

if exist('K')==0
  K=3;
end

if exist('N')==0
  N=10;
end

cv = cvpartition(L,'KFold',N);
acc = zeros(N,1);
for r=1:N
  Xtr = X(cv.training(r),:);  
  Ltr = L(cv.training(r));
  Xval = X(cv.test(r),:);
  Lval = L(cv.test(r));
  if v %R2015
    mdl = fitcknn(Xtr,Ltr,'NumNeighbors',K);
    outputs = predict(mdl,Xval);
  else
    [outputs]= knnclassify(Xval,Xtr,Ltr,K);  
  end
  acc(r) = sum(outputs==Lval)/numel(outputs);
end
