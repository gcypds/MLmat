function [dW] = backward(net,dG,W,H,S,X)

L = net.numLayers;
N = size(H{L},2);
dW = cell(L,1);
for l=1:L  
  dW{l} = zeros(size(W{l}));
end

D = cell(N,N,2);

for l=L:-1:1     
  dF  = feval(net.layers{l}.transferFcn,'dn',S{l});
  if l==L
   for n=1:N
    for m=1:N
      D{n,m,1} = dG(:,n,m).*dF(:,n);
      D{n,m,2} = dG(:,n,m).*dF(:,m);
      dW{l} = dW{l} + D{n,m,1}*[H{l-1}(:,n);1]'...
                    - D{n,m,2}*[H{l-1}(:,m);1]';
    end
   end   
  elseif l==1
   for n=1:N
    for m=1:N
      D{n,m,1} = (W{l+1}(:,1:end-1)'*D{n,m,1}).*dF(:,n);
      D{n,m,2} = (W{l+1}(:,1:end-1)'*D{n,m,2}).*dF(:,m);
      dW{l} = dW{l} + D{n,m,1}*[X(:,n);1]'...
                    - D{n,m,2}*[X(:,m);1]';
    end
   end
  else
   for n=1:N
    for m=1:N
      D{n,m,1} = (W{l+1}(:,1:end-1)'*D{n,m,1}).*dF(:,n);
      D{n,m,2} = (W{l+1}(:,1:end-1)'*D{n,m,2}).*dF(:,m);
      dW{l} = dW{l} + D{n,m,1}*[H{l-1}(:,n);1]'...
                    - D{n,m,2}*[H{l-1}(:,m);1]';
    end
   end
  end
end