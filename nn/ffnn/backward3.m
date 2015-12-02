function [dW] = backward3(net,alpha,P,W,H,S,X)

L = net.numLayers;
N = size(H{L},2);
dW = cell(L,1);
for l=1:L  
  dW{l} = zeros(size(W{l}));
end

D = cell(N,N,2);

for l=L:-1:1     
  %backpropagation
  dF  = feval(net.layers{l}.transferFcn,'dn',S{l});  
  if l==L
   for n=1:N
    for m=1:N
      U{n,m,1} = (H{L}(:,n)-H{L}(:,m)).*dF(:,n);
      U{n,m,2} = (H{L}(:,n)-H{L}(:,m)).*dF(:,m);      
    end
   end   
  else
   for n=1:N
    for m=1:N
      U{n,m,1} = (W{l+1}(:,1:end-1)'*U{n,m,1}).*dF(:,n);
      U{n,m,2} = (W{l+1}(:,1:end-1)'*U{n,m,2}).*dF(:,m);      
    end
   end
  end
  %Gradient:
  if l==1    
   for n=1:N
    for m=1:N      
      dW{l} = dW{l} + P(n,m)*(U{n,m,1}*[X(:,n);1]'...
                                  - U{n,m,2}*[X(:,m);1]');
    end
   end  
  else
   for n=1:N
    for m=1:N      
      dW{l} = dW{l} + P(n,m)*(U{n,m,1}*[H{l-1}(:,n);1]'...
                                  - U{n,m,2}*[H{l-1}(:,m);1]');
    end
   end  
  end
end