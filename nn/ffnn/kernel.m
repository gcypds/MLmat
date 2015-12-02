function [G,s,dG] = kernel(H,g,dg)

N = size(H,2);
q = size(H,1);
G = pdist2(H',H');
s = median(median(G));
G = exp(-G.^2/(2*s^2));


% G = zeros(N,N);
% 
if nargin>2
  dG = zeros(q,N,N);
end
% for i=1:N
%  for j=i:N;
%   G(i,j)  = g(H(:,i),H(:,j),s);
%   G(j,i)  = G(i,j);
%   if nargin>2
%    dG(:,i,j)  = dg(H(:,i),H(:,j),s);
%    dG(:,j,i)  = -dG(:,i,j);
%   end 
%  end
% end