function [J,dJ] = itlME(K,a)
%
%[H,dH] = itlME(K,L,alpha) computes $\alpha$-order Renyi's entropy for
%         matrices and its derivative w.r.t. the input.
%INPUTS:
% K \in \Real^{N x N}  : Input kernel matrix.
% alpha \in \Real^+    : Entropy order (default=2).
%OUTPUTS:
% H \in Real           : Entropy value.
% dH \in \Real^{N x N} : Entropy derivative w.r.t. K.
%
% Created on Wed Dec  2 16:43:08 2015
% @author: David C\'ardenas-Pe\~na

N = size(K,1);

K = K/trace(K);

if nargin<2 %default alpha:
  a=2;
end

[V,D] = eig(K);
Ka = real(V*(D.^(a))/V);
trKa = trace(Ka);
J = 1/(1-a)*log(trKa);

if nargout > 1
  dJ = real(V*(D.^(a-1))/V);
  dJ = (a/(1-a))*dJ/trKa;
end