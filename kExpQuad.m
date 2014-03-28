function k = kExpQuad(x,s)
% kExpQuad Exponentiated quadratic kernel. 
%  k(n,m) = exp( -(x_n-x_m)'*(x_n-x_m)/(2*s^2) ) 
%  k(n,m) = exp( -(x_n-x_m)'*(x_n-x_m)/(2*s^2) ) 
%
%  K = kExpQuad(X,s) returns a matrix K containing the similarity, using
%  an exponentiated quadratic kernel, between each pair of observations.
%  X      - M-by-N data matrix. Rows correspond to observations, columns
%           correspond to variables. 
%  s      - scale parameter. default: s=1.
%  K      - M-by-M kernel matrix
%
%__________________________________________________________________________
% Copyright (C) 2014 Signal Processing and Recognition Group
% David C\'ardenas Pe\~na
% $Id: kExpQuad.m 2014-03-28 10:37:45 $

if nargin==1
  s = 1;
end

k = exp(-squareform(pdist(x).^2)/(2*s^2));
