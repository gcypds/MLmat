function [sopt,vopt]= sigma_tune(x,s0)
% Automatic tuning of the scale parameter for the exponentiated quadratic
% kernel: y = exp(d.^2/(2*s^2))
% FORMAT [sigma, value] = sigma_tune(d,s0)
% d     - data points, usually: d = pdist(x);
% s0    - starting point for the search
% sigma - achieved optimum value
% value - objective function value at sigma
% 
% The tuning method used here is based on the maximization of the
% transformed data variance as a function of the scale parameter, since
% lim_{s->0}{var{y(s)}} = 0 and lim_{s->inf}{var{y(s)}} = 0 and a
% suitable scale value should maximize var{y(s)}
%__________________________________________________________________________
% Copyright (C) 2014 Signal Processing and Recognition Group

% David C\'ardenas Pe\~na
% $Id: sigma_tune.m 2014-02-22 22:40:00 $

if nargin == 1
  s0 = median(x(:));
end

x = x(:);
f = @(s)obj_fun(s,x);
[sopt, vopt] = fminsearch(f,s0);
lb = 1e-3;
ub = 1e16;
%[sopt, vopt] = fmincon(f,s0,[],[],[],[],lb,[]);
vopt = -vopt;

%%%%% objective function %%%%%%%%%%%
function [v,dv] = obj_fun(s,x)
%nx = size(x,1);
%H = eye(size(x)) - (1/nx)*ones(nx,1)*ones(1,nx);
y = exp(-x.^2/(2*s^2));
%y = reshape(y,nx,nx);
%y = H*y*H;
y = y(:);
v = -var(y);
if nargout > 1
  n = numel(y);
  xy  = x.*y;
  dxy = zeros(n);
  dy  = zeros(n);
  for i=1:numel(y)
    dxy(:,i) = xy - xy(i);
    dy(:,i)  = y - y(i);
  end
  dv = sum(sum(s^(-3)*dxy.*dy))/(n^2);  
%     tmp = squareform(pdist(x(:)).^2);
%     dv = -sum(tmp(:))/(2*numel(tmp));
end
