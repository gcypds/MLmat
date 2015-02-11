function [sopt,vopt]= kScaleOptimization_info(x,s0)
% Automatic tuning of the scale parameter for the exponentiated quadratic
% kernel: y = exp(d.^2/(2*s^2))
% FORMAT [sigma, value] = kScaleOptimization(d,s0)
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
% $Id: kScaleOptimization_info.m 2014-02-22 22:40:00 $

if nargin == 1
  s0 = median(x(:));
end

f = @(s)obj_fun(s,x);
[sopt, vopt] = fminsearch(f,s0);


%%%%% objective function %%%%%%%%%%%
function [v] = obj_fun(s,x)

k = exp(-x.^2/(2*s^2));
v = - var(mean(k,1));