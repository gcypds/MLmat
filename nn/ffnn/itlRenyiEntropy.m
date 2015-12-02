function [H,dH] = itlRenyiEntropy(K,a)
%
%[H,dH] = itlRenyiEntropy(K,alpha) computes $\alpha$-order Renyi's entropy and
%         its derivative w.r.t. the input.
%INPUTS:
% K \in \Real^{N x N}       : Input kernel matrix.
% alpha \in \Real^+         : Entropy order (default=2).
%OUTPUTS:
% H \in Real                : Entropy value.
% dH \in \Real^{N x N}      : Entropy derivative w.r.t. K.
%
% Created on Wed Dec  2 16:43:08 2015
% @author: David C\'ardenas-Pe\~na

[V,D] = eig(K);


H = (a/(1-a))*trace(1)