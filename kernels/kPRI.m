function X = kPRI(S,starting,gam,opts)
% kPRI computes a compressed version of a dataset matrix using the 
%  Principle of Relevant Information (PRI).
%
%  X = kPRI(S,gamma,Xo)
%  S     - M-by-P original data matrix. Rows correspond to observations, 
%          columns correspond to variables.
%  Xo    - N-by-P initial compressed data matrix. Rows correspond to 
%          observations, columns correspond to variables.
%  gamma - regularization term between X's entropy and X-S divergence.
%  X     - N-by-P resulting compressed data matrix. Rows correspond to 
%          observations, columns correspond to variables.
%
%  X = kPRI(S,N,gamma)
%  S     - M-by-P original data matrix. Rows correspond to observations, 
%          columns correspond to variables.
%  N     - Number of samples in the compressed data matrix. This number is
%          used to initialize the algorithm.
%  gamma - regularization term between X's entropy and X-S divergence.
%  X     - N-by-P resulting compressed data matrix. Rows correspond to 
%          observations, columns correspond to variables.
%
%  X = kPRI(S,N,gamma,opts)
%  opts  - options structure:
%          opts.MaxIter = 100; Maximum number of iterations to perform the
%                              algorithm
%          opts.TolX = 1e-4; Solution gradient stopping criterion 
% 
% The conventional unsupervised learning algorithms (clustering, principal 
% curves, vector quantization) are solutions to an information optimization 
% problem that balances the minimization of data redundancy with the 
% distortion between the original data and the solution, expressed by
%
% L[p(x|s)] = min H(X) + gamma*D(X|S) 
%
% where s in S is the original dataset
%       x in X is a compressed version of the original data achieved 
%              through processing
%       gamma  is a variational parameter
%       H(X)   is the entropy 
%       D(X|S) is the KL divergence between the original and the compressed
%              data.
%
% Chapter: Self-Organizing ITL Principles for Unsupervised Learning
% Authors: Sudhir Rao, Deniz Erdogmus, Dongxin Xu, Kenneth Hild II
% Book: Information Theoretic Learning: Renyi's Entropy and Kernel 
%       Perspectives
% Authors: Jose C. Principe
% ISBN: 978-1-4419-1569-6 (Print) 978-1-4419-1570-2 (Online)
%__________________________________________________________________________
% Copyright (C) 2014 Signal Processing and Recognition Group
%
% Andr\'es Eduardo Castro-Ospina
% $Id: kPRI.m 2014-04-11 11:29:31 $

% Number of samples in the original data matrix
m = size(S,1);
% Number of feature in the original data matrix
p = size(S,2);

if numel(starting) == 1
  n  = starting;
  Xo = zeros(n,p);
  mi = min(S); ma = max(S);
  for j = 1:p
    lw = mi(j) + 0.25*(ma(j)-mi(j)) ;
    up = ma(j) - 0.25*(ma(j)-mi(j)) ;
    Xo(:,j) = unifrnd(lw,up,n,1);
  end
else
  Xo = starting;
  n = size(Xo,1);
end
clear starting;

if nargin >= 4
  maxit = opts.MaxIter;
  tol   = opts.TolX;
else
  maxit = 100;
  tol   = 1e-4;
end


L = -1;
% Initial compressed version
X = Xo;

ii = 0;
NN = 1;
while NN > tol && ii < maxit
    
  ii = ii + 1;
  X0 = X;
    
  Dsx = pdist2(S,X);
  sig = kScaleOptimization(Dsx);
  sig_n = Multiv_sig(sig,ii,1,gam);
  smin = sig/sqrt(m);
  if sig_n < smin
    sig_n = smin;
  end
  
  % X(t) kernel
  kerX = kExpQuad2(X,X,sig_n);
  %Information potential of X(t)
  IP = sum(kerX(:))/n^2;
  % X,S kernel
  kerXS = kExpQuad2(X,S,sig_n);
  %Cross-information potential
  CIP = sum(kerXS(:))/(n*m);
    
  c = m*CIP/(n*IP);
  O = ones(m,1);
  O1 = ones(n,1);
    
  DD = repmat(kerXS*O,1,size(X,2));
  X = -c*(kerX*X)./DD + (kerXS*S)./DD + c*diag((kerX*O1)./(kerXS*O))*X;
  
  NN = norm(X0-X,'fro')/norm(X0,'fro');
    
end

function sig_n = Multiv_sig(sig0,n,k1,Z)
  sig_n = (k1*sig0)/(1+Z*k1*n);
