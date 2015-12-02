function [J,P] = itlMCE(K,L,a)
%
%[H,dH] = itlMCE(K,L,alpha) computes $\alpha$-order Renyi's conditional
%         entropy for matrices and its derivative w.r.t. the input.
%INPUTS:
% K \in \Real^{N x N}  : Input kernel matrix.
% L \in R^{N x N}      : Kernel for aligment
% alpha \in \Real^+    : Entropy order (default=2).
%OUTPUTS:
% H \in Real           : Entropy value.
% dH \in \Real^{N x N} : Entropy derivative w.r.t. K.
%
% Created on Wed Dec  2 16:43:08 2015
% @author: David C\'ardenas-Pe\~na

N = size(K,1);

if nargin<3 %default alpha:
  a=2;
end

[V,D] = eig(K);
Ka = real(V*(D.^(a-1))/V);
J = 1/(a-1)*log(trace(Ka));

if nargout > 1
  
end  
% Ka = K^a;
% % error_flag = 0;
% % eval('Ka = real(K^alpha);', 'error_flag = 1;');
% % if (error_flag)
% %     % Didn't converge; skip this one
% %     Ka = K;
% % end
% 
% Sy = 1/(1-a)*log(trace(Ka));
% KL = N*K.*L;
% KLa = KL^a;
% % error_flag = 0;
% % eval('KLa = real(KL^alpha);', 'error_flag = 1;');
% % if (error_flag)
% %     % Didn't converge; skip this one
% %     KLa = KL;
% % end
% 
% Sly = 1/(1-a)*log(trace(KLa));
% J = Sly-Sy; %cost function  
% 
% %Derivative
% % dSy = alpha/(1-alpha)/trace(Ka)*Uk*(Sk.^(alpha-1))*Vk';
% dSy = a/(1-a)/trace(Ka)*(K^(a-1));
% % dSly = alpha/(1-alpha)/trace(KLa)*Ukl*(Skl.^(alpha-1))*Vkl';
% dSly = a/(1-a)/trace(KLa)*(KL^(a-1));
% P = (N*L.*dSly - dSy).*K;
% P = P - diag(P*ones(N,1));
