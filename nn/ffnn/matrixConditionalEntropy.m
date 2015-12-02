function [J,P] = matrixConditionalEntropy(K,L,alpha)
N = size(K,1);

Ka = K^alpha;
% error_flag = 0;
% eval('Ka = real(K^alpha);', 'error_flag = 1;');
% if (error_flag)
%     % Didn't converge; skip this one
%     Ka = K;
% end

Sy = 1/(1-alpha)*log(trace(Ka));
KL = N*K.*L;
KLa = KL^alpha;
% error_flag = 0;
% eval('KLa = real(KL^alpha);', 'error_flag = 1;');
% if (error_flag)
%     % Didn't converge; skip this one
%     KLa = KL;
% end

Sly = 1/(1-alpha)*log(trace(KLa));
J = Sly-Sy; %cost function  

%Derivative
% dSy = alpha/(1-alpha)/trace(Ka)*Uk*(Sk.^(alpha-1))*Vk';
dSy = alpha/(1-alpha)/trace(Ka)*(K^(alpha-1));
% dSly = alpha/(1-alpha)/trace(KLa)*Ukl*(Skl.^(alpha-1))*Vkl';
dSly = alpha/(1-alpha)/trace(KLa)*(KL^(alpha-1));
P = (N*L.*dSly - dSy).*K;
P = P - diag(P*ones(N,1));