function [A,s,K,fnew] = kMetricLearningMahalanobis(X,L,labels,Q,plot_it,print_it,etav)
% metric learning - mahalanobis distance
% Function basics
%Brockmeier et. al. Neural Decoding with kernel-based metric learning
%Cardenas & Alvarez  Sigma tune Gaussian kernel with information potential
%USAGE:
% [A s] = kMetricLearningMahalanobis(X,L,labels,Q,plot_it,etav)
% Inputs:
%X \in R^{N x P} : data matrix, N: samples; p:features
%A_i \in R^{P x Q} : Rotatin matrix for mahalanobis distance dA(x,x') = d(xA,x'A)
%s_i \in R+ : initial kernel band width
%L \in R^{N x N} : kernel for aligment
%labels \in Z^{N x 1} group membership
%niter : max iter for gradient based optimization
%tol : convergence tolerance
%plot_it : if true every 10 iterations the gradient based results are shown
% Output:
% A \in R^{P x Q} learned rotation matrix by maximizing kernel centrel
% alignment -> log(rho(K_A,L))
%K_A : is computed using mahalanobis distance into a gaussian kernel, which kernel band-width
% is fixed by maximizing a information potential variance based function (see kscaleOptimization)

[~,A_i] = ml_pca(X,Q); %pca based
maxiter = 2e2;
tol = 1e-5;
%etav = [5e-3 5e-3];
    
%% optimization
sopt = kScaleOptimization(X*A_i);
A = A_i/(sqrt(2)*sopt);
s0 = 1/(2*sopt^2);
u_i = log(s0)/log(10);

vecAs = [A(:);u_i];
N = size(X,1);
H=eye(N)-1/N*ones(N,1)*ones(1,N);

eta_start=etav(1);
eta_end=etav(2);

eta = eta_start;
df = inf;
fold = 0;
fnew = inf;

gold = 0;
if plot_it
    figure(2)
    title('Projection...')
    
    figure(1)
    title('Cost function')
    xlabel('iteration')
    ylabel('Centered alignment')
    hold on
%     showfigs_c(2);
end

for ii = 1 : maxiter
    fold = fnew;
    
    [fnew,gradf,K,Y] = A_derivativeAs(vecAs,X,N,L,H);
    if fnew > fold %last A
        %fnew = fold;
        eta_start = eta_start-eta_start*.1;
        eta_end = eta_end-eta_end*.25;
    end
    if ii < maxiter/2
        eta = eta_start;
    else
        eta = eta_end;
    end
    
    % do linesearch
    dg = norm(gradf-gold,2)^2;
    gold = gradf;
    vecAs = vecAs - eta*gradf;
    df = abs(fold-fnew);
      
    if print_it
       fprintf('%d-%d -- eta = %.2e -- diff_f = %.2e - f = %.2e\n',...
                ii,maxiter,eta,df,fold)
    end
    if plot_it
        figure(1)
        scatter(ii,fnew,20,'r','filled')
        
        drawnow
    end
    if plot_it == true && mod(ii,2) == 0 || ii == 1 && plot_it == true
        
        figure(2)
        clf
        subplot(2,2,4)
        scatter3(Y(:,1),Y(:,2),Y(:,3),20,labels,'filled');
        axis off
        title('Y')
        
        subplot(2,2,3)
        imagesc(K), axis square, colorbar
        axis off
        title('K')
        
        subplot(2,2,1)
        imagesc(L), axis square, colorbar
        axis off
        title('L')
        
        subplot(2,2,2)
        imagesc(real(reshape(vecAs(1:end-1),size(A)))), colorbar
        title(['A' ' - \sigma = ' num2str(sqrt(1/(2*10^vecAs(end))),'%.2e')])
        axis off
        drawnow
    end
    
    if df < tol
      if print_it
        fprintf('Metric Learning done...diffi %.2e= \n',df)
      end
      break;
    end
    
end
A=real(reshape(vecAs(1:end-1),size(X,2),[]));
sp = kScaleOptimization(X*A);
A = A./(sqrt(2)*sp);
s = vecAs(end);
s = 10^s;

if plot_it == true
    figure(2)
    hold off
end

end


function [f, gradf,k,y,rho]= A_derivativeAs(vecas,x,n,l,h)
a=real(reshape(vecas(1:end-1),size(x,2),[]));
u = vecas(end);
s = 10^u;
sp = kScaleOptimization(x*a);
a = a/(sqrt(2)*sp);
y=x*a;
d=pdist2(y,y);
k=exp(-d.^2/2);

if any(isnan(k(:)))% ||any(isnan(k_ly(:)))
    fprintf('whoa')
    f=nan;
    %    gradf=0*logeta;
else
    trkl=trace(k*h*l*h);
    trkk=trace(k*h*k*h);
    
    grad_lk=(h*l*h)/trkl;
    grad_k=2*(h*k*h)/trkk;
    grad = grad_lk-.5*grad_k;
    p = grad.*k;
    
    p=(p+p')/2;
    grada = x'*(p-diag(p*ones(n,1)))*(x*a);
    grada = -4*real(grada(:));
    grads = trace((-k.*d.^2)*real(grad));
    grads = s*log(10)*grads; %function of u; s = 10^u
    
    gradf= real([grada;grads]);
    
    f=-real(log(trkl)-log(trkk)/2);
    rho = trkl/sqrt(trkk);
end

end





