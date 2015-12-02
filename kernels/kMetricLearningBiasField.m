function [A,a,a0,s,K,fnew] = kMetricLearningBiasField(X,R,L,labels,plot_it,print_it,etav)
% metric learning - mahalanobis distance
% Function basics
%Brockmeier et. al. Neural Decoding with kernel-based metric learning
%Cardenas & Alvarez  Sigma tune Gaussian kernel with information potential
%USAGE:
% [A a a0 s K_A] = kMetricLearningBiasField(X,R,L,labels,plot_it,print_it,etav)
% Inputs:
%X \in R^{N x 1} : data matrix, N: samples
%R \in R^{N x 3} : spatial matrix, N: samples; 3: dimensions
%L \in R^{N x N} : kernel for aligment
%labels \in Z^{N x 1} group membership
%plot_it : if true every 10 iterations the gradient based results are shown
%print_it : if true every 10 iterations the gradient based results are
%           printed
% Output:
%A \in R^{3 x 3} : bias(r) = r'*A*r + a'*r + a0
%a \in R^{3 x 3} : bias(r) = r'*A*r + a'*r + a0
%a0 \in R^{3 x 3} : bias(r) = r'*A*r + a'*r + a0
%s \in R+ : initial kernel band width
%Parameters learned by maximizing kernel centered alignment -> log(rho(K_A,L))
%K_A : is computed using mahalanobis distance into a gaussian kernel, which kernel band-width
% is fixed by maximizing a information potential variance based function (see kscaleOptimization)

maxiter = 2e2;
tol = 1e-5;
%etav = [5e-3 5e-3];
    
%% optimization
dim = size(R,2);
s  = kScaleOptimization(X);
A   = zeros(dim,dim);
a   = zeros(1,dim);
a0  = 1;
vecAs = [A(:);a(:);a0;s];
N = size(X,1);
H=eye(N)-1/N*ones(N,1)*ones(1,N);
k0 = exp(-pdist2(X,X).^2/(2*s^2));

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
    
    [fnew,gradf,K,Y] = derivatives(vecAs,X,R,N,L,H);
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
        imagesc(k0), axis square, colorbar
        title('K0')
        
        subplot(2,2,3)
        imagesc(K), axis square, colorbar
        axis off
        title('Kbias')
        
        subplot(2,2,1)
        imagesc(L), axis square, colorbar
        axis off
        title('L')
        
        subplot(2,2,2)
%         imagesc(real(reshape(vecAs(1:end-1),size(A)))), colorbar
%         title(['A' ' - \sigma = ' num2str(sqrt(1/(2*10^vecAs(end))),'%.2e')])
%         axis off
        drawnow
    end
    
    if df < tol
        fprintf('Metric Learning done...diffi %.2e= \n',df)
        break;
    end
    
end

s    = vecAs(end);
s    = 10^s;
A    = reshape(vecAs(1:dim^2),[dim dim]);
a    = reshape(vecAs(dim^2+1:dim^2+3),[1 dim]);
a0   = vecAs(dim^2+4);


if plot_it == true
    figure(2)
    hold off
end

end


function [f, gradf,k,y,rho]= derivatives(vecas,x,r,n,l,h)
dim  = size(r,2);
s    = vecas(end);
A    = reshape(vecas(1:dim^2),[dim dim]);
a    = reshape(vecas(dim^2+1:dim^2+dim),[1 dim]);
a0   = vecas(dim^2+dim+1);
bias = diag(r*A*r') + r*a' + a0;
y    = (x.*bias);
sp   = kScaleOptimization(y);
d    = pdist2(y,y);
k    = exp(-d.^2/(2*sp.^2));

  if any(isnan(k(:)))% ||any(isnan(k_ly(:)))
    fprintf('whoa')
    f=nan;
    %    gradf=0*logeta;
  else
%     trkl = trace(h*k*h*h*l*h);
%     trkk = trace(h*k*h*h*k*h);    
    trkl=trace(k*h*l*h);
    trkk=trace(k*h*k*h);
    grad_lk=(h*l*h)/trkl;
    grad_k=2*(h*k*h)/trkk;
    grad = grad_lk-.5*grad_k;
    p = grad.*k;
    Dy = bsxfun(@minus, y, y');
    
    grada0 = 0;
    grada  = zeros(1,dim);
    gradA  = zeros(dim,dim);
    
    for i=1:n
    for j=i+1:n
      grada0 = grada0 + 2*p(i,j)*Dy(i,j)*(x(i)-x(j));
      grada  = grada  + 2*p(i,j)*Dy(i,j)*(x(i)*r(i,:)-x(j)*r(j,:));
      gradA  = gradA  + 2*p(i,j)*Dy(i,j)*(x(i)*r(i,:)'*r(i,:)-x(j)*r(j,:)'*r(j,:));
    end
    end
    
    grads = trace((-k.*d.^2)*real(grad));
    grads = s*log(10)*grads; %function of u; s = 10^u    
    gradf = [gradA(:);grada(:);grada0;grads];
    f=-real(log(trkl)-log(trkk)/2);
    rho = trkl/sqrt(trkk);
  end
end





