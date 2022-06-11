function[y_hat,beta] = MCMC(x,y,iter,iter_burnin)
    K = size(x,2);
    T = length(y);
    k_Q = 0.1;
    k_W = 0.1;

    beta_init = (x'*x)\(x'*y);
    sigma_init = mean((y-x*beta_init).^2);
    V_beta_init = sigma_init*inv(x'*x); 

    Q_prmean = ((k_Q)^2)*40*(V_beta_init);           % Q~Inverse-Wishart
    

    W_prmean = ((k_W)^2*2);    % W~Inverse-Wishart

    Q_prvar = (100);
    W_prvar = 2;
    
    sigma_prmean    = sigma_init;                % log(sigma_0) ~ N(log(sigma_OLS),I_n)
    sigma_prvar     = 4;

% Parameters of the 7 component mixture approximation to a log(chi^2) density:
    q_s     = [  0.00730;  0.10556;  0.00002; 0.04395; 0.34001; 0.24566;  0.25750]; % probabilities
    m_s     = [-10.12999; -3.97281; -8.56686; 2.77786; 0.61942; 1.79518; -1.08819]; % means
    u2_s    = [  5.79596;  2.61369;  5.17950; 0.16735; 0.64009; 0.34023;  1.26261]; % variances

%  init value
    Btdraw = repmat(beta_init,[1,T]);
    Sigtdraw = repmat(log(sigma_init),[1,T]);
    Ht = ones(T,1);
    y_Trans   = y';
    Zs = ones(T,1);
    
    Qdraw = eye(K,K)*0.1;
    Wdraw = 0.1;
%   save states
    state_beta = zeros(K,T,iter-iter_burnin,'single');
    state_sigma = zeros(1,T,iter-iter_burnin,'single');
    state_y_hat = zeros(1,T,iter-iter_burnin,'single');
    %% MCMC loop
    for n_iter = 1:iter
        %% Btdrawc is a draw of the mean VAR coefficients, B(t)
        [Btdrawc,~] = carter_kohn(y',x,Ht,Qdraw,K,1,T,beta_init,V_beta_init);
        
        
        Btdraw = Btdrawc;
        
        %=====| Draw Q, the covariance of B(t) (from iWishart): Take the SSE in the state equation of B(t)
        Btemp = Btdraw(:,2:end)' - Btdraw(:,1:end-1)';
        sse_2 = zeros(K,K);
        for i = 1:T-1
        sse_2 = sse_2 + Btemp(i,:)'*Btemp(i,:);
        end
        % ...and subsequently draw Q, the covariance matrix of B(t)
        Qinv        = inv(sse_2 + Q_prmean);
        Qinvdraw   = wish(Qinv,T+Q_prvar);
        Qdraw       = inv(Qinvdraw);            % this is a draw from Q
         % this is a draw from Q
     
        % draw state&sigma
        yhat = zeros(1,T);
        for i = 1:T
            yhat(:,i) = y_Trans(:,i) - x(i,:)*Btdraw(:,i);
        end
        [statedraw,yss,~]           = mcmc_draw_state([],Sigtdraw,yhat,m_s,u2_s,q_s,1,T);
        [Sigtdraw,~,sigt,Wdraw]  = mcmc_draw_sigma(statedraw,Wdraw,yss,Zs,m_s,u2_s,1,T,sigma_prmean,sigma_prvar,W_prmean,W_prvar);
        % update H_t
        Ht      = sigt.^2;
        if n_iter >iter_burnin
            state_beta(:,:,n_iter-iter_burnin) = Btdraw;
            state_sigma(:,:,n_iter-iter_burnin) = Sigtdraw;
            state_y_hat(:,:,n_iter-iter_burnin) = sum(x.*Btdraw',2);
        end
    end
    y_hat = median(state_y_hat,3);
    beta = median(state_beta,3);
end