x = randn([200,1]);
y = rand(200,1);
[y_hat,beta] = MCMC(x,y,5000,1000);