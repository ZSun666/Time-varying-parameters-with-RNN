addpath('matlab_function')
addpath('simulated_data')
tic
for i = 0:99
    data_name = strcat('simulated_data/jump_data/data_',num2str(i),'.csv');
    data_raw = readtable(data_name);
    y_train = data_raw{2:end,end};
    x_train = data_raw{2:end,end-3:end-1};
    [y_hat,beta] = MCMC(x_train,y_train,2000,200);
    result = [x_train,y_train,y_hat',beta'];
    writematrix(result,['simulated_data/jump_result/result_',num2str(i),'.csv'])
    toc
    ['time:',num2str(toc/60),'min']
    [num2str(i),'/',num2str(99)]
end