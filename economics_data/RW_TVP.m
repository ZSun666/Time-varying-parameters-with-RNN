addpath('matlab_function')
addpath('econ_data')
tic
%% out of sample prediction
% load ta
data_raw = readtable('econ_data/three_variable.csv');
y_raw = data_raw{1:261,2};
x_raw = data_raw{:,2:4};
x_lag_1 = x_raw(2:261-1,:);
x_lag_2 = x_raw(1:261-2,:);
x_raw = [x_lag_1,x_lag_2,ones(length(y_raw)-2,1)];
y_raw = y_raw(3:261);

T_total = length(y_raw);
L_train = 120;
count_i = 1;
for i = T_total - L_train:T_total-1
    y_true(count_i) = y_raw(i+1,:);
    x_predict = x_raw(i+1,:);
    x_train = x_raw(1:i,:);
    y_train = y_raw(1:i);
    [y_hat,beta] = MCMC(x_train,y_train,2000,200);
    y_predict(count_i) = x_predict*beta(:,end);
    [num2str(count_i),'/',num2str(L_train)]
    count_i = count_i+1;
end
result = [y_predict',y_true'];
save("result/out_sample_pred",'result')
csvwrite("result/out_sample_pred_RW.csv",result)