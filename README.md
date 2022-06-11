# Time-varying-parameters-with-RNN

This is the replication code for the time-varying parameters model with RNN(RNN-TVP).

The 'report.pdf' introduces the methodology and the results for simulation experiment and real-world application.

# simulation study

Folder 'simulation/' contains the codes for simulation study.

'simulation_generate_data.py' is the code for generating the simulated data set. It uses the module 'simulate_data' that specifies the data generating processes for parameters. The output of the code is saved to 'simulation/simulated_data/data' as the table with columns: true parameters; x; y

'simulation_estimate.py' is used for estimating the RNN-TVP model. The output is saved to 'simulation/simulated_data/result' as the table with columns: fitted y; true value of y; true value of beta

'simulation_evaluate.py' is used for evaluating the result. It saves the figures to 'simulation/figure'.

# real-world application: predict US GDP

Folder 'economics_data/' contains the codes for predicting US GDP

'econ_out_sample_pred.py' is the code for generating out-of-sample predictions for US GDP. The output is saved to 'economics_data/result/' as the table with columns: predicted value of y; true value of y.

'econ_out_sample_evaluate.py' is the code for evaluating out-of-sample predictions for US GDP. The output is saved to 'economics_data/figure'.
