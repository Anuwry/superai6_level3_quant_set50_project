{'ridge_regression_full_ta':      fold  train_start_year  train_end_year  test_year  n_train  n_test  \
 0  fold_1              2012            2021       2022     2359     241   
 1  fold_2              2012            2022       2023     2600     243   
 2  fold_3              2012            2023       2024     2843     244   
 3  fold_4              2012            2024       2025     3087     234   
 
        rmse       mae      mape        r2  direction_accuracy  
 0  6.936808  5.371006  0.546238  0.902858            0.585062  
 1  7.804107  5.958480  0.644251  0.969236            0.547325  
 2  6.763733  5.026060  0.580785  0.976409            0.471311  
 3  9.952250  7.704858  0.977921  0.961765            0.465812  ,
 'xgboost_full_ta':      fold  train_start_year  train_end_year  test_year  n_train  n_test  \
 0  fold_1              2012            2021       2022     2359     241   
 1  fold_2              2012            2022       2023     2600     243   
 2  fold_3              2012            2023       2024     2843     244   
 3  fold_4              2012            2024       2025     3087     234   
 
         rmse        mae      mape        r2  direction_accuracy  
 0   7.577485   5.940885  0.604670  0.884085            0.506224  
 1   8.628898   6.704670  0.726190  0.962390            0.514403  
 2   8.547372   6.823927  0.793542  0.962327            0.467213  
 3  26.463390  19.478965  2.550055  0.729660            0.538462  ,
 'lightgbm_full_ta':      fold  train_start_year  train_end_year  test_year  n_train  n_test  \
 0  fold_1              2012            2021       2022     2359     241   
 1  fold_2              2012            2022       2023     2600     243   
 2  fold_3              2012            2023       2024     2843     244   
 3  fold_4              2012            2024       2025     3087     234   
 
         rmse        mae      mape        r2  direction_accuracy  
 0   7.337420   5.785724  0.588159  0.891313            0.485477  
 1   7.897610   6.022166  0.650517  0.968495            0.481481  
 2   7.573452   5.805515  0.672106  0.970423            0.520492  
 3  21.494575  15.462884  2.037905  0.821649            0.555556  ,
 'autogluon_full_ta':      fold  train_start_year  train_end_year  test_year  n_train  n_test  \
 0  fold_1              2012            2021       2022     2359     241   
 1  fold_2              2012            2022       2023     2600     243   
 2  fold_3              2012            2023       2024     2843     244   
 3  fold_4              2012            2024       2025     3087     234   
 
         rmse        mae      mape        r2  direction_accuracy  
 0   6.860974   5.336294  0.542475  0.904970            0.543568  
 1   7.705789   5.847046  0.632587  0.970007            0.473251  
 2   6.983918   5.177044  0.598329  0.974848            0.540984  
 3  22.092279  16.366678  2.156902  0.811592            0.538462  ,
 'lstm_full_ta':      fold  train_start_year  train_end_year  test_year  n_train  n_test  \
 0  fold_1              2012            2021       2022     2359     241   
 1  fold_2              2012            2022       2023     2600     243   
 2  fold_3              2012            2023       2024     2843     244   
 3  fold_4              2012            2024       2025     3087     234   
 
         rmse        mae      mape        r2  direction_accuracy  
 0  14.312324  11.055849  1.122358  0.586468            0.535270  
 1  16.397659  13.405587  1.453795  0.864183            0.522634  
 2  13.213970  10.744083  1.253991  0.909960            0.471311  
 3  19.547923  15.645646  1.991374  0.852491            0.478632  ,
 'chronos_t5_tiny_zero_shot_greedy_full_ta_reference':      fold  train_start_year  train_end_year  test_year  n_train  n_test  \
 0  fold_1              2012            2021       2022     2419     241   
 1  fold_2              2012            2022       2023     2660     243   
 2  fold_3              2012            2023       2024     2903     244   
 3  fold_4              2012            2024       2025     3147     234   
 
        rmse       mae      mape        r2  direction_accuracy  
 0  7.143991  5.415079  0.551044  0.896968            0.526971  
 1  7.869250  6.046880  0.653793  0.968721            0.460905  
 2  7.061215  5.463106  0.629777  0.974288            0.442623  
 3  9.424079  7.324703  0.926338  0.965716            0.529915  }

 อันดับ	ชุดข้อมูล / โมเดล	Accuracy เฉลี่ย	SD	ช่วง Accuracy
1	Full Non-TA + LSTM	52.59%	3.61%	48.36–56.02%
2	Full TA + AutoGluon	52.41%	3.39%	47.33–54.36%
3	Full TA + Ridge	        51.74%	5.85%	46.58–58.51%
4	Full Non-TA + AutoGluon	51.48%	1.86%	50.41–54.27%
5	Full TA + LightGBM	51.08%	3.46%	48.15–55.56%
6	Full Non-TA + XGBoost	50.96%	1.69%	49.59–53.42%