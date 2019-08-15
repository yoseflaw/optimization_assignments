# Predictive Maintenance Scheduling

The assignment focus on scheduling maintenance with predicted remaining useful lifetime (RUL) of aircraft engines. RUL is predicted with machine learning model through sensors readings. Based on the prediction, we schedule the maintenance such that the cost is minimized.

The complete description of the assignment and solution is presented in **report.pdf**.

This repository of mainly two parts:
1. The prediction part answers the prediction task of the assignment. Files included:
	- prediction.py 		: the scoring function and RUL prediction problem class for modeling
	- prediction_task.ipynb : jupyter notebook that contains the steps for model tuning and selection
	- PT 					: a folder that contains files related to prediction task, including the output file
2. The optimization part answers the otpimization tasks of the assignment. Files included:
	- optimization.py 		: the Optimization object that encapsulates all the components of the optimization problem
	- optimization_task.ipynb: 
		jupyter notebook that contains the steps to solve all the optimization tasks.
		The output schedules are also listed in this notebook.
	- OTx					: folders that contains necessary input files for Optimization Task x

In this project, I work in a group and my main contributions are:
- implementation of the prediction task and the experiment
- involved in the mathematical modeling of the optimization task
- integration of the prediction and the optimization task

Technology: Python, Scikit-learn, Keras, PuLP, LaTEX.