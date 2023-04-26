# Loan-Prediction-Model-Using-Python




ALGORITHMS --

Mainly, we need to use the Supervised Machine Learning Algorithms of Classification and regression for Loan Prediction.

1.Decision Tree Classification(Low Accuracy) – 

This algorithm will create Tree with different conditions in Parent Node and gives output in Leaf Node as YES or NO i.e. Is he/she able to approve the loan or not.

Ex.  	Credit History > 0
	Annual Income > 4Lakhs
	Then Only, Approve Loan Otherwise Reject.


2.Naive Bayes Classification(Gives better Accuracy) –

This is the Probabilistic Machine Learning Algorithm based on the Bayes theorem. 
      Formulae- P(A|B) = ( P(B|A) P(A) ) /  P(B)
      P(A|B) – Likelihood of occurrence of A, 
      P(B|A) – Likelihood of occurrence of B, 
      P(A) – probability of Occurrence of A, 
      P(B) – probability of Occurrence of B.  
      
      

To run this model, you need to be installed with python and its path setup with cmd
I had attached the requirements.txt files which contains all libraries need to be install to run the model successfully.
You have to go to this "Loan-Prediction-Model-Using-Python" folder on cmd if you are downloading zip file. 
then, type cmd " pip install requirements.txt " - This will install all libraries into the Python,
If you are using Pycharm, then, you have to install the libraries manually from 'file->settings' menu in pycharm which I am specifying below.

Python Libraries used in this Project – 
1.	Pandas- 
		
		pip install pandas
		
		
2.	NumPy-
	
	 	 pip install numpy
		 
3.	Matplotlib-
	
	 	 pip install matplotlib
		 
4.	Scikit-learn-
	
		pip install scikit-learn
		

If there occurs the error of PREMISSION or any user access is needed while installing libraries - then use this command

		
		pip install library_name --user
		

This model gives you a output in array of 1 & 0 which indicates the yes or no for loan assigning.

I had attached the .pdf which explains all about the Model with the steps which are need to be follow while making model code.
