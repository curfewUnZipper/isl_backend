NEW PROCEDURE: (append)trainer.py>cleaner.py>generateModel.py>translator.py

(26-12-24) latest objective: 
				~1. 3d trainer and model generator
				2. get better scores
				3. ~complete report
				4. ~start forming words/sentences - add space character "amitabh bachcan hand sign"
								    add this to artifact.json(get ASCII~ value of space)~




1. to overcome bad hand recognition: https://medium.com/@er_95882/asl-recognition-using-pointnet-and-mediapipe-f2efda78d089
		recognise whole body -> then guess hands -> then recognise hand landmarks
2 add Kaggle datasets for training as well
3. ~add WHO data and references~ 
	   
objectives for me: ~1. check for 'CONFIDENCE OF EACH RESULT'
				   2. Explore more models (refer mohd's yt vid on sports classification)

	~i left it at - printed the confsion MATRIX using jupyter
	~i was using files - codeTesting and codeTesting2
	~i will need to use CONFIDENCE of result to enhance the output
	~later on when it identifies the letters properly, will break it into words using GOOGLE AUTOCORRECT ahh
	// (maybe explore more models for this) work on the "difference based" model
	---------more agendas - 16/10/24-------------
	i need yashwant to complete all alphabets again with BETTER SCORE



File Information:
~OLD PROCEDURE: generateModel.py>codeTesting2.py~
trainer.py - to train the model, to add new values to csv for letter
generateModel.py - copied jupy to here to generate and save .pkl and .json
codeTesting - contains fn to create Result csv, contains the TIME TO RUN the code
codeTesting2 - to create result csv and give out final result
mediaPipeModel.ipynb - CONFUSION MATRIX
TrainingLog.txt - contains no. training data per alphabet

jupyter/mediaPipeModel - to train and create .pickle and .json
jupyter/miniProject - to use the model #final file


1. mp{x}: current testing iteration

2. trainer.py: to create csv/excel files with 
			   coordinates of each hand-point
	appendTrainer.py: to append training data to existing letter
3. GeneratingModel.ipynb: module to generate the pickel file a.k.a
		          TRAINED MODEL

4. Output.ipynb: THE FILE a.k.a we use the model to make 
	         predictions in this file


--------------

July Review References:


1. ML youtube - https://www.youtube.com/@machinelearning4806/videos
2. https://www.simplilearn.com/10-algorithms-machine-learning-engineers-need-to-know-article

3. https://realpython.com/null-in-python/
4. < add csv python site >

--------------

July Review References:

1. https://realpython.com/null-in-python/
2. < add csv python site >
3. 



-------------------------------
Bible for this #GoogleMP Documentation: https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python

-------------------
"modelVenv" is the virtualEnvironment for this project
to use virtualEnv: command1: cd to modelVenv/Scripts 
	           command2: activate
-------------------
pranav bansal used google mediapipe for the same project:
https://bansal-pranav.medium.com/indian-sign-language-recognition-using-googles-mediapipe-framework-3425ddce6748

MediaPipe tutorials:
#https://www.assemblyai.com/blog/mediapipe-for-dummies/
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

pythonGuide:
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
Notebook: https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

--------------
#venv Creation: 
1. python -m venv env_name // "D:/softwares/Python 3.12.4/python.exe" -m venv modelVenv
2. cd Scripts -> activate.bat
3. pip list #to list packages installed
4. to create requirements.txt : pip freeze > requirements.txt
5. to install from requirements.txt : pip install -r requirements.txt
6. Deactivate : deactivate
---------------


good Sites for Sign Pics:
1. Baby ISL (same as ISL): http://indiandeaf.org/Baby%20ISL/Category/Alphabet-D
2. artistic pics: https://www.talkinghands.co.in/sites/default/files/styles/max_650x650/public/2021-08/A_0.jpg?itok=voZFFgsW

---------------


Helpful Sites:
1. trainingData: https://www.semanticscholar.org/paper/Indian-Sign-Language-Character-Recognition-Lakshmi-GeethikaM/b69ae91a8a5a414be689da8e4607e59994c2ca29
4. ISL Basics: https://www.youtube.com/watch?v=qcdivQfA41Y&ab_channel=PragyaGupta
6. ISLRTC Drive: https://drive.google.com/drive/folders/1U-Pr4r1-cupgNOOq9NH_uTsQnPSVEKco

