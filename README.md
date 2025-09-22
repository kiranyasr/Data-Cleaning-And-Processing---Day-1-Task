Titanic Data Cleanup Project
-----------------------------

 => Goal: This script automatically cleans the messy Titanic passenger list and prepares it for data analysis and machine learning.

How This Project is Unique & Professional
-----------------------------------------

 => This project uses methods from professional software engineering to ensure the code is reliable, reusable, and high-quality.

Smart & Reliable Cleaning
-------------------------

 => What it is: It uses a scikit-learn pipeline to handle all data cleaning steps.

 => Why it's better: Instead of cleaning data with manual, error-prone steps, a pipeline acts like a "recipe" that guarantees a perfect, consistent result every single time. This is the standard for professional machine learning.

Advanced Data Creation (Feature Engineering)
---------------------------------------------
 => What it is: The script doesn't just clean the data; it intelligently creates new data columns like FamilySize and passenger Title from the original names.

 => Why it's better: These new features can help a machine learning model discover more powerful patterns, leading to more accurate predictions. It shows a deeper level of data analysis.

Code Testing for Reliability
----------------------------

 => What it is: The project includes a suite of tests using pytest.

 => Why it's better: Tests prove that the code works as expected. This is a critical practice in professional development that ensures the code is trustworthy and reliable.

Organized Code & Automatic Logging
-----------------------------------

 => What they are: The code is split into logical functions, and it keeps a "diary" of its actions in data_processing.log.

 => Why they're better: This makes the code easy to read, manage, and debug, just like in a professional team environment.

How to Run This Project
------------------------
1. Set Up Your Workspace (Only needs to be done once)
This creates a special, isolated "bubble" for your project so it doesn't interfere with anything else on your computer.

# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

2Install the Tools
------------------

This command installs all the necessary libraries at once. Make sure you have a requirements.txt file.

 # install 
pip install -r requirements.txt

3. Run the Script
------------------
Make sure the Titanic-Dataset.csv file is in the same folder and run:

# Run
python preprocess.py

What You Get (The Outputs)
-------------------------
 
 => cleaned_titanic_dataset.csv: The final, clean data file, ready for analysis.

 => outliers_boxplot.png: A chart showing any unusual data points.

 => data_processing.log: The script's diary, detailing every step it took.