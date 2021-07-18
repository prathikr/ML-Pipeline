# ML-Pipeline

### Instructions to setup and run example:

Step 1: Clone repository
Step 2: Create and activate conda virtual env
Step 3: Install numpy, pandas, scikit-learn, and jupyterlab
Step 4: run 'jupyter notebook' command in top level directory of ML-Pipeline
Step 5: Open example_pipeline.py and example_config.json
Step 6: run the notebook, to make changes edit example_config.json

### Intructions to contribute:

#### How to add a new imputation strategy:
Step 1: Add strategy name and applicable columns to example_config.json under "imputation_strategy" section
Step 2: In preprocess.py add code support for your desired strategy under the impute(...) function
