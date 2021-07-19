# ML-Pipeline

### Instructions to setup and run example:

Step 1: Clone repository <br>
Step 2: Create and activate conda virtual env <br>
Step 3: Install numpy, pandas, scikit-learn, and jupyterlab <br>
Step 4: run 'jupyter notebook' command in top level directory of ML-Pipeline <br>
Step 5: Open example_pipeline.py and example_config.json <br>
Step 6: run the notebook, to make changes edit example_config.json <br>

### Intructions to contribute:

#### How to add a new imputation strategy:
Step 1: Add strategy name and applicable columns to example_config.json under "imputation_strategy" section <br>
Step 2: In preprocess.py add code support for your desired strategy under the impute(...) function <br>
