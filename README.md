# Scikit-learn Pipeline and GridSearchCV Example

This repository includes a sample script for getting up and running with Scikit's Pipeline and GridSearchCV objects. Pipelines chain together multiple data transformation and modeling steps that should be performed in a series. GridSearchCV simplifies hyperparameter tuning and allows for a cross validation strategy to be utilized during the model selection process. This basic script builds a multi-class classification machine learning pipeline for use with Scikit's Iris Dataset. Classification is performed with Scikit's standard decision tree classifier.


## Instructions

Create a conda environment named *scikit-learn-pipeline-example* from the provided environment file with the following command.

```
conda env create -f environment.yml
```

Now, we can activate the virtual environment. If you are using Windows, delete "source" from the below command.

```
source activate scikit-learn-pipeline-example
```

Run the script.

```
python pipeline_example.py
```


## License

    Copyright 2019 Michael Signorotti

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.