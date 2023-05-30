# ALTLipomaNet

This repository contains the code implementation for the research paper titled "Development of 2D and 3D Deep Learning Approaches to Differentiate Atypical Lipomatous Tumors from Lipomas with Magnetic Resonance Imaging and Comparison with Radiologists." The aim of this study was to develop and evaluate 2D and 3D deep learning (DL) approaches to differentiate atypical lipomatous tumors (ALTs) from lipomas using MRI.

## Abstract
The accurate differentiation between atypical lipomatous tumors (ALTs) and lipomas is crucial for effective diagnosis and treatment planning. In this study, we developed and evaluated 2D and 3D deep learning models to differentiate ALTs from lipomas using magnetic resonance imaging (MRI). The performance of the deep learning models was compared with the assessments made by radiology residents and attending-level radiologists, who were blinded to histological and clinical data.

## Dataset
The dataset used in this study consisted of MRI scans from a total of 109 patients, including 64 lipoma cases and 45 ALT cases. The diagnosis of lipomas and ALTs was confirmed through histology and analysis of the murine double minutes (MDM2) gene, serving as the reference standard. Preoperative MRI scans included T2 weighted (T2w) and T1 weighted (T1w) sequences, as well as a fat-suppressed contrast-enhanced T1w (T1fsgd) sequence.

## Methodology
To develop and validate the deep learning models, we performed a 3-fold cross-validation on the dataset. The development cohort consisted of a randomly selected 80% of the patients, while the remaining 20% of patients were reserved for final testing. The models utilized both 2D and 3D approaches based on residual neural networks.


## Usage
To use the code provided in this repository, please follow the instructions below:

1. Clone the repository:

`git clone https://github.com/TimAlexander/ALTLipomaNet/`

2. Navigate to the cloned repository:

`cd ALTLipomaNet`

3. Install Pipenv, if not already installed:

`pip install pipenv`

4. Install the project dependencies:

`pipenv install`

5. Activate the virtual environment:

`pipenv shell`
