# EE554 - Lab3 : Hidden Markov Models (HMMs)

This is a tutorial on hidden Markov models (HMMs). It introduces the core problems
of HMMs and how to solve them with the forward and Viterbi algorithms for the
example task of classifying vowel sequences.

## Running the Notebook on Noto

### Cloning the repo
1. Go to Git > Clone a Repository
2. Enter the URL for the Repo: https://github.com/KarlHajal/EE554-Lab3
   
### Creating the kernel
1. Open a terminal: File > New > Terminal
2. Go the project's directory: ```cd EE554-Lab3/```
3. Run the following command to create the kernel: ```kbuilder_create EE554_Lab3 requirements.txt```

### Activating the kernel
1. Refresh your browser
2. Open the notebok (hmm_lab.ipynb)
3. Go to Kernel > Change Kernel
4. Choose EE554_Lab3

The notebook is now ready to run!

## Running the Notebook Locally

### Create conda environment

```bash
conda env create -f environment.yml
conda activate hmm_lab
```

### Launch the notebook

```bash
jupyter notebook hmm_lab.ipynb
```

Answers to the exercises in the notebook are provided in the PDF.

## Acknowledgements

This lab was originally developed by Sacha Krstulović, Hervé
Bourlard, Hemant Misra, and Mathew Magimai-Doss for the *Speech Processing and
Speech Recognition* course at École polytechnique fédérale de Lausanne (EPFL).
The original Matlab version is available here:
http://publications.idiap.ch/index.php/publications/show/739
