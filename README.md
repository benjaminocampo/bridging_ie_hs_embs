# [Unmasking the Hidden Meaning: Bridging Implicit and Explicit Hate Speech Embedding Representations](https://aclanthology.org/2023.findings-emnlp.441)

"Unmasking the Hidden Meaning: Bridging Implicit and Explicit Hate Speech Embedding Representations." This repository contains the datasets and experiments conducted as part of the research.

## Getting Started

To use this repository, please follow the instructions below:

1. First, run the following command to create a Conda environment with the name of the repository:

   ```bash
   make create_environment
   ```

   This command will set up a Conda environment specific to this project.

2. Activate the created environment using the following command:

   ```bash
   conda activate <name_of_the_environment>
   ```

   Replace `<name_of_the_environment>` with the name of the environment created in the previous step.

3. Install the required dependencies by running the following command:

   ```bash
   make requirements
   ```

   This command will install all the necessary packages and libraries needed to run the experiments.

## Repository Structure

The repository is structured as follows:

- **`data`**: This directory contains the datasets used in the research. You can find the relevant data files here.

- **`experiments`**: This directory contains the experiments conducted to address different research questions. It is further divided into subdirectories:

  - **`experiments/RQ1`**: Contains the experiments related to Research Question 1.
  - **`experiments/RQ2`**: Contains the experiments related to Research Question 2.
  - **`experiments/RQ3`**: Contains the experiments related to Research Question 3.

Feel free to explore the repository and access the datasets and experiment results as needed.

Full results can be found here: https://docs.google.com/spreadsheets/d/1vnbpX4I11L489gp-p1sliaW7VUOy6d4HPgQE9veqVl4/edit?usp=sharing

# Contributing

We are thrilled that you are interested in contributing to our work! Your
contributions will help to make our project even better and more useful for the
community.

Here are some ways you can contribute:

- Bug reporting: If you find a bug in our code, please report it to us by
  creating a new issue in our GitHub repository. Be sure to include detailed
  information about the bug and the steps to reproduce it.

- Code contributions: If you have experience with the technologies we are using
  and would like to contribute to the codebase, please feel free to submit a
  pull request. We welcome contributions of all sizes, whether it's a small bug
  fix or a new feature.

- Documentation: If you find that our documentation is lacking or could be
  improved, we would be grateful for your contributions. Whether it's fixing
  typos, adding new examples or explanations, or reorganizing the information,
  your help is greatly appreciated.

- Testing: Testing is an important part of our development process. We would
  appreciate it if you could test our code and let us know if you find any
  issues.

- Feature requests: If you have an idea for a new feature or improvement, please
  let us know by creating a new issue in our GitHub repository.

All contributions are welcome and appreciated! We look forward to working with
you to improve our project.

# Cite us

@inproceedings{ocampo-etal-2023-unmasking,
    title = "Unmasking the Hidden Meaning: Bridging Implicit and Explicit Hate Speech Embedding Representations",
    author = "Ocampo, Nicolas  and
      Cabrio, Elena  and
      Villata, Serena",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.441",
    pages = "6626--6637",
    abstract = "Research on automatic hate speech (HS) detection has mainly focused on identifying explicit forms of hateful expressions on user-generated content. Recently, a few works have started to investigate methods to address more implicit and subtle abusive content. However, despite these efforts, automated systems still struggle to correctly recognize implicit and more veiled forms of HS. As these systems heavily rely on proper textual representations for classification, it is crucial to investigate the differences in embedding implicit and explicit messages. Our contribution to address this challenging task is fourfold. First, we present a comparative analysis of transformer-based models, evaluating their performance across five datasets containing implicit HS messages. Second, we examine the embedding representations of implicit messages across different targets, gaining insight into how veiled cases are encoded. Third, we compare and link explicit and implicit hateful messages across these datasets through their targets, enforcing the relation between explicitness and implicitness and obtaining more meaningful embedding representations. Lastly, we show how these newer representation maintains high performance on HS labels, while improving classification in borderline cases.",
}
