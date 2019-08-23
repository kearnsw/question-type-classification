# Resource and Response Type Classification for Consumer Health Question Answering

This repository holds the code and data necessary to replicate the results of our paper:

[Kearns, W., Thomas, J. Resource and Response Type Classification for Consumer Health Question Answering. AMIA Annual Symposium 2018.](https://www.semanticscholar.org/paper/Resource-and-Response-Type-Classification-for-Kearns-Thomas/c7d48e167ee789da7869efb702c3fdd2d0035864)

## Abstract 

Health question answering systems often depend on the initial step of question type classification. Practitioners face several modeling choices for this component alone. We evaluate the effectiveness of different modeling choices in both the embeddings and architectural hyper-parameters of the classifier. In the process, we achieve improved performance over previous methods, achieving a new best 5-fold accuracy of 85.3% on the GARD dataset. The contribution of this work is to evaluate the performance of sentence classification methods on the task of consumer health question type classification and to contribute a dataset of 2,882 medical questions annotated for question type.

## Yahoo Answers Data
The license for the Yahoo Answers dataset does not allow for redistribution. To facilitate reproducibility, we have included a crosswalk and top-level script to generate the data used in this study. 
Download the [L6 - Yahoo! Answers Comprehensive Questions and Answers](webscope.sandbox.yahoo.com) corpus. Untar the files then provide the path the to the directory containing the `FullOct2007.xml.part1` and `FullOct2007.xml.part2` files to the following script located at `question-type-classification/data/raw/crosswalk.sh`. As follows:
`bash crosswalk.sh /path/to/L6_Data_Directory/`

This will reconstruct the annotated data in a file called `Yahoo_HealthQA.tsv`.
