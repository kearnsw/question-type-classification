# Resource and Response Type Classification for Consumer Health Question Answering

This repository will hold all the code and data necessary to replicate the results of the paper:

```Kearns, W., Thomas, J. Resource and Response Type Classification for Consumer Health Question Answering. AMIA Annual Symposium 2018.```

## Yahoo Answers Data

Download the [L6 - Yahoo! Answers Comprehensive Questions and Answers](webscope.sandbox.yahoo.com) corpus. Untar the files then provide the path the to the directory containing the `FullOct2007.xml.part1` and `FullOct2007.xml.part2` files to the following script located at `question-type-classification/data/raw/crosswalk.sh`. As follows:
`bash crosswalk.sh /path/to/L6_Data_Directory/`

This will reconstruct the annotated data in a file called `Yahoo_HealthQA.tsv`.

