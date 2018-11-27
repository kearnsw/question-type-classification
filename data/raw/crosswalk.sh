echo '<uri>' > uri_and_subject
echo '<subject>' >> uri_and_subject

grep -f uri_and_subject $1/FullOct2007.xml.part1 | tr "\n" " " | sed 's|</uri>|\t|g' | sed 's|</subject>|\n|g' | sed 's|<uri>||g' | sed 's|<subject>||g' > cleaned.tsv

grep -f uri_and_subject $1/FullOct2007.xml.part2 | tr "\n" " " | sed 's|</uri>|\t|g' | sed 's|</subject>|\n|g' | sed 's|<uri>||g' | sed 's|<subject>||g' >> cleaned.tsv

python merge.py

