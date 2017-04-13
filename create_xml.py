
import sys
header_src = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mteval SYSTEM "ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-xml-v1.3.dtd">
<mteval>
<srcset setid="example_set" srclang="English">
<doc docid="doc1" genre="nw">\n"""

footer_src = """</doc>
</srcset>
</mteval>"""


header_tst = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mteval SYSTEM "ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-xml-v1.3.dtd">
<mteval>
<tstset setid="example_set" srclang="English" trglang="Chinese" sysid="sample_system">
<doc docid="doc1" genre="nw">\n"""

footer_tst = """</doc>
</tstset>
</mteval>"""


header_ref = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mteval SYSTEM "ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-xml-v1.3.dtd">
<mteval>
<refset setid="example_set" srclang="English" trglang="Chinese" refid="ref1">
<doc docid="doc1" genre="nw">\n"""

footer_ref = """</doc>
</refset>
</mteval>"""


paragraph = """<p>
<seg id="{0}"> {1} </seg>
</p>\n"""


def main():

    src_fn = "/Users/elliotschumacher/Dropbox/git/Semantics-Final-Project/en-zh/test.en"
    with open(src_fn) as ifile, open(src_fn + ".xml", 'w') as ofile:
        ofile.write(header_src)

        for i, line in enumerate(ifile):
            para_line = paragraph.format(i+1, line.strip())
            ofile.write(para_line)
        ofile.write(footer_src)

    tst_fn = "/Users/elliotschumacher/Dropbox/git/Semantics-Final-Project/en-zh/retrofit_unfrozen.zh"
    with open(tst_fn) as ifile, open(tst_fn + ".xml", 'w') as ofile:
        ofile.write(header_tst)

        for i, line in enumerate(ifile):
            para_line = paragraph.format(i+1, line.strip())
            ofile.write(para_line)
        ofile.write(footer_tst)

    ref_fn = "/Users/elliotschumacher/Dropbox/git/Semantics-Final-Project/en-zh/test.zh"

    with open(ref_fn) as ifile, open(ref_fn + ".xml", 'w') as ofile:
        ofile.write(header_ref)

        for i, line in enumerate(ifile):
            para_line = paragraph.format(i+1, line.strip())
            ofile.write(para_line)
        ofile.write(footer_ref)




if __name__ == '__main__': main()
