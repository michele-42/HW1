import sys
import xml.etree.ElementTree as etree


xml = """
    <feed xml:lang='en'>
        <title>HackerRank</title>
        <subtitle lang='en'>Programming challenges</subtitle>
        <link rel='alternate' type='text/html' href='http://hackerrank.com/'/>
        <updated>2013-12-25T12:00:00</updated>
    </feed>
"""
tree = etree.ElementTree(etree.fromstring(xml))
root = tree.getroot()
print(root.attrib)
for n in root:
    print(n.attrib)
