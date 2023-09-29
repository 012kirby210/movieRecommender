from bs4 import BeautifulSoup
import re

# Beautiful soup syntax cheatsheet

html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Site to scrape</title>
</head>
<body>
    <h1 class="title">Page title</h1>
    <p>We are blabla<br><br>
    <a href="https://developer.mozilla.org/fr/docs/Web/API/Canvas_API" class="documentation" id="canvasapi">Canvas API</a>
    <a href="https://developer.mozilla.org/fr/docs/Web/API/CSS_Object_Model" class="documentation" id=cssom"">CSS OM</a>
    <a href="https://developer.mozilla.org/fr/docs/Web/API/Document_Object_Model" class="documentation doc1 doc2" id="dom">DOM</a>
    </p>
    <p class="description">...</p>
</body>
</html>
'''

soup = BeautifulSoup(html)
# Get a tag
print(soup.title)
# Get a tag name 
print(soup.title.name)
# Get a parent tag
print(soup.title.parent.name)
# Get an attribute
print(soup.h1['class'])
# Get all tags of type
print(soup.findAll('a'))
# Get a element with a selector
print(soup.find(id="canvasapi"))

# Iterate of a set of tags and get some attr from it
for link_tag in soup.find_all('a'):
    print(link_tag.get('href'))

# Get the text nodes
print(soup.get_text())

# Get a dictionnary of attributes 
print(soup.a.attrs)
# it will stop to the first of the dictionnary as a list
print(soup.a['id'])

# Change an attribute 
soup.a['id'] = 'newcanvasid'
print(soup.a['id'])
# Add an attribute
soup.a['data-new'] = 'new data'
print(soup)

# Suppress attribute
del soup.a['data-new']
print(soup)

# get all classes 
print(soup.find(id="dom")['class'])

# handling error
j = soup.find(id="dom")
j['id']=['eh','lo']
print(soup)

# string representation
tag = soup.find(id="eh lo")
print(tag.string)
print(type(tag.string))

for string in soup.strings:
    print(repr(string))

for string in soup.stripped_strings:
    print(repr(string))

# String n'a pas accès aux commentaires html
comment = '<p><!-- In a comment -->!</p>'
s2 = BeautifulSoup(comment, features="html.parser")
print(s2.p.string)

# Get the content data as an array
print(soup.body.contents)

# Get the children
for children in soup.body.children:
    print(children)

# Get the parents
tag = soup.find(id="eh lo")
for parent in tag.parents:
    print(parent)

# Get the siblings
print(tag.next_sibling)
print(tag.previous_sibling)

# With css selector
print(soup.select('.documentation'))

# filtering through regular expression
results = soup.find_all(re.compile('^b'))
for tag in results:
    print(tag.name)

results = soup.find_all(['a','p'])
for tag in results:
    print(tag)