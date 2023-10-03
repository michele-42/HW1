###################### Introduction ##########################

# 1. Say "Hello, World!" with Python
print('Hello world')


# 2. Python If-Else
import math
import os
import random
import re
import sys

n = int(input().strip())

if n%2==1: 
    print("Weird")
elif n >= 2 and n <= 6:
    print("Not Weird")
elif n >= 6 and n <=20:
    print("Weird")
elif n > 20:
    print("Not Weird")



# 3. Arithmetic operators
a = int(input())
b = int(input())

print(a + b)
print(a - b)
print(a * b)


# 4. Division and Integer division
a = int(input())
b = int(input())

print (a//b)
print (a/b)


# 5. Loops
n = int(input())

for i in range(n):
    print(i*i)


# 6. Write a function
def is_leap(year):
    leap = False
    if (year%4 == 0 and (year%100 != 0 or (year%100 == 0 and year%400 == 0))):
        leap = True
    return leap

year = int(input())
print(is_leap(year))


#7. Print function 
n = int(input())

s = ""
for x in range(n):
    s += str(x+1)

print(s)


###################### Data Types ##########################

# 1. Find the Runner-Up Score!  (O(n))
n = int(input())
arr = map(int, input().split())

print(max([x for x in arr if x < max(arr)]))


# 2. Nested Lists
records = []
for _ in range(int(input())):
    name = input()
    score = float(input())
    records.append([name, score])

scores = [record[1] for record in records]
second_min_score = min([score for score in scores if score > min(scores)])
sorted_names = sorted([record[0] for record in records if record[1] == second_min_score])

for name in sorted_names:
    print(name)


# 3. Find the percentege
n = int(input())
student_marks = {}
for _ in range(n):
    name, *line = input().split()
    scores = list(map(float, line))
    student_marks[name] = scores
query_name = input()

marks = student_marks[query_name]
print("%.2f" % (sum(marks)/len(marks)))


# 4. Lists
lis = []
N = int(input())
for _ in range(N):
    command = input().split()
    op = command[0]
    if op == 'insert':
        index = int(command[1])
        value = int(command[2])
        lis.insert(index, value)
    elif op == 'print':
        print(lis)
    elif op == 'remove':
        value = int(command[1])
        if value in lis: lis.remove(value)
    elif op == 'append':
        value = int(command[1])
        lis.append(value)
    elif op == 'sort':
        lis.sort()
    elif op == 'pop':
        if len(lis) > 0: lis.pop()
    elif op == 'reverse':
        lis.reverse()


# 5. Tuples

n = int(input())
integer_list = map(int, input().split())
t = tuple(integer_list)
    
print(hash(t))


# 6. List Comprehensions

x = int(input())
y = int(input())
z = int(input())
n = int(input())

print([[i, j, k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k != n])



###################### Strings ##########################

# 1. String split and join

def split_and_join(line):
    return '-'.join(line.split())

line = input()
result = split_and_join(line)
print(result)


# 2. What's your name

def print_full_name(first, last):
    print("Hello " + first + " " + last + "! You just delved into python.")

first_name = input()
last_name = input()
print_full_name(first_name, last_name)


# 3. Mutations

def mutate_string(string, position, character):
    return string[:position] + character + string[position+1:]

s = input()
i, c = input().split()
s_new = mutate_string(s, int(i), c)
print(s_new)


# 4. Find a string

def count_substring(string, sub_string):
    count = 0
    l_str = len(string)
    l_subs = len(sub_string)
    
    for i in range(l_str - l_subs + 1):
        if string[i:i+l_subs] == sub_string:
            count += 1
    
    return count

string = input().strip()
sub_string = input().strip()
count = count_substring(string, sub_string)
print(count)


# 5. String validators

s = input()
    
has_alpha_num = False
has_alpha = False
has_digits = False
has_lower = False
has_upper = False
    
for char in s:
    has_alpha_num = has_alpha_num or char.isalnum()
    has_alpha = has_alpha or char.isalpha()
    has_digits = has_digits or char.isdigit()
    has_lower = has_lower or char.islower()
    has_upper = has_upper or char.isupper()
    
print(has_alpha_num)
print(has_alpha)
print(has_digits)
print(has_lower)
print(has_upper)



# 6. Text Alignment

thickness = int(input()) #This must be an odd number
c = 'H'

for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))




# 7. Text Wrap

def wrap(string, max_width):
    s = ""
    for i in range(0, len(string), max_width):
        end = min(len(string), i+max_width)
        s += string[i:end]+"\n"
    return s

string, max_width = input(), int(input())
result = wrap(string, max_width)
print(result)




# 8. Designer Door Mat
hw = list(map(int, input().split()))

height = hw[0]
width = hw[1]

row = ['-' for _ in range(width)]
mat = ["" for _ in range(height)]
 
mid_x = width // 2
mid_y = height // 2

message = "WELCOME"
pref_suf = "-"*((width - len(message)) // 2)
mat[mid_y] = pref_suf + message + pref_suf

for y in range(mid_y):
    row[mid_x + y*3] = '|'
    row[mid_x-1 + y*3] = '.'
    row[mid_x+1 + y*3] = '.'
    row[mid_x - y*3] = '|'
    row[mid_x-1 - y*3] = '.'
    row[mid_x+1 - y*3] = '.'
    mat[y] = "".join(row)
    mat[-y-1] = "".join(row)

for x in mat:
    print(x)

# 9. String formatting
def print_formatted(number):
   max_binary_value = bin(number)[2:]
   format_string = "% " + str(len(max_binary_value)) + "s"
   for n in range(1, number+1):
       binary_value = bin(n)[2:]
       hex_value = hex(n)[2:].upper()
       oct_value = oct(n)[2:]
       dec_value = str(n)
       print(format_string % dec_value + " " + format_string % oct_value + " " + format_string % hex_value + " " + format_string % binary_value)
   return 0

n = int(input())
print_formatted(n)


# 10. Alphabet Rangoli
def print_rangoli(size):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    letters = list(alphabet[:size])
    
    m = ["" for _ in range(size+size-1)]
    string = "-".join(list(reversed(letters[1:])) + letters)
    middle = size - 1
    m[middle] = string
    for i in range(size-1):
        string = "--" + string[:len(string)//2 - 2] + string[len(string)//2 + 2:] + "--"
        m[middle - i - 1] = string
        m[middle + i + 1] = string
        
    for x in m:
        print(x)
   
n = int(input())
print_rangoli(n)


# 11. Capitalize
def solve(s):
    last_character = " "
    characters = list(s)

    for i in range(len(characters)):
        character = characters[i]
        if last_character == " " and character != " ":
            characters[i] = character.upper()
        last_character = character

    return "".join(characters)

print(solve(input()))


# 12. The Minnion Game

def minion_game(string):
    vowels = "AEIOU"
    consonants = "BCDFGHJKLMNPQRSTVWXYZ"
    stuart_score = 0
    kevin_score = 0
    for i in range(len(string)):
        
        if string[i] in vowels:
            kevin_score += (len(string) - i)
            
        elif string[i] in consonants:
            stuart_score += (len(string) - i)
            
    if kevin_score > stuart_score:
        print("Kevin " + str(kevin_score))
    
    elif stuart_score > kevin_score:
        print("Stuart " + str(stuart_score))
        
    else:
        print("Draw")
        
s = input()
minion_game(s)


# 13. Merge the Tools!
def merge_the_tools(string, k):
    
    for i in range(0, len(string), k):
        subs = string[i:i+k]
        s = ""
        for char in subs:
            if char not in s:
                s += char
        print(s)

string, k = input(), int(input())
merge_the_tools(string, k)


# 14 sWAP cASE
def swap_case(s):
    l = list(s)
    for i in range(len(l)):
        if l[i].isupper():
            l[i] = l[i].lower()
        elif l[i].islower():
            l[i] = l[i].upper()
    return "".join(l)

s = input()
result = swap_case(s)
print(result)





###################### Sets ##########################


# 1. Introduction to sets 
def average(array):
    return sum(set(array))/len(set(array))

n = int(input())
arr = list(map(int, input().split()))
result = average(arr)
print(result)


# 2. Symmetric difference
m = int(input())
s1 = set(map(int, input().split()))
n = int(input())
s2 = set(map(int, input().split()))
l = sorted(list(s1.union(s2).difference(s1.intersection(s2))))
for x in l:
    print(x)


# 3. No Idea!
n, m = tuple(map(int, input().split()))
arr = list(map(int, input().split()))
s1 = set(map(int, input().split()))
s2 = set(map(int, input().split()))
score = 0
for e in arr:
    if e in s1:
        score += 1
    elif e in s2:
        score -= 1
print(score)


# 4. Set add
n = int(input())
print(len(set([input() for _ in range(n)])))


# 5. Set.discard().remove() & pop()
n = int(input())
s = set(map(int, input().split()))
no_commands = int(input())
for _ in range(no_commands):
    command = input().split()
    if command[0] == 'remove' and (int(command[1]) in s):
        s.remove(int(command[1]))
    elif command[0] == 'discard':
        s.discard(int(command[1]))
    elif command[0] == 'pop' and len(s) > 0:
        s.pop()
print(sum(s))


# 6. Set.union() operation
m = int(input())
s1 = set(map(int, input().split()))
n = int(input())
s2 = set(map(int, input().split()))
print(len(s1.union(s2)))


# 7. Set.intersection() operation
m = int(input())
s1 = set(map(int, input().split()))
n = int(input())
s2 = set(map(int, input().split()))
print(len(s1.intersection(s2)))


# 8. Set.difference() operation
m = int(input())
s1 = set(map(int, input().split()))
n = int(input())
s2 = set(map(int, input().split()))
print(len(s1.difference(s2)))


# 9. Set.symmetric_difference() operation
m = int(input())
s1 = set(map(int, input().split()))
n = int(input())
s2 = set(map(int, input().split()))
print(len(s1.symmetric_difference(s2)))



# 10. Set Mutations
n = int(input())
s = set(map(int, input().split()))
n_sets = int(input())
for x in range(n_sets):
    command, no_elem = tuple(input().split())
    s1 = set(map(int, input().split()))
    if command == "intersection_update":
        s.intersection_update(s1)
    elif command == "update":
        s.update(s1)
    elif command == "difference_update":
        s.difference_update(s1)
    elif command == "symmetric_difference_update":
        s.symmetric_difference_update(s1)

print(sum(s))




# 11. The captain's room
k = int(input())
l = list(map(int, input().split()))

first_find = set()
second_find = set()

for x in l:
    if x in first_find:
        second_find.add(x)
    first_find.add(x)

print(list(first_find.difference(second_find))[0])



# 12. Check subset
cases = int(input())
for _ in range(cases):
    n1 = int(input())
    s1 = set(map(int, input().split()))
    n2 = int(input())
    s2 = set(map(int, input().split()))
    
    print(s1.issubset(s2))



# 13. Check strick superset
s = set(map(int, input().split()))
no_sets = int(input())
superset = True

for _ in range(no_sets):
    s1 = set(map(int, input().split()))
    if not s.issuperset(s1) or not s.difference(s1):
        superset = False

print(superset)


###################### Collections ##########################



# 1. collections.Counter()
from collections import Counter

n = int(input())
counter = Counter(map(int, input().split()))
n_cli = int(input())

tot = 0
for _ in range(n_cli):
    shoe_size, price = tuple(map(int, input().split()))
    no_shoes = counter[shoe_size]
    if no_shoes > 0:
        tot += price
        counter[shoe_size] = no_shoes-1
    
print(tot)



# 2. collections.namedtuple()
from collections import namedtuple
n = int(input())
Student = namedtuple('Student', ",".join(input().split()))
print("%.2f" % (sum([int(Student(*tuple(input().split())).MARKS) for _ in range(n)])/n))



# 3. collections.OrderedDict
from collections import OrderedDict

n = int(input())
d = OrderedDict()

for _ in range(n):
    order = list(input().split())
    product_name = " ".join(order[:-1])
    product_price = int(order[-1])
    if product_name in d.keys():
        d[product_name] += product_price
    else:
        d[product_name] = product_price

for key in d.keys():
    print(" ".join([key, str(d[key])]))
    


# 4. Word Order
from collections import OrderedDict

n = int(input())
d = OrderedDict()
s = set()

for _ in range(n):
    string = input()
    s.add(string)
    if string in d.keys():
        d[string] += 1
    else:
        d[string] = 1


print(len(s))
print(" ".join(map(str, d.values())))



# 5. Collections.deque()
from collections import deque

n = int(input())
d = deque()
for _ in range(n):
    command = list(input().split())
    op = command[0]
    if op == 'append':
        d.append(command[1])
    elif op == 'pop':
        d.pop()
    elif op == 'popleft':
        d.popleft()
    elif op == 'appendleft':
        d.appendleft(command[1])

print(" ".join(d))



# 6. Pilling Up!
from collections import deque
n = int(input())


for _ in range(n):
    
    stackable = True
    last_cube_size = 2**31
    
    no_cubes = int(input())
    cubes_sizes_deque = deque(map(int, input().split()))
    
    while len(cubes_sizes_deque) > 1:

        if cubes_sizes_deque[0] > cubes_sizes_deque[-1]:
            current_cube_size = cubes_sizes_deque.popleft()
        else:
            current_cube_size = cubes_sizes_deque.pop()
        
        if current_cube_size > last_cube_size:
            stackable = False
            break
            
        last_cube_size = current_cube_size
    
    print("Yes" if stackable else "No")



# 7. Company Logo!
l = list(input())
l1 = sorted([[c, l.count(c)] for c in set(l)], key=lambda x: (-x[1], x[0]))[:3]
for x in l1:
    print(x[0] + " " + str(x[1]))



# 8. DefaultDict Tutorial
from collections import defaultdict

n, m = tuple(map(int, input().split()))
A = [input() for _ in range(n)]
B = [input() for _ in range(m)]

d = defaultdict(list)

for i in range(len(A)):
    a = A[i]
    d[a].append(str(i+1))

for b in B:
    if d[b]:
        print(" ".join(d[b]))
    else:
        print("-1")






###################### Date Time ##########################



# 1. Calendar Module
import calendar
month, day, year = list(map(int, input().split()))
days_of_week = ["MONDAY", "TWESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
print(days_of_week[calendar.weekday(year, month, day)])



# 2. Time Delta
from datetime import datetime

def time_delta(t1, t2):
    date1 = datetime.strptime(t1, "%a %d %b %Y %H:%M:%S %z").timestamp()
    date2 = datetime.strptime(t2, "%a %d %b %Y %H:%M:%S %z").timestamp()
    
    return str(abs(int(date1 - date2)))



###################### Exceptions ##########################

# 1. Exceptions problem
n = int(input())

test_cases = [input().split() for _ in range(n)]

for test_case in test_cases:
    a = test_case[0]
    b = test_case[1]
    try:
        print(int(a)//int(b))
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e)


        
###################### Built-ins ##########################


# 1. Zipped
no_students, no_subjects = list(map(int, input().split()))
subjects_grades = []

for _ in range(no_subjects):
    subjects_grades.append(list(map(float, input().split())))

for stud_grades in zip(*subjects_grades):
    print(sum(stud_grades)/no_subjects)



# 2. Python-sort-sort

nm = input().split()
n = int(nm[0])
m = int(nm[1])
arr = []
for _ in range(n):
    arr.append(list(map(int, input().rstrip().split())))
k = int(input())
    
l = sorted(arr, key=lambda x: x[k])
for x in l:
    print(" ".join(map(str, x)))


# 3. ginorts    
s = list(input())

lower_case = sorted(filter(lambda x: x.islower(), s))
upper_case = sorted(filter(lambda x: x.isupper(), s))
odd_numbers = sorted(filter(lambda x: x.isnumeric() and int(x) % 2 == 1, s))
even_numbers = sorted(filter(lambda x: x.isnumeric() and int(x) % 2 == 0, s))

print("".join(lower_case + upper_case + odd_numbers + even_numbers))



###################### Python Functionals ##########################


# 1. map and lambda expression

cube = lambda x: x*x*x

def fibonacci(n):
    if n == 0:
        return []
    if n == 1:
        return [0]
    fib = [0, 1]
    for _ in range(n-2):
        fib.append(fib[-1] + fib[-2])
    return fib

n = int(input())
print(list(map(cube, fibonacci(n))))




################ Regex and Parsing ##################

# 1. Re.split()

regex_pattern = r"[,.]"	# Do not delete 'r'.

import re
print("\n".join(re.split(regex_pattern, input())))

### DO LATER






###################### XML ##########################


# XML 1 - Find the Score
import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    count = len(node.attrib)
    for child in node:
        count += get_attr_number(child)
    return count


if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))



# XML 1 - Find the maximum depth
maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    if level > maxdepth:
        maxdepth = level
    for child in elem:
        depth(child, level)
        

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)








########### Closures and decorations ###############



# 1. Standardize mobile number using decorators
def wrapper(f):
    def fun(l):
        l1 = []
        for x in l:
            if len(x) == 10:
                s = "+91 " + x[:5] + " " + x[5:]
            elif x[:3] == "+91":
                s = x[:3] + " " + x[3:8] + " " + x[8:]
            elif x[:2] == "91":
                s = "+" + x[:2] + " " + x[2:7] + " " + x[7:]
            elif x[0] == "0":
                s = "+91 " + x[1:6] + " " + x[6:]
            l1.append(s)
        f(l1)

    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 



# 2. Name directory
def person_lister(f):
    def inner(people):
        for i in range(len(people)):
            people[i].append(i)
        people.sort(key=lambda x: (int(x[2]), x[4]))
        return list(map(f, people))
            
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')




###################### Numpy ##########################



# 1. Arrays
import numpy

def arrays(arr):
    a = numpy.array(arr, float)
    return str(numpy.flip(a))

arr = input().strip().split(' ')
result = arrays(arr)
print(result)



# 2. Shape and reshape
a = numpy.array(list(map(int, input().split())))
a.shape = (3, 3)
print(a)



# 3. Transpose and Flatten
n, m = list(map(int, input().split()))
a = numpy.array([list(map(int, input().split())) for _ in range(n)])
print(numpy.transpose(a))
print(a.flatten())



# 4. Concatenate
n, m, p = list(map(int, input().split()))
array_1 = numpy.array([list(map(int, input().split())) for _ in range(n)])
array_2 = numpy.array([list(map(int, input().split())) for _ in range(m)])
print(numpy.concatenate((array_1, array_2), axis=0))



# 5. Zeroes and ones
t = tuple(map(int, input().split()))
print(numpy.zeros(t, dtype=int))
print(numpy.ones(t, dtype=int))



# 6. Eye and Identity
numpy.set_printoptions(legacy='1.13')
m, n = list(map(int, input().split()))
print(numpy.eye(m, n, k=0))



# 7. Array Mathematics
n, m = list(map(int, input().split()))
a = numpy.array([list(map(int, input().split())) for _ in range(n)])
b = numpy.array([list(map(int, input().split())) for _ in range(n)])
print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a % b)
print(a**b)


# 8. Floor, Ceil and Rint
numpy.set_printoptions(legacy='1.13')
arr = numpy.array(list(map(float, input().split())))
print(numpy.floor(arr))
print(numpy.ceil(arr))
print(numpy.rint(arr))



# 9. Sum and Prod
n, m = list(map(int, input().split()))
arr = numpy.array([list(map(int, input().split())) for _ in range(n)])
print(numpy.prod(numpy.sum(arr, axis=0)))



# 10. Min and Max
n, m = list(map(int, input().split()))
arr = numpy.array([list(map(int, input().split())) for _ in range(n)])
print(numpy.max(numpy.min(arr, axis=1)))



# 11. Mean Var and Std
n, m = list(map(int, input().split()))
arr = numpy.array([list(map(int, input().split())) for _ in range(n)])
print(numpy.mean(arr, axis=1))
print(numpy.var(arr, axis=0))
print(round(numpy.std(arr, axis=None), 11))



# 12. Inner and Outer 
a = numpy.array(list(map(int, input().split())))
b = numpy.array(list(map(int, input().split())))

print(numpy.inner(a, b))
print(numpy.outer(a, b))



# 13. Polynomials
coef = numpy.array(list(map(float, input().split())))
x = int(input())

print(numpy.polyval(coef, x))



# 14. Linear Algebra
n = int(input())
arr = numpy.array([list(map(float, input().split())) for _ in range(n)])

print(round(numpy.linalg.det(arr), 2))



# 15. Dot and Cross
n = int(input())
A = numpy.array([list(map(int, input().split())) for _ in range(n)])
B = numpy.array([list(map(int, input().split())) for _ in range(n)])

TB = numpy.transpose(B)
M = numpy.zeros((n, n), dtype=int)

for i in range(n):
    for j in range(n):
        M[i][j] = numpy.dot(A[i], TB[j])

print(M)


################## Birthday Cake Candles ########################

def birthdayCakeCandles(candles):
    return candles.count(max(candles))



######################## Cangaroo ###############################

def kangaroo(x1, v1, x2, v2):
    if v1 == v2:
        return "NO"
        
    a = (x1 - x2) % (v2 - v1)
    b = (x1 - x2) / (v2 - v1)
    
    return "YES" if a == 0 and b > 0 else "NO"


######################## Cangaroo ###############################











