from Preprocessing import process_text

file = open('temp.md',mode='r')

text_list = []
# read all lines at once
lines = file.readlines()

simple = ''

for line in lines:
    simple = simple+' '+line
    #text_list.append(line)
print(simple)
text, vocab = process_text(simple, lower=True)
print(text)
#
# x = 1
#
# for line in text:
#     print(x)
#     print(line)
#     x+=1
